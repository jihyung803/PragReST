from __future__ import annotations

import gc
import os
import time
from typing import Any, Callable

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.thinking_tags import (
    contains_thinking_close,
    normalize_early_stopping_text,
    split_after_last_thinking_close,
)

from .vllm_boxed_mcqa import (
    DEFAULT_EARLY_STOPPING_TEXT,
    _tokenizer_extra_kwargs,
    extract_boxed_answer,
    is_gemma4_model_spec,
    is_ministral3_model_spec,
    resolve_model_spec,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def configure_transformers_determinism(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if _env_flag("TRANSFORMERS_EVAL_DETERMINISTIC", True):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _resolve_torch_dtype(dtype_raw: str) -> torch.dtype:
    raw = str(dtype_raw or "float32").strip().lower()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported TRANSFORMERS_EVAL_DTYPE: {dtype_raw}")
    return mapping[raw]


def _load_model(
    *,
    model_path: str | dict[str, Any],
    torch_dtype: torch.dtype,
    device_map: str | None,
    device: str,
) -> tuple[Any, Any, dict[str, str | None]]:
    resolved = resolve_model_spec(model_path)
    base_model = str(resolved["base_model"])
    tokenizer_name = str(resolved["tokenizer"])
    lora_adapter = resolved["lora_adapter"]

    if is_ministral3_model_spec(model_path):
        raise RuntimeError(
            "Current transformers backend cannot load Ministral-3 in this environment. "
            "Use EVAL_BACKEND=vllm."
        )

    if is_gemma4_model_spec(model_path):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            **_tokenizer_extra_kwargs(model_path),
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    attn_implementation = os.environ.get("TRANSFORMERS_EVAL_ATTN_IMPL", "").strip()
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if device_map:
        model_kwargs["device_map"] = device_map

    if is_gemma4_model_spec(model_path):
        from transformers import Gemma4ForConditionalGeneration

        model = Gemma4ForConditionalGeneration.from_pretrained(base_model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if lora_adapter:
        model = PeftModel.from_pretrained(
            model,
            str(lora_adapter),
            is_trainable=False,
            autocast_adapter_dtype=False,
        )

    if not device_map:
        model = model.to(device)

    model.eval()
    return model, tokenizer, resolved


def _prepare_inputs(
    tokenizer,
    prompt_batch: list[str],
    *,
    max_prompt_length: int,
    device: str,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max(1, int(max_prompt_length)),
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def _generate_batch(
    *,
    model,
    tokenizer,
    prompt_batch: list[str],
    max_prompt_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    device: str,
    skip_special_tokens: bool = True,
) -> list[str]:
    inputs = _prepare_inputs(
        tokenizer,
        prompt_batch,
        max_prompt_length=max_prompt_length,
        device=device,
    )
    input_len = int(inputs["input_ids"].shape[1])

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)
        if int(top_k) > 0:
            generate_kwargs["top_k"] = int(top_k)
        if float(min_p) > 0.0:
            generate_kwargs["min_p"] = float(min_p)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    new_tokens = outputs[:, input_len:]
    return tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=bool(skip_special_tokens),
        clean_up_tokenization_spaces=False,
    )


def run_transformers_boxed_mcqa(
    *,
    model_path: str | dict[str, Any],
    prompt_texts: list[str],
    example_keys: list[str],
    thinking_budget_tokens: int,
    answer_max_new_tokens: int,
    max_model_len: int,
    temperature: float,
    do_sample: bool | None = None,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    seed: int = 0,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    early_stopping_text: str = DEFAULT_EARLY_STOPPING_TEXT,
    mode_label: str = "eval",
    use_tqdm: bool = True,
    answer_parser: Callable[[str], Any] = extract_boxed_answer,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    del gpu_memory_utilization, tensor_parallel_size  # Not used in transformers path.

    if len(prompt_texts) != len(example_keys):
        raise ValueError("prompt_texts and example_keys must have the same length")

    dtype_raw = os.environ.get("TRANSFORMERS_EVAL_DTYPE", "float32")
    torch_dtype = _resolve_torch_dtype(dtype_raw)
    device_map = os.environ.get("TRANSFORMERS_EVAL_DEVICE_MAP", "").strip() or None
    device = os.environ.get("TRANSFORMERS_EVAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu").strip()
    batch_size = max(1, int(os.environ.get("TRANSFORMERS_EVAL_BATCH_SIZE", "2")))
    do_sample = (float(temperature) > 0.0) if do_sample is None else bool(do_sample)

    configure_transformers_determinism(int(seed))

    model, tokenizer, resolved = _load_model(
        model_path=model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        device=device,
    )
    keep_special_tokens = is_gemma4_model_spec(model_path)
    early_stopping_text = normalize_early_stopping_text(
        early_stopping_text,
        str(resolved["base_model"]),
    )

    max_prompt_length = int(max_model_len) - int(thinking_budget_tokens) - 8
    eval_start = time.perf_counter()
    generated_texts: list[str] = []
    iterator = range(0, len(prompt_texts), batch_size)
    progress = tqdm(
        iterator,
        total=(len(prompt_texts) + batch_size - 1) // batch_size,
        desc=f"transformers:{mode_label}:think",
        disable=not bool(use_tqdm),
        unit="batch",
    )
    for start in progress:
        batch = prompt_texts[start : start + batch_size]
        generated_texts.extend(
            _generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_batch=batch,
                max_prompt_length=max_prompt_length,
                max_new_tokens=thinking_budget_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                device=device,
                skip_special_tokens=not keep_special_tokens,
            )
        )

    results: dict[str, dict[str, Any]] = {}
    needs_forcing: list[tuple[int, str, str, str]] = []
    for idx, key in enumerate(example_keys):
        gen_text = str(generated_texts[idx])
        predicted = answer_parser(gen_text)
        force_reason = ""
        if predicted is None:
            if not contains_thinking_close(gen_text):
                force_reason = "budget_forced_no_think_end"
            else:
                force_reason = "think_done_missing_boxed"
            needs_forcing.append((idx, key, gen_text, force_reason))
        results[key] = {
            "raw_output": gen_text,
            "predicted_answer": predicted,
            "budget_forced": False,
            "force_reason": force_reason,
        }

    if needs_forcing:
        forced_prompts: list[str] = []
        for idx, _, gen_text, _ in needs_forcing:
            thinking_only = gen_text
            if contains_thinking_close(gen_text):
                answer_tail = split_after_last_thinking_close(gen_text)
                if answer_tail and gen_text.endswith(answer_tail):
                    thinking_only = gen_text[: -len(answer_tail)].rstrip()
            forced_prompts.append(prompt_texts[idx] + thinking_only + str(early_stopping_text))

        max_forced_prompt_length = int(max_model_len) - int(answer_max_new_tokens) - 8
        forced_texts: list[str] = []
        iterator = range(0, len(forced_prompts), batch_size)
        progress = tqdm(
            iterator,
            total=(len(forced_prompts) + batch_size - 1) // batch_size,
            desc=f"transformers:{mode_label}:force",
            disable=not bool(use_tqdm),
            unit="batch",
        )
        for start in progress:
            batch = forced_prompts[start : start + batch_size]
            forced_texts.extend(
                _generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_batch=batch,
                    max_prompt_length=max_forced_prompt_length,
                    max_new_tokens=answer_max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    device=device,
                    skip_special_tokens=not keep_special_tokens,
                )
            )

        for j, (idx, key, gen_text, force_reason) in enumerate(needs_forcing):
            answer_text = str(forced_texts[j])
            full_text = gen_text + str(early_stopping_text) + answer_text
            predicted = answer_parser(full_text)
            results[key] = {
                "raw_output": full_text,
                "predicted_answer": predicted,
                "budget_forced": True,
                "force_reason": force_reason,
            }

    elapsed = time.perf_counter() - eval_start
    meta = {
        "mode_label": mode_label,
        "num_examples": len(prompt_texts),
        "elapsed_seconds": float(elapsed),
        "budget_forced_count": int(sum(1 for x in results.values() if bool(x["budget_forced"]))),
        "backend": "transformers",
        "dtype": str(dtype_raw),
        "device": str(device),
        "device_map": device_map,
        "attn_implementation": os.environ.get("TRANSFORMERS_EVAL_ATTN_IMPL", "").strip() or None,
        "batch_size": batch_size,
        "base_model": str(resolved["base_model"]),
        "lora_adapter": resolved["lora_adapter"],
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)

    return results, meta
