from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from src.utils.thinking_tags import (
    contains_thinking_close,
    normalize_early_stopping_text,
    split_after_last_thinking_close,
)


def _prepare_vllm_env() -> None:
    # These evaluators/builders do not rely on vLLM's batch_invariant mode.
    # Force it off so a user shell export does not break engine startup.
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    cache_base = (
        os.environ.get("XDG_CACHE_HOME")
        or os.environ.get("SCRATCH")
        or os.environ.get("TMPDIR")
        or f"/tmp/{os.environ.get('USER', 'vllm')}"
    )
    cache_root = Path(cache_base) / "vllm"
    config_root = Path(cache_base) / "vllm_config"
    torchinductor_root = Path(cache_base) / "torchinductor"
    triton_root = Path(cache_base) / "triton"
    for path in (cache_root, config_root, torchinductor_root, triton_root):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VLLM_CACHE_ROOT", str(cache_root))
    os.environ.setdefault("VLLM_CONFIG_ROOT", str(config_root))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_root))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_root))
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")


DEFAULT_EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the "
    "solution based on the thinking directly now.\n</think>\n\n\\boxed{"
)


def is_gemma4_model_spec(model_path: str | dict[str, Any]) -> bool:
    if isinstance(model_path, dict):
        candidates = [
            model_path.get("base_model"),
            model_path.get("tokenizer"),
            model_path.get("name"),
        ]
    else:
        candidates = [model_path]
    raw = " ".join(str(x or "") for x in candidates).strip().lower()
    return "gemma-4" in raw or "gemma4" in raw


def _gemma4_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if isinstance(content, list):
            gemma_content = content
        else:
            gemma_content = [{"type": "text", "text": str(content or "")}]
        out.append({"role": role, "content": gemma_content})
    return out


def _cleanup_vllm_dist_and_memory() -> None:
    gc.collect()
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_boxed_answer(text: str) -> int | None:
    patterns = [
        r"\\boxed\{\s*(\d+)(?:\s*[\).][^}]*)?\s*\}",
        r"\$\\boxed\{\s*(\d+)(?:\s*[\).][^}]*)?\s*\}\$",
        r"Answer:\s*(\d+)",
        r"Final Answer:\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return int(match.group(1).strip())
            except Exception:
                return None
    return None


def resolve_model_spec(model_spec: str | dict[str, Any]) -> dict[str, str | None]:
    if isinstance(model_spec, dict):
        base_model = str(
            model_spec.get("base_model")
            or model_spec.get("model")
            or model_spec.get("model_path")
            or ""
        ).strip()
        tokenizer = str(
            model_spec.get("tokenizer")
            or model_spec.get("tokenizer_name")
            or model_spec.get("tokenizer_path")
            or ""
        ).strip()
        lora_adapter = str(
            model_spec.get("lora_adapter")
            or model_spec.get("adapter")
            or model_spec.get("lora_path")
            or ""
        ).strip()
        display_name = str(model_spec.get("name") or "").strip()
    else:
        raw = str(model_spec).strip()
        if "::" in raw:
            base_model, lora_adapter = [part.strip() for part in raw.split("::", 1)]
            display_name = Path(lora_adapter).name
        else:
            base_model = raw
            lora_adapter = ""
            display_name = ""
        tokenizer = ""

    if not base_model:
        raise ValueError(f"Invalid model spec without base model: {model_spec!r}")

    if not display_name:
        display_name = Path(lora_adapter).name if lora_adapter else base_model.rsplit("/", 1)[-1]

    return {
        "base_model": base_model,
        "tokenizer": tokenizer or base_model,
        "lora_adapter": lora_adapter or None,
        "display_name": display_name,
    }


def _local_model_config(model_name: str) -> dict[str, Any] | None:
    config_path = Path(str(model_name).strip()).expanduser() / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_ministral3_model_spec(model_spec: str | dict[str, Any]) -> bool:
    resolved = resolve_model_spec(model_spec)
    base_model = str(resolved["base_model"]).strip()
    raw = base_model.lower()
    if "ministral-3" in raw or "ministral3" in raw:
        return True

    config = _local_model_config(base_model)
    if not config:
        return False

    if str(config.get("model_type") or "").strip().lower() == "mistral3":
        return True

    architectures = [str(x).strip().lower() for x in (config.get("architectures") or [])]
    return any("mistral3" in arch for arch in architectures)


def _tokenizer_extra_kwargs(model_spec: str | dict[str, Any]) -> dict[str, Any]:
    # Recent Mistral tokenizers require this flag to avoid incorrect regex tokenization.
    if is_ministral3_model_spec(model_spec):
        return {"fix_mistral_regex": True}
    return {}


def _format_llama2_chat(messages: list[dict[str, str]]) -> str:
    system_parts: list[str] = []
    turns: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role == "system":
            system_parts.append(content)
        else:
            turns.append({"role": role, "content": content})

    if not turns:
        turns = [{"role": "user", "content": "\n\n".join(system_parts).strip()}]
        system_parts = []

    rendered: list[str] = []
    system_text = "\n\n".join(x for x in system_parts if x).strip()
    idx = 0
    while idx < len(turns):
        user = turns[idx]
        if user["role"] != "user":
            idx += 1
            continue
        user_text = user["content"]
        if system_text:
            user_text = f"<<SYS>>\n{system_text}\n<</SYS>>\n\n{user_text}"
            system_text = ""
        chunk = f"<s>[INST] {user_text} [/INST]"
        if idx + 1 < len(turns) and turns[idx + 1]["role"] == "assistant":
            chunk += f" {turns[idx + 1]['content']} </s>"
            idx += 2
        else:
            idx += 1
        rendered.append(chunk)
    return "".join(rendered)


def _apply_chat_template_compat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        if enable_thinking:
            return tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except ValueError as exc:
        if "chat_template is not set" not in str(exc):
            raise
        return _format_llama2_chat(messages)


def _checkpoint_step_key(path: Path) -> tuple[int, str]:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    if m:
        return int(m.group(1)), path.name
    return 10**18, path.name


def expand_model_specs(model_specs: list[Any]) -> list[Any]:
    expanded: list[Any] = []
    for spec in model_specs:
        if not isinstance(spec, dict):
            expanded.append(spec)
            continue

        if "model_root" in spec or "full_model_root" in spec or "checkpoint_root" in spec:
            model_root = Path(
                str(
                    spec.get("model_root")
                    or spec.get("full_model_root")
                    or spec.get("checkpoint_root")
                    or ""
                ).strip()
            ).expanduser()
            include_root = bool(spec.get("include_root", False))
            checkpoint_steps_raw = spec.get("checkpoint_steps")
            checkpoint_names_raw = spec.get("checkpoint_names")
            name_prefix = str(spec.get("name") or spec.get("name_prefix") or model_root.name).strip()
            tokenizer = str(
                spec.get("tokenizer")
                or spec.get("tokenizer_name")
                or spec.get("tokenizer_path")
                or ""
            ).strip()
            if not model_root.exists() or not model_root.is_dir():
                raise FileNotFoundError(f"model_root not found or not a directory: {model_root}")

            checkpoint_steps: set[int] | None = None
            if checkpoint_steps_raw is not None:
                checkpoint_steps = {int(step) for step in checkpoint_steps_raw}

            checkpoint_names: set[str] | None = None
            if checkpoint_names_raw is not None:
                checkpoint_names = {str(name).strip() for name in checkpoint_names_raw if str(name).strip()}

            if include_root and (model_root / "config.json").exists():
                root_spec = {"model": str(model_root), "name": name_prefix or model_root.name}
                if tokenizer:
                    root_spec["tokenizer"] = tokenizer
                expanded.append(root_spec)

            checkpoints = sorted(
                [p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
                key=_checkpoint_step_key,
            )
            for ckpt in checkpoints:
                if checkpoint_steps is not None:
                    match = re.search(r"checkpoint-(\d+)$", ckpt.name)
                    if match is None or int(match.group(1)) not in checkpoint_steps:
                        continue
                if checkpoint_names is not None and ckpt.name not in checkpoint_names:
                    continue
                if (ckpt / "config.json").exists():
                    display_name = f"{name_prefix}:{ckpt.name}" if name_prefix else ckpt.name
                    ckpt_spec = {"model": str(ckpt), "name": display_name}
                    if tokenizer:
                        ckpt_spec["tokenizer"] = tokenizer
                    expanded.append(ckpt_spec)
            continue

        if "adapter_root" not in spec:
            expanded.append(spec)
            continue

        base_model = str(spec.get("base_model") or "").strip()
        adapter_root = Path(str(spec.get("adapter_root") or "").strip()).expanduser()
        include_root = bool(spec.get("include_root", False))
        checkpoint_steps_raw = spec.get("checkpoint_steps")
        checkpoint_names_raw = spec.get("checkpoint_names")
        if not base_model:
            raise ValueError(f"Missing base_model for checkpoint scan spec: {spec!r}")
        if not adapter_root.exists() or not adapter_root.is_dir():
            raise FileNotFoundError(f"adapter_root not found or not a directory: {adapter_root}")

        checkpoint_steps: set[int] | None = None
        if checkpoint_steps_raw is not None:
            checkpoint_steps = {int(step) for step in checkpoint_steps_raw}

        checkpoint_names: set[str] | None = None
        if checkpoint_names_raw is not None:
            checkpoint_names = {str(name).strip() for name in checkpoint_names_raw if str(name).strip()}

        if include_root and (adapter_root / "adapter_config.json").exists():
            expanded.append(f"{base_model}::{adapter_root}")

        checkpoints = sorted(
            [p for p in adapter_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=_checkpoint_step_key,
        )
        for ckpt in checkpoints:
            if checkpoint_steps is not None:
                match = re.search(r"checkpoint-(\d+)$", ckpt.name)
                if match is None or int(match.group(1)) not in checkpoint_steps:
                    continue
            if checkpoint_names is not None and ckpt.name not in checkpoint_names:
                continue
            if (ckpt / "adapter_config.json").exists():
                expanded.append(f"{base_model}::{ckpt}")
    return expanded


def build_prompt_texts(
    model_path: str | dict[str, Any],
    message_batches: list[list[dict[str, str]]],
    *,
    enable_thinking: bool,
) -> tuple[Any, list[str]]:
    resolved = resolve_model_spec(model_path)
    if is_gemma4_model_spec(model_path):
        from transformers import AutoProcessor

        processor_or_tokenizer = AutoProcessor.from_pretrained(
            str(resolved["tokenizer"]),
            trust_remote_code=True,
        )
        tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt_texts = []
        for messages in message_batches:
            prompt_texts.append(
                processor_or_tokenizer.apply_chat_template(
                    _gemma4_messages(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(enable_thinking),
                )
            )
        return tokenizer, prompt_texts

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved["tokenizer"]),
        trust_remote_code=True,
        **_tokenizer_extra_kwargs(model_path),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts: list[str] = []
    for messages in message_batches:
        text = _apply_chat_template_compat(
            tokenizer,
            messages,
            enable_thinking=bool(enable_thinking),
        )
        prompt_texts.append(text)
    return tokenizer, prompt_texts


def run_vllm_boxed_mcqa(
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
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    early_stopping_text: str = DEFAULT_EARLY_STOPPING_TEXT,
    mode_label: str = "eval",
    use_tqdm: bool = True,
    answer_parser: Callable[[str], Any] = extract_boxed_answer,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _prepare_vllm_env()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    resolved = resolve_model_spec(model_path)
    base_model = str(resolved["base_model"])
    tokenizer_name = str(resolved["tokenizer"])
    lora_adapter = resolved["lora_adapter"]

    llm_kwargs: dict[str, Any] = {
        "model": base_model,
        "tokenizer": tokenizer_name,
        "dtype": "bfloat16",
        "max_model_len": int(max_model_len),
        "trust_remote_code": True,
        "seed": int(seed),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "tensor_parallel_size": int(tensor_parallel_size),
    }
    if max_num_seqs is not None and int(max_num_seqs) > 0:
        llm_kwargs["max_num_seqs"] = int(max_num_seqs)
    if max_num_batched_tokens is not None and int(max_num_batched_tokens) > 0:
        llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if is_ministral3_model_spec(model_path):
        llm_kwargs["tokenizer_mode"] = "mistral"
    if is_gemma4_model_spec(model_path):
        # Gemma4 is multimodal; our MCQA eval is text-only. Avoid allocating
        # image/audio processor capacity during vLLM profiling.
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}
    if lora_adapter:
        llm_kwargs["enable_lora"] = True
    try:
        llm = LLM(**llm_kwargs)
    except Exception:
        _cleanup_vllm_dist_and_memory()
        raise

    lora_request = None
    if lora_adapter:
        lora_request = LoRARequest(
            lora_name=str(resolved["display_name"]),
            lora_int_id=1,
            lora_path=str(lora_adapter),
            base_model_name=base_model,
        )
    early_stopping_text = normalize_early_stopping_text(early_stopping_text, base_model)

    eval_start = time.perf_counter()
    do_sample = (float(temperature) > 0.0) if do_sample is None else bool(do_sample)
    sampling_params = SamplingParams(
        max_tokens=max(1, int(thinking_budget_tokens)),
        temperature=float(temperature) if do_sample else 0.0,
        top_p=float(top_p) if do_sample else 1.0,
        top_k=int(top_k) if do_sample else -1,
        min_p=float(min_p) if do_sample else 0.0,
        n=1,
    )
    outputs = llm.generate(prompt_texts, sampling_params, use_tqdm=use_tqdm, lora_request=lora_request)

    results: dict[str, dict[str, Any]] = {}
    needs_forcing: list[tuple[int, str, str, str]] = []

    for idx, key in enumerate(example_keys):
        gen_text = outputs[idx].outputs[0].text
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

        answer_params = SamplingParams(
            max_tokens=max(1, int(answer_max_new_tokens)),
            temperature=float(temperature) if do_sample else 0.0,
            top_p=float(top_p) if do_sample else 1.0,
            top_k=int(top_k) if do_sample else -1,
            min_p=float(min_p) if do_sample else 0.0,
        )
        forced_outputs = llm.generate(
            forced_prompts,
            answer_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

        for j, (idx, key, gen_text, force_reason) in enumerate(needs_forcing):
            answer_text = forced_outputs[j].outputs[0].text
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
        "base_model": base_model,
        "lora_adapter": lora_adapter,
    }

    try:
        llm.sleep(level=2)
    except Exception:
        pass

    try:
        del llm
    except Exception:
        pass

    _cleanup_vllm_dist_and_memory()
    time.sleep(5)

    return results, meta
