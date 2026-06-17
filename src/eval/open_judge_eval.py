from __future__ import annotations

import gc
import os
import re
import time
from typing import Any

import torch

from .transformers_boxed_mcqa import (
    _generate_batch as _transformers_generate_batch,
    _load_model as _load_transformers_model,
    configure_transformers_determinism,
)
from .vllm_boxed_mcqa import resolve_model_spec


def _prepare_vllm_env() -> None:
    # These evaluators/builders do not rely on vLLM's batch_invariant mode.
    # Force it off so a user shell export does not break engine startup.
    os.environ["VLLM_BATCH_INVARIANT"] = "0"


OPEN_ENDED_EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the "
    "solution based on the thinking directly now.\n</think>\n\n"
    "Write your final answer in \\boxed{...}.\n"
)

THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{\s*([\s\S]*?)\s*\}")


def is_openai_model(model_path: str) -> bool:
    return str(model_path).startswith("openai:")


def resolve_openai_model_name(model_path: str) -> str:
    if not is_openai_model(model_path):
        return str(model_path)
    return str(model_path).split(":", 1)[1].strip()


def compute_mean_stderr(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    stderr = (variance / len(values)) ** 0.5
    return mean, stderr


def strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        return m.group(1).strip()
    return raw


def extract_open_answer(text: str) -> str | None:
    raw = str(text or "")
    if THINK_END_RE.search(raw):
        raw = THINK_END_RE.split(raw, maxsplit=1)[-1]
    elif "<think>" in raw.lower():
        return None
    raw = strip_code_fence(raw).strip()
    boxed = BOXED_RE.search(raw)
    if boxed:
        raw = boxed.group(1).strip()
    raw = raw.strip()
    return raw or None


def extract_openai_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text

    output_items = getattr(response, "output", None) or []
    chunks: list[str] = []
    for item in output_items:
        contents = getattr(item, "content", None)
        if contents is None and isinstance(item, dict):
            contents = item.get("content", [])
        for content in contents or []:
            if isinstance(content, dict):
                candidate = content.get("text")
            else:
                candidate = getattr(content, "text", None)
            if candidate:
                chunks.append(str(candidate))
    return "".join(chunks).strip()


def call_openai_responses(
    *,
    client,
    model_name: str,
    prompt_text: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
    max_retries: int = 3,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model_name,
        "input": prompt_text,
        "max_output_tokens": max(1, int(max_tokens)),
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": str(reasoning_effort)}

    last_error: Exception | None = None
    for attempt in range(int(max_retries) + 1):
        try:
            response = client.responses.create(**kwargs)
            return extract_openai_text(response)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt >= int(max_retries):
                break
            time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"OpenAI API call failed for model={model_name}: {last_error}")


def run_vllm_open_ended(
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
    early_stopping_text: str = OPEN_ENDED_EARLY_STOPPING_TEXT,
    mode_label: str = "eval",
    use_tqdm: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _prepare_vllm_env()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    resolved = resolve_model_spec(model_path)
    base_model = str(resolved["base_model"])
    lora_adapter = resolved["lora_adapter"]

    llm_kwargs: dict[str, Any] = {
        "model": base_model,
        "dtype": "bfloat16",
        "max_model_len": int(max_model_len),
        "trust_remote_code": True,
        "seed": int(seed),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "tensor_parallel_size": int(tensor_parallel_size),
    }
    if lora_adapter:
        llm_kwargs["enable_lora"] = True
    llm = LLM(**llm_kwargs)

    lora_request = None
    if lora_adapter:
        lora_request = LoRARequest(
            lora_name=str(resolved["display_name"]),
            lora_int_id=1,
            lora_path=str(lora_adapter),
            base_model_name=base_model,
        )

    eval_start = time.perf_counter()
    do_sample = (float(temperature) > 0.0) if do_sample is None else bool(do_sample)
    sampling_params = SamplingParams(
        max_tokens=max(1, int(thinking_budget_tokens)),
        temperature=float(temperature) if do_sample else 0.0,
        top_p=float(top_p) if do_sample else 1.0,
        top_k=int(top_k) if do_sample else -1,
        min_p=float(min_p) if do_sample else 0.0,
    )
    outputs = llm.generate(prompt_texts, sampling_params, use_tqdm=use_tqdm, lora_request=lora_request)

    results: dict[str, dict[str, Any]] = {}
    needs_forcing: list[tuple[int, str, str, str]] = []

    for idx, key in enumerate(example_keys):
        gen_text = str(outputs[idx].outputs[0].text)
        answer_text = extract_open_answer(gen_text)
        force_reason = ""
        if answer_text is None:
            if "</think>" not in gen_text:
                force_reason = "budget_forced_no_think_end"
            else:
                force_reason = "think_done_missing_answer"
            needs_forcing.append((idx, key, gen_text, force_reason))
        results[key] = {
            "raw_output": gen_text,
            "answer_text": answer_text,
            "budget_forced": False,
            "force_reason": force_reason,
        }

    if needs_forcing:
        forced_prompts: list[str] = []
        for idx, _, gen_text, _ in needs_forcing:
            if "</think>" in gen_text:
                think_end = gen_text.rfind("</think>")
                thinking_only = gen_text[:think_end]
            else:
                thinking_only = gen_text
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
            answer_tail = str(forced_outputs[j].outputs[0].text)
            full_text = gen_text + str(early_stopping_text) + answer_tail
            results[key] = {
                "raw_output": full_text,
                "answer_text": extract_open_answer(full_text),
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

    del llm
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

    return results, meta


def run_transformers_open_ended(
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
    gpu_memory_utilization: float = 0.0,
    tensor_parallel_size: int = 1,
    early_stopping_text: str = OPEN_ENDED_EARLY_STOPPING_TEXT,
    mode_label: str = "eval",
    use_tqdm: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    del gpu_memory_utilization, tensor_parallel_size
    from tqdm.auto import tqdm

    if len(prompt_texts) != len(example_keys):
        raise ValueError("prompt_texts and example_keys must have the same length")

    dtype_raw = os.environ.get("TRANSFORMERS_EVAL_DTYPE", "bfloat16")
    device_map = os.environ.get("TRANSFORMERS_EVAL_DEVICE_MAP", "").strip() or None
    device = os.environ.get("TRANSFORMERS_EVAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu").strip()
    batch_size = max(1, int(os.environ.get("TRANSFORMERS_EVAL_BATCH_SIZE", "1")))
    do_sample = (float(temperature) > 0.0) if do_sample is None else bool(do_sample)
    torch_dtype = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }.get(str(dtype_raw).strip().lower())
    if torch_dtype is None:
        raise ValueError(f"Unsupported TRANSFORMERS_EVAL_DTYPE: {dtype_raw}")

    configure_transformers_determinism(int(seed))
    model, tokenizer, resolved = _load_transformers_model(
        model_path=model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        device=device,
    )

    max_prompt_length = int(max_model_len) - int(thinking_budget_tokens) - 8
    eval_start = time.perf_counter()
    generated_texts: list[str] = []
    starts = range(0, len(prompt_texts), batch_size)
    iterator = tqdm(
        starts,
        total=(len(prompt_texts) + batch_size - 1) // batch_size,
        desc=f"transformers:{mode_label}:think",
        disable=not bool(use_tqdm),
        unit="batch",
    )
    for start in iterator:
        generated_texts.extend(
            _transformers_generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_batch=prompt_texts[start : start + batch_size],
                max_prompt_length=max_prompt_length,
                max_new_tokens=thinking_budget_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                device=device,
            )
        )

    results: dict[str, dict[str, Any]] = {}
    needs_forcing: list[tuple[int, str, str, str]] = []
    for idx, key in enumerate(example_keys):
        gen_text = str(generated_texts[idx])
        answer_text = extract_open_answer(gen_text)
        force_reason = ""
        if answer_text is None:
            if "</think>" not in gen_text:
                force_reason = "budget_forced_no_think_end"
            else:
                force_reason = "think_done_missing_answer"
            needs_forcing.append((idx, key, gen_text, force_reason))
        results[key] = {
            "raw_output": gen_text,
            "answer_text": answer_text,
            "budget_forced": False,
            "force_reason": force_reason,
        }

    if needs_forcing:
        forced_prompts: list[str] = []
        for idx, _, gen_text, _ in needs_forcing:
            if "</think>" in gen_text:
                think_end = gen_text.rfind("</think>")
                thinking_only = gen_text[:think_end]
            else:
                thinking_only = gen_text
            forced_prompts.append(prompt_texts[idx] + thinking_only + str(early_stopping_text))

        max_forced_prompt_length = int(max_model_len) - int(answer_max_new_tokens) - 8
        forced_texts: list[str] = []
        starts = range(0, len(forced_prompts), batch_size)
        iterator = tqdm(
            starts,
            total=(len(forced_prompts) + batch_size - 1) // batch_size,
            desc=f"transformers:{mode_label}:force",
            disable=not bool(use_tqdm),
            unit="batch",
        )
        for start in iterator:
            forced_texts.extend(
                _transformers_generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_batch=forced_prompts[start : start + batch_size],
                    max_prompt_length=max_forced_prompt_length,
                    max_new_tokens=answer_max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    device=device,
                )
            )
        for j, (idx, key, gen_text, force_reason) in enumerate(needs_forcing):
            answer_tail = str(forced_texts[j])
            full_text = gen_text + str(early_stopping_text) + answer_tail
            results[key] = {
                "raw_output": full_text,
                "answer_text": extract_open_answer(full_text),
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
        "batch_size": batch_size,
        "base_model": str(resolved["base_model"]),
        "lora_adapter": resolved["lora_adapter"],
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, meta


def run_openai_open_ended(
    *,
    client,
    model_name: str,
    prompt_texts: list[str],
    example_keys: list[str],
    thinking_budget_tokens: int,
    answer_max_new_tokens: int,
    reasoning_effort: str | None,
    max_retries: int,
    mode_label: str,
    use_tqdm_factory,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    eval_start = time.perf_counter()

    for key, prompt_text in use_tqdm_factory(
        list(zip(example_keys, prompt_texts)),
        total=len(example_keys),
        desc=mode_label,
        unit="ex",
    ):
        gen_text = call_openai_responses(
            client=client,
            model_name=model_name,
            prompt_text=prompt_text,
            max_tokens=int(thinking_budget_tokens),
            reasoning_effort=reasoning_effort,
            max_retries=max_retries,
        )
        answer_text = extract_open_answer(gen_text)
        force_reason = ""
        budget_forced = False
        if answer_text is None:
            if "</think>" not in gen_text:
                force_reason = "budget_forced_no_think_end"
            else:
                force_reason = "think_done_missing_answer"
            budget_forced = True
            if "</think>" in gen_text:
                think_end = gen_text.rfind("</think>")
                thinking_only = gen_text[:think_end]
            else:
                thinking_only = gen_text
            answer_tail = call_openai_responses(
                client=client,
                model_name=model_name,
                prompt_text=prompt_text + thinking_only + str(OPEN_ENDED_EARLY_STOPPING_TEXT),
                max_tokens=int(answer_max_new_tokens),
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
            )
            gen_text = gen_text + str(OPEN_ENDED_EARLY_STOPPING_TEXT) + answer_tail
            answer_text = extract_open_answer(gen_text)

        results[str(key)] = {
            "raw_output": gen_text,
            "answer_text": answer_text,
            "budget_forced": budget_forced,
            "force_reason": force_reason,
        }

    elapsed = time.perf_counter() - eval_start
    meta = {
        "mode_label": mode_label,
        "num_examples": len(prompt_texts),
        "elapsed_seconds": float(elapsed),
        "budget_forced_count": int(sum(1 for x in results.values() if bool(x["budget_forced"]))),
    }
    return results, meta


def _parse_judge_verdict(text: str) -> tuple[bool | None, str]:
    raw = strip_code_fence(text).strip()
    upper = raw.upper()
    if "INCORRECT" in upper:
        return False, raw
    if re.search(r"\bCORRECT\b", upper):
        return True, raw
    if re.search(r"\bTRUE\b", upper):
        return True, raw
    if re.search(r"\bFALSE\b", upper):
        return False, raw
    return None, raw


class OpenAIJudge:
    def __init__(
        self,
        *,
        model_name: str = "gpt-5.2",
        timeout: float = 180.0,
        reasoning_effort: str | None = "medium",
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(timeout=float(timeout))
        self.model_name = str(model_name)
        self.reasoning_effort = reasoning_effort
        self.max_retries = int(max_retries)
        self._cache: dict[tuple[str, str, str, str], tuple[bool, str]] = {}

    def judge(
        self,
        *,
        dataset_name: str,
        task_prompt: str,
        gold_answer: str,
        candidate_answer: str | None,
    ) -> tuple[bool, str]:
        candidate = str(candidate_answer or "").strip()
        if not candidate:
            return False, "empty_answer"

        key = (str(dataset_name), str(task_prompt), str(gold_answer), candidate)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        judge_prompt = (
            f"[Scenario]: {task_prompt}\n"
            f"[Model's Answer]: {candidate}\n"
            f"[Gold Answer for Reference]: {gold_answer}\n\n"
            "We request your evaluation of the AI model's answer in relation to the provided "
            "scenario and the gold answer. Assess the responses based on the following criteria: "
            "1. Social Understanding: How well does the model's answer grasp the social dynamics "
            "or pragmatic nuances of the scenario? "
            "2. Appropriateness: Is the model's answer appropriate and contextually fitting for the scenario? "
            "3. Insightfulness: Does the answer demonstrate a deep understanding of the underlying intentions, "
            "implicature, deceit, irony, sarcasm, humor, metaphor, etc.? "
            "4. Completeness: How comprehensive is the model's response in capturing the essential elements "
            "of the scenario?\n\n"
            "Return exactly one word only: CORRECT or INCORRECT."
        )
        raw = call_openai_responses(
            client=self._client,
            model_name=self.model_name,
            prompt_text=judge_prompt,
            max_tokens=6000,
            reasoning_effort=self.reasoning_effort,
            max_retries=self.max_retries,
        )
        correct, parsed = _parse_judge_verdict(raw)
        if correct is None and self.reasoning_effort is not None:
            retry_raw = call_openai_responses(
                client=self._client,
                model_name=self.model_name,
                prompt_text=judge_prompt,
                max_tokens=8000,
                reasoning_effort=None,
                max_retries=self.max_retries,
            )
            if str(retry_raw).strip():
                raw = retry_raw
            correct, parsed = _parse_judge_verdict(raw)
        if correct is None:
            correct = False
            parsed = f"judge_parse_error: {(str(raw).strip() or 'EMPTY_JUDGE_OUTPUT')[:120]}"
        else:
            parsed = str(raw).strip() or str(parsed).strip() or ("CORRECT" if correct else "INCORRECT")
        result = (bool(correct), str(parsed))
        self._cache[key] = result
        return result
