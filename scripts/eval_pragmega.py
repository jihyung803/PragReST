#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.eval_common import (  # noqa: E402
    DEFAULT_ANSWER_MAX_NEW_TOKENS,
    DEFAULT_MAX_PROMPT_LENGTH,
    JUDGE_EXTRACTION_PROMPT_TEMPLATE,
    get_eval_results_dir,
    load_pragmega_data,
    make_messages,
)
from src.utils.thinking_tags import ALL_CLOSE_TAGS, ALL_OPEN_TAGS  # noqa: E402
from src.eval.transformers_boxed_mcqa import run_transformers_boxed_mcqa  # noqa: E402
from src.eval.vllm_boxed_mcqa import (  # noqa: E402
    build_prompt_texts,
    expand_model_specs,
    is_gemma4_model_spec,
    resolve_model_spec,
    run_vllm_boxed_mcqa,
)
from src.eval.sglang_boxed_mcqa import run_sglang_boxed_mcqa  # noqa: E402


MODELS = [
    "Qwen/Qwen3-8B",
]


def _configured_models():
    eval_model_spec = os.environ.get("EVAL_MODEL_SPEC", "").strip()
    if eval_model_spec:
        if eval_model_spec.startswith(("{", "[")):
            parsed = json.loads(eval_model_spec)
            return parsed if isinstance(parsed, list) else [parsed]
        return [eval_model_spec]

    single_model_spec = os.environ.get("PRAGMEGA_MODEL_SPEC", "").strip()
    if single_model_spec:
        return [single_model_spec]

    full_model_root = os.environ.get("PRAGMEGA_MODEL_ROOT", "").strip()
    if not full_model_root:
        return MODELS

    spec = {
        "model_root": full_model_root,
        "tokenizer": os.environ.get("PRAGMEGA_TOKENIZER", "").strip() or full_model_root,
        "include_root": os.environ.get("PRAGMEGA_INCLUDE_ROOT", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        "name": os.environ.get("PRAGMEGA_MODEL_NAME", Path(full_model_root).name).strip()
        or Path(full_model_root).name,
    }

    checkpoint_steps = os.environ.get("PRAGMEGA_CHECKPOINT_STEPS", "").strip()
    if checkpoint_steps:
        spec["checkpoint_steps"] = [
            int(x.strip()) for x in checkpoint_steps.split(",") if x.strip()
        ]

    checkpoint_names = os.environ.get("PRAGMEGA_CHECKPOINT_NAMES", "").strip()
    if checkpoint_names:
        spec["checkpoint_names"] = [x.strip() for x in checkpoint_names.split(",") if x.strip()]

    return [spec]

DATA_PATH = os.environ.get("PRAGMEGA_DATA_PATH", "data/eval/prompts/selected")
FOCUS_PHENOMENA = None
NUM_EXAMPLES = None
SEEDS = [
    int(x.strip())
    for x in os.environ.get("PRAGMEGA_SEEDS", "1").split(",")
    if x.strip()
]
EVAL_BACKEND = os.environ.get("EVAL_BACKEND", "vllm").strip().lower()

PHENOM_DISPLAY = {
    "CIV": "Coh",
    "DV": "Deceits",
    "HV": "Humour",
    "ISV": "Ind",
    "IV": "Irony",
    "MV": "Maxims",
    "MPV": "Metaphor",
}
PHENOM_ORDER = ["CIV", "DV", "HV", "ISV", "IV", "MV", "MPV"]

ENABLE_THINKING = True
MAX_PROMPT_LENGTH = int(os.environ.get("PRAGMEGA_MAX_PROMPT_LENGTH", "8192"))
THINKING_BUDGET_TOKENS = int(os.environ.get("PRAGMEGA_THINKING_BUDGET_TOKENS", "2048"))
ANSWER_MAX_NEW_TOKENS = int(os.environ.get("PRAGMEGA_ANSWER_MAX_NEW_TOKENS", "456"))
DO_SAMPLE = os.environ.get("PRAGMEGA_DO_SAMPLE", "False").strip().lower() not in {"0", "false", "no", "off"}
TEMPERATURE = float(os.environ.get("PRAGMEGA_TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("PRAGMEGA_TOP_P", "0.95"))
TOP_K = int(os.environ.get("PRAGMEGA_TOP_K", "20"))
MIN_P = float(os.environ.get("PRAGMEGA_MIN_P", "0.0"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("PRAGMEGA_GPU_MEMORY_UTILIZATION", "0.8"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("PRAGMEGA_TENSOR_PARALLEL_SIZE", "1"))
VLLM_MAX_NUM_SEQS = int(os.environ.get("PRAGMEGA_VLLM_MAX_NUM_SEQS", "128"))
VLLM_MAX_NUM_BATCHED_TOKENS = int(os.environ.get("PRAGMEGA_VLLM_MAX_NUM_BATCHED_TOKENS", "0"))
ISOLATE_EACH_RUN = True
USE_EXTRACTION_FALLBACK = os.environ.get("PRAGMEGA_USE_EXTRACTION_FALLBACK", "1").strip().lower() not in {"0", "false", "no", "off"}
EXTRACTION_FALLBACK_MODEL = os.environ.get("PRAGMEGA_EXTRACTION_MODEL", os.environ.get("BK_LINK_JUDGE_MODEL", "gpt-5-mini")).strip()
EXTRACTION_FALLBACK_BASE_URL = os.environ.get("PRAGMEGA_EXTRACTION_BASE_URL", os.environ.get("BK_LINK_JUDGE_BASE_URL", "https://api.openai.com/v1")).strip()
EXTRACTION_FALLBACK_API_KEY = os.environ.get("PRAGMEGA_EXTRACTION_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip()
EXTRACTION_FALLBACK_CONCURRENCY = int(os.environ.get("PRAGMEGA_EXTRACTION_CONCURRENCY", "8"))

def _short_name(model_path: str | dict) -> str:
    return str(resolve_model_spec(model_path)["display_name"])


def _model_key(model_path: str | dict) -> str:
    return _short_name(model_path)


def _compute_mean_stderr(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return mean, stderr


def _try_int_eq(left, right) -> bool:
    try:
        return int(left) == int(right)
    except Exception:
        return False


def _fmt(mean: float, stderr: float) -> str:
    return f"{mean:.2f}\u00b1{stderr:.2f}"


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return cleaned.strip("._") or "model"


def _result_predicted(entry) -> int | None:
    if isinstance(entry, dict):
        return entry.get("predicted_answer")
    return entry[1]


def _result_raw_output(entry) -> str:
    if isinstance(entry, dict):
        return str(entry.get("raw_output", ""))
    return str(entry[0])


def _write_example_outputs(results_dir: str, timestamp: str, model_specs, examples, all_results) -> str:
    outputs_dir = os.path.join(results_dir, f"pragmega_eval_outputs_{timestamp}")
    os.makedirs(outputs_dir, exist_ok=True)

    for model_path in model_specs:
        short = _short_name(model_path)
        safe_model = _safe_filename(short)
        res = all_results[_model_key(model_path)]
        for seed_idx, seed in enumerate(SEEDS):
            seed_results = res["per_seed_results"][seed_idx]
            out_path = os.path.join(outputs_dir, f"{safe_model}_seed{seed}.jsonl")
            fail_path = os.path.join(outputs_dir, f"{safe_model}_seed{seed}_failures.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                with open(fail_path, "w", encoding="utf-8") as ff:
                    for ex in examples:
                        entry = seed_results[ex["unique_key"]]
                        raw_output = _result_raw_output(entry)
                        predicted_answer = _result_predicted(entry)
                        extraction_failure = predicted_answer is None
                        correct = bool(
                            predicted_answer is not None
                            and _try_int_eq(predicted_answer, ex.get("correct_answer_idx"))
                        )
                        row = {
                            "model": short,
                            "seed": seed,
                            "id": ex["unique_key"],
                            "phenomenon": ex.get("phenomenon"),
                            "item_id": ex.get("item_id"),
                            "full_prompt": ex.get("full_prompt"),
                            "gold_index": ex.get("correct_answer_idx"),
                            "predicted_answer": predicted_answer,
                            "correct": correct,
                            "extraction_failure": extraction_failure,
                            "raw_output": raw_output,
                        }
                        if isinstance(entry, dict):
                            row["final_output"] = entry.get("final_output", "")
                            row["reasoning_output"] = entry.get("reasoning_output", "")
                            row["extraction_recovered"] = bool(entry.get("extraction_recovered", False))
                            row["extraction_source"] = entry.get("extraction_source", "")
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if extraction_failure:
                            ff.write(json.dumps(row, ensure_ascii=False) + "\n")
    return outputs_dir


def _extract_think_text(raw_output: str) -> str:
    text = str(raw_output or "")
    lowered = text.lower()

    start_idx = -1
    start_len = 0
    for tag in ALL_OPEN_TAGS:
        idx = lowered.find(tag.lower())
        if idx != -1 and (start_idx == -1 or idx < start_idx):
            start_idx = idx
            start_len = len(tag)

    end_idx = -1
    for tag in ALL_CLOSE_TAGS:
        idx = lowered.find(tag.lower())
        if idx != -1 and (end_idx == -1 or idx < end_idx):
            end_idx = idx

    if start_idx != -1 and end_idx != -1 and start_idx + start_len <= end_idx:
        return text[start_idx + start_len : end_idx]
    if end_idx != -1:
        return text[:end_idx]
    if start_idx != -1:
        return text[start_idx + start_len :]
    return ""


def _count_think_tokens(tokenizer, raw_output: str) -> int:
    think_text = _extract_think_text(raw_output)
    if not think_text.strip():
        return 0
    return len(tokenizer.encode(think_text, add_special_tokens=False))


def _extract_gemma4_vllm_thought_text(raw_output: str) -> str:
    text = str(raw_output or "").strip()
    if not text:
        return ""
    tagged = _extract_think_text(text)
    if tagged.strip():
        return tagged

    match = re.match(r"^\s*thought\s*\n+(.*)$", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    body = match.group(1).strip()
    boxed = re.search(r"\\boxed\{", body)
    if boxed:
        body = body[: boxed.start()].strip()
    return body


def _count_reasoning_tokens(tokenizer, raw_output: str, *, model_path: str | dict) -> int:
    if is_gemma4_model_spec(model_path):
        thought_text = _extract_gemma4_vllm_thought_text(raw_output)
        if not thought_text.strip():
            return 0
        return len(tokenizer.encode(thought_text, add_special_tokens=False))
    return _count_think_tokens(tokenizer, raw_output)


def _build_message_batches(model_path: str | dict, examples: list[dict]) -> list[list[dict[str, str]]]:
    return [make_messages(ex) for ex in examples]


def _parse_extractor_answer(text: str) -> int | None:
    raw = str(text or "").strip()
    if not raw or raw.upper() == "NA":
        return None
    match = re.search(r"\b([1-5])\b", raw)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _build_extraction_fallback_prompt(full_prompt: str, raw_output: str) -> str:
    return (
        "You are recovering the intended final answer from a model response to a multiple-choice question.\n"
        "Infer which option number (1, 2, 3, or 4) the model meant to choose, even if it forgot to box it.\n"
        "If the response does not reveal a single intended choice, output NA.\n\n"
        f"Question:\n{full_prompt}\n\n"
        + JUDGE_EXTRACTION_PROMPT_TEMPLATE.format(text=raw_output)
    )


async def _extract_failed_answer_one(client, semaphore, *, full_prompt: str, raw_output: str, idx: int) -> int | None:
    async with semaphore:
        prompt = _build_extraction_fallback_prompt(full_prompt, raw_output)
        for attempt in range(2):
            try:
                request_kwargs = {
                    "model": EXTRACTION_FALLBACK_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You extract the intended final answer option from a model response. "
                                "Return only one token: 1, 2, 3, 4, or NA."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                }
                if os.environ.get("PRAGMEGA_EXTRACTION_TEMPERATURE", "").strip():
                    request_kwargs["temperature"] = float(os.environ["PRAGMEGA_EXTRACTION_TEMPERATURE"])
                response = await client.chat.completions.create(
                    **request_kwargs,
                )
                parsed = _parse_extractor_answer(response.choices[0].message.content or "")
                if parsed is not None:
                    return parsed
                content = str(response.choices[0].message.content or "").strip()
                if content.upper() == "NA":
                    return None
            except Exception as exc:
                print(f"[extractor] item {idx} attempt {attempt + 1} failed: {exc}")
        return None


async def _extract_failed_answers_async(items: list[tuple[int, str, str]]) -> list[int | None]:
    from openai import AsyncOpenAI  # pyright: ignore[reportMissingImports]

    semaphore = asyncio.Semaphore(max(1, EXTRACTION_FALLBACK_CONCURRENCY))
    async with AsyncOpenAI(
        base_url=EXTRACTION_FALLBACK_BASE_URL,
        api_key=EXTRACTION_FALLBACK_API_KEY,
    ) as client:
        tasks = [
            _extract_failed_answer_one(
                client,
                semaphore,
                full_prompt=full_prompt,
                raw_output=raw_output,
                idx=idx,
            )
            for idx, full_prompt, raw_output in items
        ]
        return list(await asyncio.gather(*tasks))


def _recover_extraction_failures(examples: list[dict], raw_results: dict[str, dict]) -> int:
    if not USE_EXTRACTION_FALLBACK or not EXTRACTION_FALLBACK_API_KEY:
        return 0
    try:
        from openai import AsyncOpenAI  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        return 0

    failure_items: list[tuple[int, str, str]] = []
    failure_keys: list[str] = []
    for idx, ex in enumerate(examples):
        key = str(ex["unique_key"])
        result = raw_results[key]
        if result.get("predicted_answer") is not None:
            continue
        raw_output = str(result.get("raw_output", ""))
        if not raw_output.strip():
            continue
        failure_items.append((idx, str(ex.get("full_prompt", "")), raw_output))
        failure_keys.append(key)

    if not failure_items:
        return 0

    recovered = 0
    parsed_answers = asyncio.run(_extract_failed_answers_async(failure_items))
    for key, parsed in zip(failure_keys, parsed_answers):
        if parsed is None:
            continue
        raw_results[key]["predicted_answer"] = parsed
        raw_results[key]["extraction_recovered"] = True
        raw_results[key]["extraction_source"] = "llm_fallback"
        recovered += 1
    return recovered


def _run_inference_subprocess(model_path: str | dict, examples: list[dict], *, seed: int, mode: str):
    payload = {
        "model_path": model_path,
        "examples": examples,
        "seed": int(seed),
        "mode": str(mode),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as in_f:
        json.dump(payload, in_f)
        input_path = in_f.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as out_f:
        output_path = out_f.name

    env = os.environ.copy()
    env["PRAGMEGA_EVAL_WORKER"] = "1"
    env["PRAGMEGA_EVAL_INPUT"] = input_path
    env["PRAGMEGA_EVAL_OUTPUT"] = output_path
    # Avoid inheriting host CUDA runtime paths that break torch/sglang init.
    env["LD_LIBRARY_PATH"] = ""

    try:
        subprocess.run([sys.executable, os.path.abspath(__file__)], check=True, env=env)
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        return result["formatted"], result["meta"]
    finally:
        for path in (input_path, output_path):
            try:
                os.remove(path)
            except OSError:
                pass


def _maybe_run_worker() -> bool:
    if os.environ.get("PRAGMEGA_EVAL_WORKER") != "1":
        return False

    input_path = os.environ["PRAGMEGA_EVAL_INPUT"]
    output_path = os.environ["PRAGMEGA_EVAL_OUTPUT"]
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    formatted, meta = run_inference(
        payload["model_path"],
        payload["examples"],
        seed=int(payload["seed"]),
        mode=str(payload["mode"]),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"formatted": formatted, "meta": meta}, f)
    return True


def load_data() -> list[dict]:
    examples = load_pragmega_data(DATA_PATH, focus_phenomena=FOCUS_PHENOMENA)
    for idx, ex in enumerate(examples):
        ex["unique_key"] = f"{ex['phenomenon']}_{ex['item_id']}_{idx}"
        ex["correct_idx"] = ex["correct_answer_idx"]

    if NUM_EXAMPLES is not None and FOCUS_PHENOMENA is not None:
        final_set = []
        random.seed(SEEDS[0])
        for phenom in FOCUS_PHENOMENA:
            subset = [x for x in examples if x["phenomenon"] == phenom]
            final_set.extend(random.sample(subset, min(len(subset), NUM_EXAMPLES)))
        print(f"Loaded {len(final_set)} examples (sampled {NUM_EXAMPLES} per category)")
        return final_set

    phenom_counts = defaultdict(int)
    for ex in examples:
        phenom_counts[ex["phenomenon"]] += 1
    print(f"Loaded {len(examples)} examples")
    print(f"Phenomena: {dict(phenom_counts)}")
    return examples


def run_inference(model_path: str, examples: list[dict], *, seed: int, mode: str):
    message_batches = _build_message_batches(model_path, examples)
    tokenizer, prompt_texts = build_prompt_texts(
        model_path,
        message_batches,
        enable_thinking=bool(ENABLE_THINKING),
    )
    max_model_len = (
        int(MAX_PROMPT_LENGTH)
        + int(THINKING_BUDGET_TOKENS)
        + int(ANSWER_MAX_NEW_TOKENS)
        + 512
    )
    if EVAL_BACKEND == "sglang":
        runner = run_sglang_boxed_mcqa
    elif EVAL_BACKEND in {"transformers", "hf"}:
        runner = run_transformers_boxed_mcqa
    else:
        runner = run_vllm_boxed_mcqa
    runner_kwargs = dict(
        model_path=model_path,
        prompt_texts=prompt_texts,
        example_keys=[str(ex["unique_key"]) for ex in examples],
        thinking_budget_tokens=int(THINKING_BUDGET_TOKENS),
        answer_max_new_tokens=int(ANSWER_MAX_NEW_TOKENS),
        max_model_len=max_model_len,
        temperature=float(TEMPERATURE),
        do_sample=bool(DO_SAMPLE),
        top_p=float(TOP_P),
        top_k=int(TOP_K),
        min_p=float(MIN_P),
        seed=int(seed),
        gpu_memory_utilization=float(GPU_MEMORY_UTILIZATION),
        tensor_parallel_size=int(TENSOR_PARALLEL_SIZE),
        mode_label=f"{mode}:seed{seed}",
        use_tqdm=True,
    )
    if runner is run_vllm_boxed_mcqa and VLLM_MAX_NUM_SEQS > 0:
        runner_kwargs["max_num_seqs"] = int(VLLM_MAX_NUM_SEQS)
    if runner is run_vllm_boxed_mcqa and VLLM_MAX_NUM_BATCHED_TOKENS > 0:
        runner_kwargs["max_num_batched_tokens"] = int(VLLM_MAX_NUM_BATCHED_TOKENS)
    if EVAL_BACKEND == "sglang":
        runner_kwargs["message_batches"] = message_batches
    raw_results, meta = runner(**runner_kwargs)

    initial_failures = sum(
        1 for ex in examples if raw_results[str(ex["unique_key"])].get("predicted_answer") is None
    )
    recovered_failures = _recover_extraction_failures(examples, raw_results)

    formatted = {}
    think_token_counts = []
    for ex in examples:
        result = raw_results[str(ex["unique_key"])]
        final_output = str(result.get("raw_output", ""))
        reasoning_output = str(result.get("reasoning_output", ""))
        display_output = final_output
        think_token_counts.append(
            _count_reasoning_tokens(tokenizer, display_output, model_path=model_path)
        )
        formatted[str(ex["unique_key"])] = {
            "raw_output": display_output,
            "final_output": final_output,
            "reasoning_output": reasoning_output,
            "predicted_answer": result.get("predicted_answer"),
            "extraction_recovered": bool(result.get("extraction_recovered", False)),
            "extraction_source": str(result.get("extraction_source", "")),
        }

    correct = sum(
        1
        for ex in examples
        if _result_predicted(formatted[ex["unique_key"]]) is not None
        and _try_int_eq(_result_predicted(formatted[ex["unique_key"]]), ex["correct_answer_idx"])
    )
    failures = sum(1 for ex in examples if _result_predicted(formatted[ex["unique_key"]]) is None)
    acc = correct / len(examples) * 100 if examples else 0.0
    print(
        f"[{mode}] seed={seed} acc={correct}/{len(examples)} = {acc:.2f}% "
        f"(extraction failures: {failures}, recovered: {recovered_failures}, initial_failures: {initial_failures}, avg_think_tokens: "
        f"{(sum(think_token_counts) / len(think_token_counts)) if think_token_counts else 0.0:.2f}, "
        f"time: {meta['elapsed_seconds']:.1f}s)"
    )
    meta["avg_think_tokens"] = (
        float(sum(think_token_counts) / len(think_token_counts)) if think_token_counts else 0.0
    )
    meta["initial_extraction_failures"] = int(initial_failures)
    meta["recovered_extraction_failures"] = int(recovered_failures)
    meta["final_extraction_failures"] = int(failures)
    return formatted, meta


def _per_seed_accuracies(per_seed_results, examples):
    accs = []
    for sr in per_seed_results:
        correct = sum(
            1
            for ex in examples
            if _result_predicted(sr[ex["unique_key"]]) is not None
            and _try_int_eq(_result_predicted(sr[ex["unique_key"]]), ex["correct_answer_idx"])
        )
        accs.append(correct / len(examples) * 100 if examples else 0.0)
    return accs


def _per_seed_phenom_accuracies(per_seed_results, examples, phenomenon):
    sub = [ex for ex in examples if ex["phenomenon"] == phenomenon]
    if not sub:
        return None
    accs = []
    for sr in per_seed_results:
        correct = sum(
            1
            for ex in sub
            if _result_predicted(sr[ex["unique_key"]]) is not None
            and _try_int_eq(_result_predicted(sr[ex["unique_key"]]), ex["correct_answer_idx"])
        )
        accs.append(correct / len(sub) * 100)
    return accs


def _describe_model_spec_issue(spec) -> str:
    if not isinstance(spec, dict):
        return f"- {spec!r}: direct model spec"

    root_value = spec.get("model_root") or spec.get("full_model_root") or spec.get("checkpoint_root")
    adapter_value = spec.get("adapter_root")
    if root_value:
        root = Path(str(root_value)).expanduser()
        if not root.exists():
            return f"- {root}: path does not exist"
        if not root.is_dir():
            return f"- {root}: path is not a directory"
        checkpoint_dirs = sorted(root.glob("checkpoint-*"))
        archives = sorted(list(root.glob("checkpoint-*.tar")) + list(root.glob("checkpoint-*.tar.zst")) + list(root.glob("checkpoint-*.zip")))
        return (
            f"- {root}: found {len(checkpoint_dirs)} checkpoint dirs with "
            f"{sum(1 for p in checkpoint_dirs if (p / 'config.json').exists())} loadable config.json files; "
            f"found {len(archives)} checkpoint archives"
        )
    if adapter_value:
        root = Path(str(adapter_value)).expanduser()
        if not root.exists():
            return f"- {root}: path does not exist"
        checkpoint_dirs = sorted(root.glob("checkpoint-*")) if root.is_dir() else []
        return (
            f"- {root}: found {len(checkpoint_dirs)} adapter checkpoint dirs with "
            f"{sum(1 for p in checkpoint_dirs if (p / 'adapter_config.json').exists())} loadable adapter_config.json files"
        )
    return f"- {spec!r}: no model_root/checkpoint_root/adapter_root to expand"


def _require_model_specs(model_specs, configured_specs) -> None:
    if model_specs:
        return
    details = "\n".join(_describe_model_spec_issue(spec) for spec in configured_specs)
    raise RuntimeError(
        "No model checkpoints were resolved from MODELS.\n"
        f"Configured MODELS diagnostics:\n{details}"
    )


def main() -> None:
    if _maybe_run_worker():
        return

    examples = load_data()
    configured_specs = _configured_models()
    model_specs = expand_model_specs(configured_specs)
    _require_model_specs(model_specs, configured_specs)
    phenomena_in_data = sorted(set(ex["phenomenon"] for ex in examples))
    phenom_cols = [p for p in PHENOM_ORDER if p in phenomena_in_data]
    all_results = {}

    for model_path in model_specs:
        short = _short_name(model_path)
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {short}")
        print(f"{'=' * 60}")

        per_seed_results = []
        per_seed_meta = []
        for seed in SEEDS:
            if ISOLATE_EACH_RUN:
                results, meta = _run_inference_subprocess(model_path, examples, seed=seed, mode=short)
            else:
                results, meta = run_inference(model_path, examples, seed=seed, mode=short)
            per_seed_results.append(results)
            per_seed_meta.append(meta)

        overall_accs = _per_seed_accuracies(per_seed_results, examples)
        phenom_accs = {}
        for p in phenom_cols:
            accs = _per_seed_phenom_accuracies(per_seed_results, examples, p)
            if accs is not None:
                phenom_accs[p] = accs

        all_results[_model_key(model_path)] = {
            "per_seed_results": per_seed_results,
            "per_seed_meta": per_seed_meta,
            "overall_accs": overall_accs,
            "phenom_accs": phenom_accs,
        }

    phenom_headers = [PHENOM_DISPLAY.get(p, p) for p in phenom_cols]
    name_width = max(len(_short_name(c)) for c in model_specs)
    name_width = max(name_width, len("Model"))
    col_w = 12

    header_parts = [f"{'Model':<{name_width}}", f"{'Overall':>{col_w}}", f"{'ThinkTok':>{col_w}}"]
    header_parts.extend(f"{h:>{col_w}}" for h in phenom_headers)
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print(f"\n\n{'=' * len(header)}")
    print("PRAGMEGA EVALUATION RESULTS")
    print(f"Seeds: {SEEDS}  |  n={len(examples)} examples")
    print(f"backend={EVAL_BACKEND}, do_sample={DO_SAMPLE}, temperature={TEMPERATURE}, top_k={TOP_K}, min_p={MIN_P}, top_p={TOP_P}, thinking_budget={THINKING_BUDGET_TOKENS}")
    print(f"{'=' * len(header)}\n")
    print(header)
    print(sep)

    for model_path in model_specs:
        res = all_results[_model_key(model_path)]
        short = _short_name(model_path)
        overall_mean, overall_se = _compute_mean_stderr(res["overall_accs"])
        think_mean, think_se = _compute_mean_stderr([m["avg_think_tokens"] for m in res["per_seed_meta"]])
        row_parts = [
            f"{short:<{name_width}}",
            f"{_fmt(overall_mean, overall_se):>{col_w}}",
            f"{_fmt(think_mean, think_se):>{col_w}}",
        ]
        for p in phenom_cols:
            if p in res["phenom_accs"]:
                m, se = _compute_mean_stderr(res["phenom_accs"][p])
                row_parts.append(f"{_fmt(m, se):>{col_w}}")
            else:
                row_parts.append(f"{'N/A':>{col_w}}")
        print(" | ".join(row_parts))

    print(sep)
    print()

    json_out = {
        "task": "pragmega",
        "config": {
            "data_path": DATA_PATH,
            "seeds": SEEDS,
            "num_examples": len(examples),
            "models": model_specs,
            "eval_backend": EVAL_BACKEND,
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "min_p": MIN_P,
            "thinking_budget_tokens": THINKING_BUDGET_TOKENS,
            "answer_max_new_tokens": ANSWER_MAX_NEW_TOKENS,
        },
        "results": {},
    }
    for model_path in model_specs:
        res = all_results[_model_key(model_path)]
        overall_mean, overall_se = _compute_mean_stderr(res["overall_accs"])
        entry = {
            "overall": {
                "mean": round(overall_mean, 2),
                "stderr": round(overall_se, 2),
                "per_seed": [round(a, 2) for a in res["overall_accs"]],
            },
            "avg_think_tokens": {
                "mean": round(_compute_mean_stderr([m["avg_think_tokens"] for m in res["per_seed_meta"]])[0], 2),
                "stderr": round(_compute_mean_stderr([m["avg_think_tokens"] for m in res["per_seed_meta"]])[1], 2),
                "per_seed": [round(m["avg_think_tokens"], 2) for m in res["per_seed_meta"]],
            },
            "per_phenomenon": {},
        }
        for p in phenom_cols:
            if p in res["phenom_accs"]:
                m, se = _compute_mean_stderr(res["phenom_accs"][p])
                entry["per_phenomenon"][p] = {
                    "display_name": PHENOM_DISPLAY.get(p, p),
                    "mean": round(m, 2),
                    "stderr": round(se, 2),
                    "per_seed": [round(a, 2) for a in res["phenom_accs"][p]],
                }
        json_out["results"][_model_key(model_path)] = entry

    results_dir = str(get_eval_results_dir(ROOT, "pragmega"))
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = _write_example_outputs(results_dir, timestamp, model_specs, examples, all_results)

    json_out["config"]["outputs_dir"] = outputs_dir

    json_path = os.path.join(results_dir, f"pragmega_eval_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON results saved to: {json_path}")

    csv_path = os.path.join(results_dir, f"pragmega_eval_summary_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header_row = ["model", "overall_mean", "overall_stderr", "avg_think_tokens_mean", "avg_think_tokens_stderr"]
        for p in phenom_cols:
            display = PHENOM_DISPLAY.get(p, p)
            header_row.extend([f"{display}_mean", f"{display}_stderr"])
        writer.writerow(header_row)

        for model_path in model_specs:
            res = all_results[_model_key(model_path)]
            overall_mean, overall_se = _compute_mean_stderr(res["overall_accs"])
            think_mean, think_se = _compute_mean_stderr([m["avg_think_tokens"] for m in res["per_seed_meta"]])
            row = [
                _short_name(model_path),
                round(overall_mean, 2),
                round(overall_se, 2),
                round(think_mean, 2),
                round(think_se, 2),
            ]
            for p in phenom_cols:
                if p in res["phenom_accs"]:
                    m, se = _compute_mean_stderr(res["phenom_accs"][p])
                    row.extend([round(m, 2), round(se, 2)])
                else:
                    row.extend(["", ""])
            writer.writerow(row)
    print(f"CSV summary saved to: {csv_path}")
    print(f"Example outputs saved to: {outputs_dir}")


if __name__ == "__main__":
    main()
