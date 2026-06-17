#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import os
import re
import warnings
from datetime import datetime

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

from scripts.eval_common import get_eval_results_dir  # noqa: E402

load_dotenv()


def _configure_runtime_logging() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

    for logger_name in [
        "vllm",
        "vllm.engine",
        "vllm.executor",
        "vllm.worker",
        "vllm.core",
        "vllm.distributed",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", category=FutureWarning)


_configure_runtime_logging()


MODELS = ["Qwen/Qwen3-8B"]

TASK_NAME = "metoqa"
MAX_EXAMPLES = 0
SEEDS = [1]
EVAL_BACKEND = os.environ.get("EVAL_BACKEND", "vllm").strip().lower()
SHARD_INDEX = int(os.environ.get("METOQA_SHARD_INDEX", "0"))
SHARD_TOTAL = int(os.environ.get("METOQA_SHARD_TOTAL", "1"))

SYSTEM_PROMPT = (
    "Choose the best answer option based on the given context.\n"
    "Return only the option number in \\boxed{...}."
)

ENABLE_THINKING = True
MAX_PROMPT_LENGTH = 4096
THINKING_BUDGET_TOKENS = 2048
ANSWER_MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0
MIN_P = 0.0
GPU_MEMORY_UTILIZATION = 0.9
TENSOR_PARALLEL_SIZE = 1
ISOLATE_EACH_RUN = False

build_prompt_texts = None
expand_model_specs = None
resolve_model_spec = None
run_vllm_boxed_mcqa = None
run_sglang_boxed_mcqa = None
run_transformers_boxed_mcqa = None


def _ensure_eval_imports() -> None:
    global build_prompt_texts, expand_model_specs, resolve_model_spec
    global run_vllm_boxed_mcqa, run_sglang_boxed_mcqa, run_transformers_boxed_mcqa
    if all(x is not None for x in [
        build_prompt_texts,
        expand_model_specs,
        resolve_model_spec,
        run_vllm_boxed_mcqa,
        run_sglang_boxed_mcqa,
        run_transformers_boxed_mcqa,
    ]):
        return
    from src.eval.vllm_boxed_mcqa import (  # noqa: E402
        build_prompt_texts as _build_prompt_texts,
        expand_model_specs as _expand_model_specs,
        resolve_model_spec as _resolve_model_spec,
        run_vllm_boxed_mcqa as _run_vllm_boxed_mcqa,
    )
    from src.eval.sglang_boxed_mcqa import run_sglang_boxed_mcqa as _run_sglang_boxed_mcqa  # noqa: E402
    from src.eval.transformers_boxed_mcqa import run_transformers_boxed_mcqa as _run_transformers_boxed_mcqa  # noqa: E402

    build_prompt_texts = _build_prompt_texts
    expand_model_specs = _expand_model_specs
    resolve_model_spec = _resolve_model_spec
    run_vllm_boxed_mcqa = _run_vllm_boxed_mcqa
    run_sglang_boxed_mcqa = _run_sglang_boxed_mcqa
    run_transformers_boxed_mcqa = _run_transformers_boxed_mcqa


def _configured_models():
    eval_model_spec = os.environ.get("EVAL_MODEL_SPEC", "").strip()
    if eval_model_spec:
        if eval_model_spec.startswith(("{", "[")):
            parsed = json.loads(eval_model_spec)
            return parsed if isinstance(parsed, list) else [parsed]
        return [eval_model_spec]

    metoqa_model_spec = os.environ.get("METOQA_MODEL_SPEC", "").strip()
    if metoqa_model_spec:
        if metoqa_model_spec.startswith(("{", "[")):
            parsed = json.loads(metoqa_model_spec)
            return parsed if isinstance(parsed, list) else [parsed]
        return [metoqa_model_spec]

    return MODELS


def _is_openai_model(model_spec: str | dict) -> bool:
    if isinstance(model_spec, dict):
        raw = str(model_spec.get("model") or model_spec.get("base_model") or "").strip()
    else:
        raw = str(model_spec).strip()
    return raw.startswith("openai:")


def _resolve_openai_model_name(model_spec: str | dict) -> str:
    raw = str(model_spec.get("model") if isinstance(model_spec, dict) else model_spec).strip()
    return raw.split(":", 1)[1].strip()


def _resolve_data_path() -> str:
    candidates = [
        os.environ.get("METOQA_DATA_PATH", "").strip(),
        os.path.join(ROOT, "data", "eval", "metoqa", "metoqa.jsonl"),
        os.path.join(ROOT, "data", "eval", "task_14", "task_14.jsonl"),
        os.path.join(ROOT, "data", "task_14", "task_14.jsonl"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[1]


def _short_name(model_path: str | dict) -> str:
    if _is_openai_model(model_path):
        return _resolve_openai_model_name(model_path)
    _ensure_eval_imports()
    return str(resolve_model_spec(model_path)["display_name"])


def _compute_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.2f}\u00b1{std:.2f}"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _extract_boxed_number(text: str) -> int | None:
    patterns = [
        r"\\boxed\{\s*([1-4])\s*\}",
        r"\$\\boxed\{\s*([1-4])\s*\}\$",
        r"\b([1-4])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _load_data(task_name: str, data_path: str) -> list[dict]:
    examples: list[dict] = []
    skipped_invalid_label = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            options = list(row.get("options") or [])
            answer_text = str(row.get("correct answer", "")).strip()
            normalized_answer = _normalize_text(answer_text)
            if normalized_answer in {"", "nan", "none", "null"}:
                skipped_invalid_label += 1
                continue
            answer_idx = None
            for opt_idx, option in enumerate(options, start=1):
                if _normalize_text(option) == normalized_answer:
                    answer_idx = opt_idx
                    break
            if answer_idx is None:
                raise ValueError(f"Could not map correct answer to options for id={row.get('id')}")
            examples.append(
                {
                    "id": str(row.get("id", idx)),
                    "unique_key": str(row.get("id", idx)),
                    "pretext": str(row.get("pretext", "")).strip(),
                    "options": [str(x) for x in options],
                    "correct_answer_idx": int(answer_idx),
                    "correct_answer_text": answer_text,
                }
            )
    if SHARD_TOTAL > 1:
        if SHARD_INDEX < 0 or SHARD_INDEX >= SHARD_TOTAL:
            raise ValueError(f"Invalid MetoQA shard {SHARD_INDEX}/{SHARD_TOTAL}")
        before_shard = len(examples)
        examples = [
            ex for idx, ex in enumerate(examples)
            if idx % int(SHARD_TOTAL) == int(SHARD_INDEX)
        ]
        print(
            f"Shard {SHARD_INDEX}/{SHARD_TOTAL}: kept {len(examples)}/{before_shard} "
            f"{task_name.upper()} examples"
        )
    if MAX_EXAMPLES > 0:
        examples = examples[: int(MAX_EXAMPLES)]
    print(f"Loaded {len(examples)} {task_name.upper()} examples from {data_path}")
    if skipped_invalid_label:
        print(f"Skipped {skipped_invalid_label} {task_name.upper()} rows with invalid gold labels")
    return examples


def _build_user_prompt(ex: dict) -> str:
    options_block = "\n".join(
        f"{idx}. {option.strip()}" for idx, option in enumerate(ex["options"], start=1)
    )
    return (
        f"{ex['pretext'].strip()}\n"
        f"Options:\n{options_block}\n\n"
        "Answer with the best option number only.\n"
        "Write your final answer in \\boxed{...}."
    )


def _build_openai_prompt(ex: dict) -> str:
    return f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(ex)}"


def _write_example_outputs(task_name: str, results_dir: str, timestamp: str, model_specs, examples, all_results) -> str:
    outputs_dir = os.path.join(results_dir, f"{task_name}_eval_outputs_{timestamp}")
    os.makedirs(outputs_dir, exist_ok=True)

    for model_path in model_specs:
        short = _short_name(model_path)
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", short).strip("._") or "model"
        res = all_results[model_path]
        for seed_idx, seed in enumerate(SEEDS):
            seed_results = res["per_seed_results"][seed_idx]
            out_path = os.path.join(outputs_dir, f"{safe_model}_seed{seed}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for ex in examples:
                    raw_output, predicted_answer = seed_results[ex["unique_key"]]
                    row = {
                        "model": short,
                        "seed": seed,
                        "id": ex["id"],
                        "pretext": ex["pretext"],
                        "options": ex["options"],
                        "gold_index": ex["correct_answer_idx"],
                        "gold_text": ex["correct_answer_text"],
                        "predicted_answer": predicted_answer,
                        "correct": bool(
                            predicted_answer is not None
                            and int(predicted_answer) == int(ex["correct_answer_idx"])
                        ),
                        "raw_output": raw_output,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return outputs_dir


def run_inference(model_path: str, examples: list[dict], *, seed: int, mode: str):
    if _is_openai_model(model_path):
        from openai import OpenAI
        from src.eval.open_judge_eval import call_openai_responses

        model_name = _resolve_openai_model_name(model_path)
        client = OpenAI()
        formatted = {}
        think_token_counts = []
        start = datetime.now()

        for ex in tqdm(examples, total=len(examples), desc=f"{mode}:seed{seed}", unit="ex"):
            prompt_text = _build_openai_prompt(ex)
            raw_output = call_openai_responses(
                client=client,
                model_name=model_name,
                prompt_text=prompt_text,
                max_tokens=max(1, int(THINKING_BUDGET_TOKENS)),
                reasoning_effort="medium" if ENABLE_THINKING else None,
            )
            predicted_answer = _extract_boxed_number(raw_output)
            if predicted_answer is None:
                if "</think>" in raw_output:
                    thinking_only = raw_output[: raw_output.rfind("</think>")]
                else:
                    thinking_only = raw_output
                answer_tail = call_openai_responses(
                    client=client,
                    model_name=model_name,
                    prompt_text=prompt_text
                    + thinking_only
                    + "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n\\boxed{",
                    max_tokens=max(1, int(ANSWER_MAX_NEW_TOKENS)),
                    reasoning_effort="medium" if ENABLE_THINKING else None,
                )
                raw_output = (
                    raw_output
                    + "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n\\boxed{"
                    + answer_tail
                )
                predicted_answer = _extract_boxed_number(raw_output)

            think_text = ""
            if "<think>" in raw_output and "</think>" in raw_output:
                think_text = raw_output.split("<think>", 1)[-1].split("</think>", 1)[0]
            think_token_counts.append(len(think_text.split()) if think_text.strip() else 0)
            formatted[str(ex["unique_key"])] = (raw_output, predicted_answer)

        elapsed = (datetime.now() - start).total_seconds()
        correct = sum(
            1
            for ex in examples
            if formatted[ex["unique_key"]][1] is not None
            and int(formatted[ex["unique_key"]][1]) == int(ex["correct_answer_idx"])
        )
        failures = sum(1 for ex in examples if formatted[ex["unique_key"]][1] is None)
        acc = correct / len(examples) * 100 if examples else 0.0
        print(
            f"[{mode}] seed={seed} acc={correct}/{len(examples)} = {acc:.2f}% "
            f"(extraction failures: {failures}, avg_think_tokens: "
            f"{(sum(think_token_counts) / len(think_token_counts)) if think_token_counts else 0.0:.2f}, "
            f"time: {elapsed:.1f}s)"
        )
        return formatted, {
            "elapsed_seconds": float(elapsed),
            "avg_think_tokens": float(sum(think_token_counts) / len(think_token_counts)) if think_token_counts else 0.0,
        }

    _ensure_eval_imports()
    message_batches = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(ex)},
        ]
        for ex in examples
    ]
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
    raw_results, meta = runner(
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
        answer_parser=_extract_boxed_number,
    )

    formatted = {}
    think_token_counts = []
    for ex in examples:
        result = raw_results[str(ex["unique_key"])]
        raw_output = str(result.get("raw_output", ""))
        think_text = ""
        if "<think>" in raw_output and "</think>" in raw_output:
            think_text = raw_output.split("<think>", 1)[-1].split("</think>", 1)[0]
        think_token_counts.append(
            len(tokenizer.encode(think_text, add_special_tokens=False)) if think_text.strip() else 0
        )
        formatted[str(ex["unique_key"])] = (
            raw_output,
            result.get("predicted_answer"),
        )

    correct = sum(
        1
        for ex in examples
        if formatted[ex["unique_key"]][1] is not None
        and int(formatted[ex["unique_key"]][1]) == int(ex["correct_answer_idx"])
    )
    failures = sum(1 for ex in examples if formatted[ex["unique_key"]][1] is None)
    acc = correct / len(examples) * 100 if examples else 0.0
    print(
        f"[{mode}] seed={seed} acc={correct}/{len(examples)} = {acc:.2f}% "
        f"(extraction failures: {failures}, avg_think_tokens: "
        f"{(sum(think_token_counts) / len(think_token_counts)) if think_token_counts else 0.0:.2f}, "
        f"time: {meta['elapsed_seconds']:.1f}s)"
    )
    meta["avg_think_tokens"] = (
        float(sum(think_token_counts) / len(think_token_counts)) if think_token_counts else 0.0
    )
    return formatted, meta


def _per_seed_accuracies(per_seed_results, examples):
    accs = []
    for sr in per_seed_results:
        correct = sum(
            1
            for ex in examples
            if sr[ex["unique_key"]][1] is not None
            and int(sr[ex["unique_key"]][1]) == int(ex["correct_answer_idx"])
        )
        accs.append(correct / len(examples) * 100 if examples else 0.0)
    return accs


def _evaluate_metoqa() -> tuple[dict, str]:
    _ensure_eval_imports()
    task_name = TASK_NAME
    data_path = _resolve_data_path()
    examples = _load_data(TASK_NAME, data_path)
    model_specs = expand_model_specs(_configured_models())
    all_results = {}

    for model_path in tqdm(model_specs, total=len(model_specs), desc=f"{task_name}:models", unit="model"):
        short = _short_name(model_path)
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {short}")
        print(f"{'=' * 60}")

        per_seed_results = []
        per_seed_meta = []
        for seed in tqdm(SEEDS, total=len(SEEDS), desc=f"{short}:seeds", unit="seed"):
            results, meta = run_inference(model_path, examples, seed=seed, mode=short)
            per_seed_results.append(results)
            per_seed_meta.append(meta)

        overall_accs = _per_seed_accuracies(per_seed_results, examples)
        all_results[model_path] = {
            "per_seed_results": per_seed_results,
            "per_seed_meta": per_seed_meta,
            "overall_accs": overall_accs,
        }

    name_width = max(len(_short_name(c)) for c in model_specs)
    name_width = max(name_width, len("Model"))
    col_w = 12
    header = " | ".join(
        [
            f"{'Model':<{name_width}}",
            f"{'Acc':>{col_w}}",
            f"{'ThinkTok':>{col_w}}",
        ]
    )
    sep = "-" * len(header)

    print(f"\n\n{'=' * len(header)}")
    print(f"{task_name.upper()} EVALUATION RESULTS")
    print(f"Seeds: {SEEDS}  |  n={len(examples)} examples")
    print(
        f"backend={EVAL_BACKEND}, do_sample={DO_SAMPLE}, temperature={TEMPERATURE}, top_k={TOP_K}, "
        f"min_p={MIN_P}, top_p={TOP_P}, thinking_budget={THINKING_BUDGET_TOKENS}"
    )
    print(f"{'=' * len(header)}\n")
    print(header)
    print(sep)

    json_out = {
        "task": task_name,
        "config": {
            "data_path": data_path,
            "task": TASK_NAME,
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
            "isolate_each_run": ISOLATE_EACH_RUN,
            "shard_index": int(SHARD_INDEX),
            "shard_total": int(SHARD_TOTAL),
        },
        "results": {},
    }

    for model_path in model_specs:
        res = all_results[model_path]
        short = _short_name(model_path)
        acc_mean, acc_std = _compute_mean_std(res["overall_accs"])
        think_mean, think_std = _compute_mean_std([m["avg_think_tokens"] for m in res["per_seed_meta"]])
        print(
            " | ".join(
                [
                    f"{short:<{name_width}}",
                    f"{_fmt(acc_mean, acc_std):>{col_w}}",
                    f"{_fmt(think_mean, think_std):>{col_w}}",
                ]
            )
        )
        json_out["results"][model_path] = {
            "overall": {
                "mean": round(acc_mean, 2),
                "std": round(acc_std, 2),
                "per_seed": [round(a, 2) for a in res["overall_accs"]],
            },
            "avg_think_tokens": {
                "mean": round(think_mean, 2),
                "std": round(think_std, 2),
                "per_seed": [round(m["avg_think_tokens"], 2) for m in res["per_seed_meta"]],
            },
        }

    print(sep)
    print()

    results_dir = str(get_eval_results_dir(ROOT, task_name))
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = _write_example_outputs(task_name, results_dir, timestamp, model_specs, examples, all_results)
    json_out["config"]["outputs_dir"] = outputs_dir

    json_path = os.path.join(results_dir, f"{task_name}_eval_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(results_dir, f"{task_name}_eval_summary_{timestamp}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "model,acc_mean,acc_std,avg_think_tokens_mean,avg_think_tokens_std,"
            "num_examples,num_correct,num_extraction_failures,shard_index,shard_total\n"
        )
        for model_path in model_specs:
            res = all_results[model_path]
            acc_mean, acc_std = _compute_mean_std(res["overall_accs"])
            think_mean, think_std = _compute_mean_std([m["avg_think_tokens"] for m in res["per_seed_meta"]])
            first_seed_results = res["per_seed_results"][0] if res["per_seed_results"] else {}
            num_correct = sum(
                1
                for ex in examples
                if first_seed_results.get(ex["unique_key"], ("", None))[1] is not None
                and int(first_seed_results[ex["unique_key"]][1]) == int(ex["correct_answer_idx"])
            )
            num_failures = sum(
                1
                for ex in examples
                if first_seed_results.get(ex["unique_key"], ("", None))[1] is None
            )
            f.write(
                f"{_short_name(model_path)},{acc_mean:.2f},{acc_std:.2f},"
                f"{think_mean:.2f},{think_std:.2f},{len(examples)},{num_correct},"
                f"{num_failures},{int(SHARD_INDEX)},{int(SHARD_TOTAL)}\n"
            )

    print(f"JSON results saved to: {json_path}")
    print(f"CSV summary saved to: {csv_path}")
    print(f"Example outputs saved to: {outputs_dir}")
    return json_out, json_path


def main() -> None:
    print(f"\n{'#' * 80}")
    print("Starting MetoQA evaluation")
    print(f"{'#' * 80}")
    _evaluate_metoqa()


if __name__ == "__main__":
    main()
