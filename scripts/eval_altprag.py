#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.eval_common import get_eval_results_dir  # noqa: E402

from src.eval.altprag import (  # noqa: E402
    ALTPRAG_DATASET_NAME,
    ALTPRAG_EARLY_STOPPING_TEXT,
    ALTPRAG_MAXIM_ORDER,
    ALTPRAG_TASK_DISPLAY,
    ALTPRAG_TASK_ORDER,
    AltPragExample,
    build_altprag_system_prompt,
    build_altprag_user_prompt,
    extract_altprag_maxim,
    load_altprag_examples,
    normalize_altprag_answer,
)
from src.eval.open_judge_eval import (  # noqa: E402
    compute_mean_stderr,
    run_transformers_open_ended,
    run_vllm_open_ended,
    strip_code_fence,
)
from src.eval.sglang_boxed_mcqa import run_sglang_open_ended  # noqa: E402
from src.eval.vllm_boxed_mcqa import build_prompt_texts, expand_model_specs, resolve_model_spec  # noqa: E402


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

    altprag_model_spec = os.environ.get("ALTPRAG_MODEL_SPEC", "").strip()
    if altprag_model_spec:
        if altprag_model_spec.startswith(("{", "[")):
            parsed = json.loads(altprag_model_spec)
            return parsed if isinstance(parsed, list) else [parsed]
        return [altprag_model_spec]

    return MODELS

SEEDS = [
    int(x.strip())
    for x in os.environ.get("ALTPRAG_SEEDS", "1").split(",")
    if x.strip()
]
EVAL_BACKEND = os.environ.get("EVAL_BACKEND", "vllm").strip().lower()

DATASET_NAME = ALTPRAG_DATASET_NAME
HF_SPLIT = "test"
TASK_MODE = "candidate1"
MAX_EXAMPLES = 0
SUBSET_SEED = 42
ENABLE_THINKING = True
MAX_PROMPT_LENGTH = 8192
THINKING_BUDGET_TOKENS = 2048
ANSWER_MAX_NEW_TOKENS = int(os.environ.get("ALTPRAG_ANSWER_MAX_NEW_TOKENS", "456"))
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0
MIN_P = 0.0
GPU_MEMORY_UTILIZATION = 0.9
TENSOR_PARALLEL_SIZE = 1
ISOLATE_EACH_RUN = True

JUDGE_MODEL = os.environ.get("ALTPRAG_JUDGE_MODEL", "gpt-4.1")
JUDGE_TEMPERATURE = float(os.environ.get("ALTPRAG_JUDGE_TEMPERATURE", "0.0"))
JUDGE_MAX_TOKENS = int(os.environ.get("ALTPRAG_JUDGE_MAX_TOKENS", "300"))
JUDGE_TIMEOUT = float(os.environ.get("ALTPRAG_JUDGE_TIMEOUT", "180.0"))
JUDGE_MAX_RETRIES = int(os.environ.get("ALTPRAG_JUDGE_MAX_RETRIES", "3"))


class AltPragJudge:
    def __init__(
        self,
        *,
        model_name: str = "gpt-4.1",
        temperature: float = 0.3,
        max_tokens: int = 300,
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(timeout=float(timeout))
        self.model_name = str(model_name)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self._cache: dict[tuple[str, str], dict[str, object]] = {}

    def _build_prompt(self, example: AltPragExample, candidate_answer: str) -> str:
        return (
            "You are an expert evaluator of pragmatic reasoning responses.\n\n"
            "The tested model saw a conversation structure like this:\n"
            f"\"context\": {json.dumps(example.context, ensure_ascii=False)}\n"
            f"\"dialogue root\": {json.dumps(example.root, ensure_ascii=False)}\n"
            f"\"candidate_sentence_1\": {json.dumps(example.candidate_sentence_1, ensure_ascii=False)}\n"
            f"\"candidate_sentence_2\": {json.dumps(example.candidate_sentence_2, ensure_ascii=False)}\n\n"
            "The task was to explain: What is the intention behind candidate_sentence_1? "
            "Why or when might someone prefer candidate_sentence_1 over candidate_sentence_2?\n\n"
            f"Reference golden intention: {json.dumps(example.gold_intention, ensure_ascii=False)}\n\n"
            f"Model response: {json.dumps(candidate_answer, ensure_ascii=False)}\n\n"
            "Score the model response on a scale of 1 to 10, or Invalid, where:\n"
            "- 10: The response perfectly captures the pragmatic intention and contextual preference.\n"
            "- 1: The response poorly captures the intention or misses the pragmatic contrast.\n"
            "- Invalid: The response is nonsense, empty, or unusable.\n\n"
            "Return ONLY a JSON object with this exact schema:\n"
            "{\n"
            '  "score": <integer 1-10 or "Invalid">,\n'
            '  "reason": "<brief explanation, <= 25 words>"\n'
            "}\n"
        )

    @staticmethod
    def _parse(raw_text: str) -> dict[str, object]:
        raw = strip_code_fence(str(raw_text or "")).strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            return {
                "score": None,
                "invalid": True,
                "reason": f"judge_parse_error: {(raw or 'EMPTY')[:120]}",
                "raw": raw,
            }

        score = parsed.get("score")
        reason = str(parsed.get("reason") or "").strip()

        if isinstance(score, int) and 1 <= score <= 10:
            return {"score": int(score), "invalid": False, "reason": reason, "raw": raw}

        if isinstance(score, str):
            cleaned = score.strip()
            if cleaned.lower() == "invalid":
                return {"score": None, "invalid": True, "reason": reason or "Invalid", "raw": raw}
            try:
                numeric = int(cleaned)
            except Exception:
                numeric = None
            if numeric is not None and 1 <= numeric <= 10:
                return {"score": int(numeric), "invalid": False, "reason": reason, "raw": raw}

        return {
            "score": None,
            "invalid": True,
            "reason": reason or f"judge_invalid_score: {(raw or 'EMPTY')[:120]}",
            "raw": raw,
        }

    def score(self, *, example: AltPragExample, candidate_answer: str | None) -> dict[str, object]:
        answer = str(candidate_answer or "").strip()
        if not answer:
            return {
                "score": None,
                "invalid": True,
                "reason": "empty_answer",
                "raw": "empty_answer",
            }

        cache_key = (str(example.id), answer)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        prompt = self._build_prompt(example, answer)
        result: dict[str, object] | None = None
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a careful evaluator of pragmatic reasoning responses."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                result = self._parse(str(response.choices[0].message.content or ""))
                if not str(result.get("reason") or "").startswith("judge_parse_error"):
                    break
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                if attempt >= self.max_retries:
                    break

        if result is None:
            result = {
                "score": None,
                "invalid": True,
                "reason": f"judge_api_error: {(last_error or 'UNKNOWN')[:120]}",
                "raw": last_error or "",
            }

        self._cache[cache_key] = dict(result)
        return dict(result)


def _short_name(model_path: str | dict) -> str:
    return str(resolve_model_spec(model_path)["display_name"])


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_") or "model"


def _fmt(mean: float, stderr: float) -> str:
    return f"{mean:.2f}\u00b1{stderr:.2f}"


def _load_examples_and_stats():
    return load_altprag_examples(
        DATASET_NAME,
        hf_split=HF_SPLIT,
        task_mode=TASK_MODE,
        max_examples=int(MAX_EXAMPLES),
        subset_seed=int(SUBSET_SEED),
    )


def _compute_group_mean(examples, results: dict[str, dict], *, attr: str, keys: list[str]) -> dict[str, float]:
    grouped: dict[str, list[int]] = {key: [] for key in keys}
    for ex in examples:
        score = results[str(ex.id)].get("judge_score")
        if not isinstance(score, int):
            continue
        group_key = str(getattr(ex, attr))
        grouped.setdefault(group_key, []).append(int(score))
    return {key: (sum(values) / len(values)) if values else 0.0 for key, values in grouped.items()}


def _compute_metrics(results: dict[str, dict], examples) -> dict[str, object]:
    total = len(examples)
    empty_answers = 0
    maxim_extracted = 0
    maxim_correct_total = 0
    maxim_correct_extracted = 0
    valid_scores: list[int] = []

    for ex in examples:
        row = results[str(ex.id)]
        answer = str(row.get("normalized_answer") or "").strip()
        if not answer:
            empty_answers += 1

        predicted_maxim = row.get("predicted_maxim")
        if predicted_maxim is not None:
            maxim_extracted += 1
            if predicted_maxim == ex.gold_maxim:
                maxim_correct_extracted += 1
        if bool(row.get("maxim_correct", False)):
            maxim_correct_total += 1

        score = row.get("judge_score")
        if isinstance(score, int):
            valid_scores.append(int(score))

    mean_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
    std_score = statistics.pstdev(valid_scores) if len(valid_scores) > 1 else 0.0
    min_score = min(valid_scores) if valid_scores else 0.0
    max_score = max(valid_scores) if valid_scores else 0.0
    median_score = statistics.median(valid_scores) if valid_scores else 0.0

    return {
        "mean_score": float(mean_score),
        "std_score": float(std_score),
        "min_score": float(min_score),
        "max_score": float(max_score),
        "median_score": float(median_score),
        "valid_judge_count": int(len(valid_scores)),
        "valid_judge_rate": (len(valid_scores) / total * 100.0) if total else 0.0,
        "invalid_judge_rate": ((total - len(valid_scores)) / total * 100.0) if total else 0.0,
        "extraction_failure_rate": (empty_answers / total * 100.0) if total else 0.0,
        "budget_forced_rate": (sum(1 for x in results.values() if bool(x.get("budget_forced"))) / total * 100.0) if total else 0.0,
        "maxim_extraction_rate": (maxim_extracted / total * 100.0) if total else 0.0,
        "maxim_accuracy": (maxim_correct_total / total * 100.0) if total else 0.0,
        "maxim_accuracy_on_extracted": (maxim_correct_extracted / maxim_extracted * 100.0) if maxim_extracted else 0.0,
        "per_task_mean_score": _compute_group_mean(examples, results, attr="task", keys=ALTPRAG_TASK_ORDER),
        "per_maxim_mean_score": _compute_group_mean(examples, results, attr="gold_maxim", keys=ALTPRAG_MAXIM_ORDER),
    }


def _write_seed_outputs(*, output_path: Path, model_path: str | dict, seed: int, examples, results: dict[str, dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row = results[str(ex.id)]
            out = {
                "model": _short_name(model_path),
                "seed": int(seed),
                "id": str(ex.id),
                "split": ex.split,
                "row_index": int(ex.row_index),
                "task": ex.task,
                "source_candidate": int(ex.source_candidate),
                "context": ex.context,
                "root": ex.root,
                "candidate_sentence_1": ex.candidate_sentence_1,
                "candidate_sentence_2": ex.candidate_sentence_2,
                "gold_intention": ex.gold_intention,
                "gold_maxim": ex.gold_maxim,
                "predicted_answer": str(row.get("normalized_answer") or ""),
                "predicted_maxim": row.get("predicted_maxim"),
                "maxim_correct": bool(row.get("maxim_correct", False)),
                "judge_score": row.get("judge_score"),
                "judge_invalid": bool(row.get("judge_invalid", False)),
                "judge_reason": str(row.get("judge_reason") or ""),
                "judge_raw": str(row.get("judge_raw") or ""),
                "budget_forced": bool(row.get("budget_forced", False)),
                "force_reason": str(row.get("force_reason") or ""),
                "raw_output": str(row.get("raw_output") or ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def run_inference(model_path: str | dict, examples, *, seed: int, mode: str, judge: AltPragJudge):
    system_prompt = build_altprag_system_prompt()
    message_batches = []
    for ex in examples:
        messages = []
        if str(system_prompt).strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": build_altprag_user_prompt(ex)})
        message_batches.append(messages)

    _, prompt_texts = build_prompt_texts(
        model_path,
        message_batches,
        enable_thinking=bool(ENABLE_THINKING),
    )
    max_model_len = int(MAX_PROMPT_LENGTH) + int(THINKING_BUDGET_TOKENS) + int(ANSWER_MAX_NEW_TOKENS) + 512
    if EVAL_BACKEND == "sglang":
        runner = run_sglang_open_ended
    elif EVAL_BACKEND in {"transformers", "hf"}:
        runner = run_transformers_open_ended
    else:
        runner = run_vllm_open_ended
    results, meta = runner(
        model_path=model_path,
        prompt_texts=prompt_texts,
        example_keys=[str(ex.id) for ex in examples],
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
        early_stopping_text=ALTPRAG_EARLY_STOPPING_TEXT,
        mode_label=f"{mode}:seed{seed}",
        use_tqdm=True,
    )

    for ex in examples:
        row = results[str(ex.id)]
        normalized = normalize_altprag_answer(row.get("answer_text"))
        row["normalized_answer"] = normalized
        predicted_maxim = extract_altprag_maxim(normalized)
        row["predicted_maxim"] = predicted_maxim
        row["maxim_correct"] = bool(predicted_maxim == ex.gold_maxim) if predicted_maxim is not None else False
        judge_result = judge.score(example=ex, candidate_answer=normalized)
        row["judge_score"] = judge_result.get("score")
        row["judge_invalid"] = bool(judge_result.get("invalid", False))
        row["judge_reason"] = str(judge_result.get("reason") or "")
        row["judge_raw"] = str(judge_result.get("raw") or "")

    metrics = _compute_metrics(results, examples)
    metrics["elapsed_seconds"] = float(meta["elapsed_seconds"])
    print(
        f"[{mode}] seed={seed} mean={metrics['mean_score']:.2f} "
        f"valid={metrics['valid_judge_rate']:.2f}% invalid={metrics['invalid_judge_rate']:.2f}% "
        f"maxim_acc={metrics['maxim_accuracy']:.2f}% forced={metrics['budget_forced_rate']:.2f}% "
        f"time={metrics['elapsed_seconds']:.1f}s"
    )
    return results, metrics


def _run_inference_subprocess(model_path: str | dict, *, seed: int, mode: str):
    payload = {"model_path": model_path, "seed": int(seed), "mode": str(mode)}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as in_f:
        json.dump(payload, in_f)
        input_path = in_f.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as out_f:
        output_path = out_f.name

    env = os.environ.copy()
    env["ALTPRAG_EVAL_WORKER"] = "1"
    env["ALTPRAG_EVAL_INPUT"] = input_path
    env["ALTPRAG_EVAL_OUTPUT"] = output_path
    try:
        subprocess.run([sys.executable, os.path.abspath(__file__)], check=True, env=env)
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["results"], data["metrics"]
    finally:
        for path in (input_path, output_path):
            try:
                os.remove(path)
            except OSError:
                pass


def _maybe_run_worker() -> bool:
    if os.environ.get("ALTPRAG_EVAL_WORKER") != "1":
        return False

    input_path = os.environ["ALTPRAG_EVAL_INPUT"]
    output_path = os.environ["ALTPRAG_EVAL_OUTPUT"]
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    examples, _ = _load_examples_and_stats()
    judge = AltPragJudge(
        model_name=JUDGE_MODEL,
        temperature=float(JUDGE_TEMPERATURE),
        max_tokens=int(JUDGE_MAX_TOKENS),
        timeout=float(JUDGE_TIMEOUT),
        max_retries=int(JUDGE_MAX_RETRIES),
    )
    results, metrics = run_inference(
        payload["model_path"],
        examples,
        seed=int(payload["seed"]),
        mode=str(payload["mode"]),
        judge=judge,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "metrics": metrics}, f, ensure_ascii=False)
    return True


def main() -> None:
    if _maybe_run_worker():
        return

    examples, data_stats = _load_examples_and_stats()
    if not examples:
        raise RuntimeError("No AltPrag examples loaded")

    configured_specs = _configured_models()
    model_specs = expand_model_specs(configured_specs)
    judge = AltPragJudge(
        model_name=JUDGE_MODEL,
        temperature=float(JUDGE_TEMPERATURE),
        max_tokens=int(JUDGE_MAX_TOKENS),
        timeout=float(JUDGE_TIMEOUT),
        max_retries=int(JUDGE_MAX_RETRIES),
    )

    all_model_payloads: dict[str, dict[str, object]] = {}
    for model_path in model_specs:
        short = _short_name(model_path)
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {short}")
        print(f"{'=' * 60}")

        per_seed_metrics = []
        per_seed_outputs = []
        for seed in SEEDS:
            if ISOLATE_EACH_RUN:
                outputs, metrics = _run_inference_subprocess(model_path, seed=int(seed), mode=short)
            else:
                outputs, metrics = run_inference(model_path, examples, seed=int(seed), mode=short, judge=judge)
            per_seed_outputs.append(outputs)
            per_seed_metrics.append(metrics)

        mean_score, stderr_score = compute_mean_stderr([float(m["mean_score"]) for m in per_seed_metrics])
        mean_valid, stderr_valid = compute_mean_stderr([float(m["valid_judge_rate"]) for m in per_seed_metrics])
        mean_invalid, stderr_invalid = compute_mean_stderr([float(m["invalid_judge_rate"]) for m in per_seed_metrics])
        mean_fail, stderr_fail = compute_mean_stderr([float(m["extraction_failure_rate"]) for m in per_seed_metrics])
        mean_forced, stderr_forced = compute_mean_stderr([float(m["budget_forced_rate"]) for m in per_seed_metrics])
        mean_maxim_acc, stderr_maxim_acc = compute_mean_stderr([float(m["maxim_accuracy"]) for m in per_seed_metrics])
        mean_maxim_ex, stderr_maxim_ex = compute_mean_stderr([float(m["maxim_extraction_rate"]) for m in per_seed_metrics])

        per_task_summary = {}
        for task_name in ALTPRAG_TASK_ORDER:
            task_mean, task_stderr = compute_mean_stderr(
                [float(m["per_task_mean_score"].get(task_name, 0.0)) for m in per_seed_metrics]
            )
            per_task_summary[task_name] = {"mean": float(task_mean), "stderr": float(task_stderr)}

        per_maxim_summary = {}
        for maxim_name in ALTPRAG_MAXIM_ORDER:
            maxim_mean, maxim_stderr = compute_mean_stderr(
                [float(m["per_maxim_mean_score"].get(maxim_name, 0.0)) for m in per_seed_metrics]
            )
            per_maxim_summary[maxim_name] = {"mean": float(maxim_mean), "stderr": float(maxim_stderr)}

        all_model_payloads[short] = {
            "model_path": model_path,
            "per_seed_metrics": per_seed_metrics,
            "per_seed_outputs": per_seed_outputs,
            "summary": {
                "mean_score": float(mean_score),
                "stderr_score": float(stderr_score),
                "valid_judge_rate": float(mean_valid),
                "stderr_valid_judge_rate": float(stderr_valid),
                "invalid_judge_rate": float(mean_invalid),
                "stderr_invalid_judge_rate": float(stderr_invalid),
                "extraction_failure_rate": float(mean_fail),
                "stderr_extraction_failure_rate": float(stderr_fail),
                "budget_forced_rate": float(mean_forced),
                "stderr_budget_forced_rate": float(stderr_forced),
                "maxim_accuracy": float(mean_maxim_acc),
                "stderr_maxim_accuracy": float(stderr_maxim_acc),
                "maxim_extraction_rate": float(mean_maxim_ex),
                "stderr_maxim_extraction_rate": float(stderr_maxim_ex),
                "per_task_mean_score": per_task_summary,
                "per_maxim_mean_score": per_maxim_summary,
            },
        }

        print(
            f"{short}: score={_fmt(mean_score, stderr_score)} | valid={_fmt(mean_valid, stderr_valid)} | "
            f"invalid={_fmt(mean_invalid, stderr_invalid)} | fail={_fmt(mean_fail, stderr_fail)} | "
            f"maxim_acc={_fmt(mean_maxim_acc, stderr_maxim_acc)}"
        )

    name_width = max(max(len(str(name)) for name in all_model_payloads), len("Model"))
    col_w = 14
    task_cols = [task_name for task_name in ALTPRAG_TASK_ORDER]
    header_cols = [
        f"{'Model':<{name_width}}",
        f"{'Score':>{col_w}}",
        f"{'Valid':>{col_w}}",
        f"{'Invalid':>{col_w}}",
        f"{'Fail':>{col_w}}",
        f"{'Forced':>{col_w}}",
        f"{'MaximAcc':>{col_w}}",
    ] + [f"{ALTPRAG_TASK_DISPLAY.get(task_name, task_name):>{col_w}}" for task_name in task_cols]
    header = " | ".join(header_cols)
    sep = "-" * len(header)

    print(f"\n\n{'=' * len(header)}")
    print("ALTPRAG EVALUATION RESULTS")
    print(f"Seeds: {SEEDS}  |  n={len(examples)} examples")
    print(
        f"dataset={DATASET_NAME}, split={HF_SPLIT}, task_mode={TASK_MODE}, "
        f"task_counts={dict(data_stats.task_counts)}, maxim_counts={dict(data_stats.maxim_counts)}"
    )
    print(
        f"backend={EVAL_BACKEND}, do_sample={DO_SAMPLE}, temperature={TEMPERATURE}, top_k={TOP_K}, min_p={MIN_P}, "
        f"top_p={TOP_P}, thinking_budget={THINKING_BUDGET_TOKENS}, judge={JUDGE_MODEL}"
    )
    print(f"{'=' * len(header)}\n")
    print(header)
    print(sep)

    for short, payload in all_model_payloads.items():
        summary = payload["summary"]
        row = [
            f"{short:<{name_width}}",
            f"{_fmt(float(summary['mean_score']), float(summary['stderr_score'])):>{col_w}}",
            f"{_fmt(float(summary['valid_judge_rate']), float(summary['stderr_valid_judge_rate'])):>{col_w}}",
            f"{_fmt(float(summary['invalid_judge_rate']), float(summary['stderr_invalid_judge_rate'])):>{col_w}}",
            f"{_fmt(float(summary['extraction_failure_rate']), float(summary['stderr_extraction_failure_rate'])):>{col_w}}",
            f"{_fmt(float(summary['budget_forced_rate']), float(summary['stderr_budget_forced_rate'])):>{col_w}}",
            f"{_fmt(float(summary['maxim_accuracy']), float(summary['stderr_maxim_accuracy'])):>{col_w}}",
        ]
        for task_name in task_cols:
            task_summary = summary["per_task_mean_score"][task_name]
            row.append(f"{_fmt(float(task_summary['mean']), float(task_summary['stderr'])):>{col_w}}")
        print(" | ".join(row))
    print(sep)
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = get_eval_results_dir(ROOT, "altprag")
    results_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = results_dir / f"altprag_eval_outputs_{timestamp}"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for short, payload in all_model_payloads.items():
        model_path = payload["model_path"]
        for seed, outputs in zip(SEEDS, payload["per_seed_outputs"], strict=False):
            output_path = outputs_dir / f"{_safe_name(short)}_seed{int(seed)}.jsonl"
            _write_seed_outputs(
                output_path=output_path,
                model_path=model_path,
                seed=int(seed),
                examples=examples,
                results=outputs,
            )

    json_path = results_dir / f"altprag_eval_results_{timestamp}.json"
    serializable_results = {
        short: payload["summary"] | {"per_seed": payload["per_seed_metrics"]}
        for short, payload in all_model_payloads.items()
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": DATASET_NAME,
                "hf_split": HF_SPLIT,
                "task_mode": TASK_MODE,
                "num_examples": len(examples),
                "data_stats": {
                    "total_rows": int(data_stats.total_rows),
                    "selected_examples": int(data_stats.selected_examples),
                    "task_counts": dict(data_stats.task_counts),
                    "maxim_counts": dict(data_stats.maxim_counts),
                },
                "config": {
                    "eval_backend": EVAL_BACKEND,
                    "enable_thinking": bool(ENABLE_THINKING),
                    "thinking_budget_tokens": int(THINKING_BUDGET_TOKENS),
                    "answer_max_new_tokens": int(ANSWER_MAX_NEW_TOKENS),
                    "do_sample": bool(DO_SAMPLE),
                    "temperature": float(TEMPERATURE),
                    "top_p": float(TOP_P),
                    "top_k": int(TOP_K),
                    "min_p": float(MIN_P),
                    "isolate_each_run": bool(ISOLATE_EACH_RUN),
                    "judge_model": JUDGE_MODEL,
                    "judge_temperature": float(JUDGE_TEMPERATURE),
                },
                "results": serializable_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = results_dir / f"altprag_eval_summary_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "mean_score",
                "stderr_score",
                "valid_judge_rate",
                "invalid_judge_rate",
                "extraction_failure_rate",
                "budget_forced_rate",
                "maxim_accuracy",
                "maxim_extraction_rate",
                "candidate1_mean_score",
                "candidate2_mean_score",
            ]
        )
        for short, payload in all_model_payloads.items():
            summary = payload["summary"]
            writer.writerow(
                [
                    short,
                    f"{float(summary['mean_score']):.6f}",
                    f"{float(summary['stderr_score']):.6f}",
                    f"{float(summary['valid_judge_rate']):.6f}",
                    f"{float(summary['invalid_judge_rate']):.6f}",
                    f"{float(summary['extraction_failure_rate']):.6f}",
                    f"{float(summary['budget_forced_rate']):.6f}",
                    f"{float(summary['maxim_accuracy']):.6f}",
                    f"{float(summary['maxim_extraction_rate']):.6f}",
                    f"{float(summary['per_task_mean_score']['candidate1']['mean']):.6f}",
                    f"{float(summary['per_task_mean_score']['candidate2']['mean']):.6f}",
                ]
            )

    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved CSV summary to {csv_path}")
    print(f"Saved per-example outputs to {outputs_dir}")


if __name__ == "__main__":
    main()
