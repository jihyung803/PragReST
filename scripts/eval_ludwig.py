#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from urllib.request import urlopen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.eval_common import get_eval_results_dir  # noqa: E402

from src.eval.vllm_boxed_mcqa import (  # noqa: E402
    DEFAULT_EARLY_STOPPING_TEXT,
    expand_model_specs,
    resolve_model_spec,
    run_vllm_boxed_mcqa,
)
from src.eval.sglang_boxed_mcqa import run_sglang_boxed_mcqa  # noqa: E402
from src.eval.transformers_boxed_mcqa import run_transformers_boxed_mcqa  # noqa: E402

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

    ludwig_model_spec = os.environ.get("LUDWIG_MODEL_SPEC", "").strip()
    if ludwig_model_spec:
        if ludwig_model_spec.startswith(("{", "[")):
            parsed = json.loads(ludwig_model_spec)
            return parsed if isinstance(parsed, list) else [parsed]
        return [ludwig_model_spec]

    return MODELS




SEEDS = [
    int(x.strip())
    for x in os.environ.get("LUDWIG_SEEDS", "1").split(",")
    if x.strip()
]
EVAL_BACKEND = os.environ.get("EVAL_BACKEND", "vllm").strip().lower()

DATA_URL_BASE = "https://raw.githubusercontent.com/ucl-dark/ludwig/main/"
DEV_CSV_URL = DATA_URL_BASE + "dev_conversational_implicatures.csv"
TEST_CSV_URL = DATA_URL_BASE + "test_conversational_implicatures.csv"
DEV_CSV_PATH = ""
TEST_CSV_PATH = ""
TEMPLATE_NAMES = [
    "template_1",
]
EVAL_SPLIT = "test"
K_SHOT = 0
MAX_EXAMPLES = 0
SHARD_INDEX = int(os.environ.get("LUDWIG_SHARD_INDEX", "0"))
SHARD_TOTAL = int(os.environ.get("LUDWIG_SHARD_TOTAL", "1"))
MAX_PROMPT_LENGTH = 4096
THINKING_BUDGET_TOKENS = 2048
ANSWER_MAX_NEW_TOKENS = int(os.environ.get("LUDWIG_ANSWER_MAX_NEW_TOKENS", "456"))
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1
TOP_K = 0
MIN_P = 0.0
GPU_MEMORY_UTILIZATION = 0.9
TENSOR_PARALLEL_SIZE = 1
FEWSHOT_SEED = 0
ISOLATE_EACH_RUN = False

GENERATION_INSTRUCTION = (
    "Use exactly one word inside the box: yes or no.\n"
    "Final answer of the question must be enclosed in \\boxed{...}.\n"
)
BOXED_START_RE = re.compile(r"\\boxed\s*\{", re.IGNORECASE)
YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

TEMPLATE_TEXT = {
    "template_1": "Does the following response to the question imply yes or no?\n\nquestion: {utterance}\nresponse: {response}\nimplicature:",
    "template_2": "Finish the following text:\n\nEsther asked \"{utterance}\" and Juan responded \"{response}\", which means",
    "template_3": "Is the implied meaning of the following response yes or no:\n\nquestion: {utterance}\nresponse: {response}\nmeaning:",
    "template_4": "What is the intent of the following response, yes or no?\n\nquestion: {utterance}\nresponse: {response}\nintent:",
    "template_5": "Finish the following text:\n\nKaren asked \"{utterance}\" and William responded \"{response}\", which means",
    "template_6": "Finish the following text:\n\nBob asked \"{utterance}\" and Alice responded \"{response}\", which means",
}


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


def _fmt(mean: float, stderr: float) -> str:
    return f"{mean:.2f}\u00b1{stderr:.2f}"


def _normalize_implicature(value: str) -> str | None:
    lower = str(value).strip().lower()
    if lower.startswith("yes"):
        return "yes"
    if lower.startswith("no"):
        return "no"
    return None


def _extract_boxed_text(text: str) -> str:
    raw = str(text or "")
    for match in BOXED_START_RE.finditer(raw):
        brace_open_idx = match.end() - 1
        depth = 0
        for i in range(brace_open_idx, len(raw)):
            ch = raw[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[brace_open_idx + 1 : i].strip()
    return ""


def _parse_binary_answer(text: str) -> str | None:
    candidates = []
    boxed = _extract_boxed_text(text)
    if boxed:
        candidates.append(boxed)
    candidates.append(str(text or ""))
    for candidate in candidates:
        match = YES_NO_RE.search(candidate)
        if match:
            return match.group(1).lower()
    return None


def _read_csv_rows(path_or_url: str) -> list[dict[str, str]]:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urlopen(path_or_url) as resp:
            content = resp.read().decode("utf-8")
        return list(csv.DictReader(io.StringIO(content)))
    with open(path_or_url, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_split_rows() -> tuple[list[dict], list[dict]]:
    dev_source = DEV_CSV_PATH if DEV_CSV_PATH else DEV_CSV_URL
    test_source = TEST_CSV_PATH if TEST_CSV_PATH else TEST_CSV_URL
    return _read_csv_rows(dev_source), _read_csv_rows(test_source)


def _process_row(row: dict[str, str]) -> dict | None:
    implicature = _normalize_implicature(row.get("Implicature", ""))
    if implicature is None:
        return None
    return {
        "utterance": row["Context utterance"].strip("\n"),
        "response": row["Response utterance"].strip("\n"),
        "implicature": implicature,
    }


def _build_examples(rows: list[dict[str, str]]) -> list[dict]:
    examples = []
    for row_idx, row in enumerate(rows):
        ex = _process_row(row)
        if ex is not None:
            ex["_row_idx"] = row_idx
            examples.append(ex)
    return examples


def _apply_template(example: dict, template_name: str) -> tuple[str, str]:
    try:
        template = TEMPLATE_TEXT[template_name]
    except KeyError as exc:
        raise ValueError(f"Unknown template_name={template_name}") from exc
    prompt = template.format(utterance=example["utterance"], response=example["response"])
    target = example["implicature"]
    return prompt, target


def _add_prompts(split_examples: list[dict], dev_examples: list[dict], k_shot: int) -> list[dict]:
    rng = random.Random(FEWSHOT_SEED)
    with_prompts = []
    for idx, example in enumerate(split_examples):
        prompt_pool = dev_examples
        if split_examples is dev_examples:
            prompt_pool = [candidate for candidate in dev_examples if candidate is not example]
        sample_size = min(int(k_shot), len(prompt_pool))
        prompt_examples = rng.sample(prompt_pool, sample_size) if sample_size > 0 else []
        with_prompts.append({**example, "prompts": prompt_examples, "id": str(idx + 1)})
    return with_prompts


def _build_prompt_text(example: dict, template_name: str) -> str:
    fewshot_chunks = []
    for shot in example["prompts"]:
        shot_prompt, shot_target = _apply_template(shot, template_name)
        fewshot_chunks.append(f"{shot_prompt} {shot_target}")
    prompt, _ = _apply_template(example, template_name)
    ctx = "\n\n".join(fewshot_chunks + [prompt]) if fewshot_chunks else prompt
    return ctx + GENERATION_INSTRUCTION


def run_inference(model_path: str, examples: list[dict], *, seed: int, mode: str, template_name: str):
    prompt_texts = [_build_prompt_text(ex, template_name) for ex in examples]
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
    results, meta = runner(
        model_path=model_path,
        prompt_texts=prompt_texts,
        example_keys=[str(ex["id"]) for ex in examples],
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
        early_stopping_text=DEFAULT_EARLY_STOPPING_TEXT,
        mode_label=f"{mode}:seed{seed}",
        use_tqdm=True,
        answer_parser=_parse_binary_answer,
    )

    total = len(examples)
    correct = 0
    extraction_failures = 0
    for ex in examples:
        pred = results[str(ex["id"])].get("predicted_answer")
        if pred is None:
            extraction_failures += 1
        elif str(pred).lower() == str(ex["implicature"]).lower():
            correct += 1

    metrics = {
        "accuracy": (correct / total * 100.0) if total else 0.0,
        "extraction_failure_rate": (extraction_failures / total * 100.0) if total else 0.0,
        "budget_forced_rate": (float(meta["budget_forced_count"]) / total * 100.0) if total else 0.0,
        "num_examples": int(total),
        "num_correct": int(correct),
        "num_extraction_failures": int(extraction_failures),
        "num_budget_forced": int(meta["budget_forced_count"]),
        "elapsed_seconds": float(meta["elapsed_seconds"]),
    }
    print(
        f"[{mode}] seed={seed} acc={correct}/{total} = {metrics['accuracy']:.2f}% "
        f"fail={metrics['extraction_failure_rate']:.2f}% forced={metrics['budget_forced_rate']:.2f}% "
        f"time={metrics['elapsed_seconds']:.1f}s"
    )
    return results, metrics


def _load_eval_examples() -> tuple[list[dict], str]:
    if int(SHARD_TOTAL) < 1:
        raise ValueError(f"LUDWIG_SHARD_TOTAL must be >= 1, got {SHARD_TOTAL}")
    if int(SHARD_INDEX) < 0 or int(SHARD_INDEX) >= int(SHARD_TOTAL):
        raise ValueError(
            f"LUDWIG_SHARD_INDEX must be in [0, {int(SHARD_TOTAL) - 1}], got {SHARD_INDEX}"
        )

    dev_rows, test_rows = _load_split_rows()
    dev_examples = _build_examples(dev_rows)
    test_examples = _build_examples(test_rows)
    split_name = str(EVAL_SPLIT).strip().lower()
    if split_name == "dev":
        split_examples = dev_examples
    elif split_name == "test":
        split_examples = test_examples
    else:
        raise ValueError(f"Unsupported EVAL_SPLIT={EVAL_SPLIT!r}; expected 'dev' or 'test'")

    examples = _add_prompts(split_examples, dev_examples, int(K_SHOT))
    if MAX_EXAMPLES:
        examples = examples[: int(MAX_EXAMPLES)]
    if int(SHARD_TOTAL) > 1:
        total = len(examples)
        start = total * int(SHARD_INDEX) // int(SHARD_TOTAL)
        end = total * (int(SHARD_INDEX) + 1) // int(SHARD_TOTAL)
        examples = examples[start:end]
    if not examples:
        raise RuntimeError("No valid LUDWIG examples loaded")
    return examples, split_name


def _run_inference_subprocess(model_path: str | dict, *, seed: int, mode: str, template_name: str):
    payload = {
        "model_path": model_path,
        "seed": int(seed),
        "mode": str(mode),
        "template_name": str(template_name),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as in_f:
        json.dump(payload, in_f)
        input_path = in_f.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as out_f:
        output_path = out_f.name

    env = os.environ.copy()
    env["LUDWIG_EVAL_WORKER"] = "1"
    env["LUDWIG_EVAL_INPUT"] = input_path
    env["LUDWIG_EVAL_OUTPUT"] = output_path

    try:
        subprocess.run([sys.executable, os.path.abspath(__file__)], check=True, env=env)
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        return result["results"], result["metrics"]
    finally:
        for path in (input_path, output_path):
            try:
                os.remove(path)
            except OSError:
                pass


def _maybe_run_worker() -> bool:
    if os.environ.get("LUDWIG_EVAL_WORKER") != "1":
        return False

    input_path = os.environ["LUDWIG_EVAL_INPUT"]
    output_path = os.environ["LUDWIG_EVAL_OUTPUT"]
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    examples, _ = _load_eval_examples()
    results, metrics = run_inference(
        payload["model_path"],
        examples,
        seed=int(payload["seed"]),
        mode=str(payload["mode"]),
        template_name=str(payload["template_name"]),
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "metrics": metrics}, f)
    return True


def main() -> None:
    if _maybe_run_worker():
        return

    examples, split_name = _load_eval_examples()
    configured_specs = _configured_models()
    model_specs = expand_model_specs(configured_specs)

    template_names = [str(name) for name in TEMPLATE_NAMES]
    for template_name in template_names:
        if template_name not in TEMPLATE_TEXT:
            raise ValueError(f"Unknown template in TEMPLATE_NAMES: {template_name}")

    all_results: dict[str, dict] = {}
    for template_name in template_names:
        template_results = {}
        for model_path in model_specs:
            short = _short_name(model_path)
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {short} | template={template_name}")
            print(f"{'=' * 60}")

            per_seed_metrics = []
            for seed in SEEDS:
                if ISOLATE_EACH_RUN:
                    _, metrics = _run_inference_subprocess(
                        model_path,
                        seed=seed,
                        mode=short,
                        template_name=template_name,
                    )
                else:
                    _, metrics = run_inference(
                        model_path,
                        examples,
                        seed=seed,
                        mode=short,
                        template_name=template_name,
                    )
                per_seed_metrics.append(metrics)
            template_results[_model_key(model_path)] = per_seed_metrics
        all_results[template_name] = template_results

    name_width = max(max(len(_short_name(c)) for c in model_specs), len("Model"))
    col_w = 14
    header = " | ".join([
        f"{'Model':<{name_width}}",
        f"{'Acc':>{col_w}}",
        f"{'Fail':>{col_w}}",
        f"{'Forced':>{col_w}}",
    ])
    sep = "-" * len(header)

    for template_name in template_names:
        print(f"\n\n{'=' * len(header)}")
        print("LUDWIG EVALUATION RESULTS")
        print(
            f"Seeds: {SEEDS}  |  split={split_name}  |  n={len(examples)} examples  |  "
            f"shard={SHARD_INDEX}/{SHARD_TOTAL}"
        )
        print(
            f"backend={EVAL_BACKEND}, template={template_name}, k_shot={K_SHOT}, do_sample={DO_SAMPLE}, temperature={TEMPERATURE}, "
            f"top_k={TOP_K}, min_p={MIN_P}, top_p={TOP_P}"
        )
        print(f"{'=' * len(header)}\n")
        print(header)
        print(sep)

        for model_path in model_specs:
            seed_metrics = all_results[template_name][_model_key(model_path)]
            acc_mean, acc_se = _compute_mean_stderr([m["accuracy"] for m in seed_metrics])
            fail_mean, fail_se = _compute_mean_stderr([m["extraction_failure_rate"] for m in seed_metrics])
            forced_mean, forced_se = _compute_mean_stderr([m["budget_forced_rate"] for m in seed_metrics])
            print(" | ".join([
                f"{_short_name(model_path):<{name_width}}",
                f"{_fmt(acc_mean, acc_se):>{col_w}}",
                f"{_fmt(fail_mean, fail_se):>{col_w}}",
                f"{_fmt(forced_mean, forced_se):>{col_w}}",
            ]))

        print(sep)
        print()

    results_dir = str(get_eval_results_dir(ROOT, "ludwig"))
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_out = {
        "task": "ludwig",
        "config": {
            "template_names": template_names,
            "eval_split": split_name,
            "k_shot": K_SHOT,
            "seeds": SEEDS,
            "num_examples": len(examples),
            "shard_index": int(SHARD_INDEX),
            "shard_total": int(SHARD_TOTAL),
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
            "dev_csv_path": DEV_CSV_PATH,
            "test_csv_path": TEST_CSV_PATH,
            "dev_csv_url": DEV_CSV_URL,
            "test_csv_url": TEST_CSV_URL,
        },
        "results": {},
    }
    for template_name in template_names:
        json_out["results"][template_name] = {}
        for model_path in model_specs:
            seed_metrics = all_results[template_name][_model_key(model_path)]
            acc_mean, acc_se = _compute_mean_stderr([m["accuracy"] for m in seed_metrics])
            fail_mean, fail_se = _compute_mean_stderr([m["extraction_failure_rate"] for m in seed_metrics])
            forced_mean, forced_se = _compute_mean_stderr([m["budget_forced_rate"] for m in seed_metrics])
            json_out["results"][template_name][_model_key(model_path)] = {
                "accuracy": {
                    "mean": round(acc_mean, 2),
                    "stderr": round(acc_se, 2),
                    "per_seed": [round(m["accuracy"], 2) for m in seed_metrics],
                },
                "extraction_failure_rate": {
                    "mean": round(fail_mean, 2),
                    "stderr": round(fail_se, 2),
                    "per_seed": [round(m["extraction_failure_rate"], 2) for m in seed_metrics],
                },
                "budget_forced_rate": {
                    "mean": round(forced_mean, 2),
                    "stderr": round(forced_se, 2),
                    "per_seed": [round(m["budget_forced_rate"], 2) for m in seed_metrics],
                },
            }

    json_path = os.path.join(results_dir, f"ludwig_eval_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON results saved to: {json_path}")

    csv_path = os.path.join(results_dir, f"ludwig_eval_summary_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "template",
            "model",
            "acc_mean",
            "acc_stderr",
            "fail_mean",
            "fail_stderr",
            "forced_mean",
            "forced_stderr",
            "shard_index",
            "shard_total",
            "num_examples",
            "num_correct",
            "num_extraction_failures",
            "num_budget_forced",
        ])
        for template_name in template_names:
            for model_path in model_specs:
                seed_metrics = all_results[template_name][_model_key(model_path)]
                acc_mean, acc_se = _compute_mean_stderr([m["accuracy"] for m in seed_metrics])
                fail_mean, fail_se = _compute_mean_stderr([m["extraction_failure_rate"] for m in seed_metrics])
                forced_mean, forced_se = _compute_mean_stderr([m["budget_forced_rate"] for m in seed_metrics])
                num_examples = sum(int(m.get("num_examples", 0)) for m in seed_metrics)
                num_correct = sum(int(m.get("num_correct", 0)) for m in seed_metrics)
                num_fail = sum(int(m.get("num_extraction_failures", 0)) for m in seed_metrics)
                num_forced = sum(int(m.get("num_budget_forced", 0)) for m in seed_metrics)
                writer.writerow([
                    template_name,
                    _short_name(model_path),
                    round(acc_mean, 2),
                    round(acc_se, 2),
                    round(fail_mean, 2),
                    round(fail_se, 2),
                    round(forced_mean, 2),
                    round(forced_se, 2),
                    int(SHARD_INDEX),
                    int(SHARD_TOTAL),
                    num_examples,
                    num_correct,
                    num_fail,
                    num_forced,
                ])
    print(f"CSV summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
