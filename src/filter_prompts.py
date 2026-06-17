"""
Filter out QA prompts that are too easy for GRPO training, where zero-variance
groups yield no learning signal. Optionally also filter prompts that are too
hard because no rollout succeeds.

Steps:
  1. Load train.parquet (VERL schema: prompt, reward_model, extra_info, data_source)
  2. Generate N rollouts per prompt using the SFT model via vLLM
  3. Judge each rollout with the same single-token QA correctness judge used by
     reward_fn.py.
  4. A rollout is correct iff P(yes) - P(no) > --judge_margin_threshold. Keep
     the prompt unless every rollout is correct. With --filter_hard, also drop
     prompts where no rollout is correct.
  5. Save filtered parquet.

Requires:
  - A single vLLM server for both generation and judging

Usage:
  # Start vLLM server:
  python -m vllm.entrypoints.openai.api_server \
      --model your-org/Qwen3-8B-PK-SFT --port 8200 \
      --served-model-name policy --dtype bfloat16 --trust-remote-code

  # Run filtering:
  python src/filter_prompts.py \
      --input data/verl-mixed-qwen-8b/train.parquet \
      --output data/verl-mixed-qwen-8b/train_filtered.parquet \
      --base_url http://localhost:8200 \
      --model_name policy \
      --n_samples 8 \
      --rollout_chunk_size 4 \
      --judge_chunk_workers 4 \
      --judge_margin_threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

# ── thread-safe logging ────────────────────────────────────────────────────────

_print_lock = threading.Lock()
_judge_log_lock = threading.Lock()
_judge_log_path = None  # set in main() based on --output


# ── rollout parsing (policy output) ──────────────────────────────────────────

def strip_thinking(raw: str) -> str:
    idx = raw.rfind("</think>")
    return raw[idx + len("</think>"):].strip() if idx != -1 else raw


def extract_final_response(raw: str) -> str | None:
    after = strip_thinking(raw)
    start = after.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    while i < len(after):
        c = after[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = after[start + len("\\boxed{"):i].strip()
                return candidate or None
        i += 1
    return None


def check_response_format(text: str) -> bool:
    return extract_final_response(text) is not None


# ── judge output parsing ─────────────────────────────────────────────────────

def extract_yes_no_probs(response_json: dict) -> tuple[float, float]:
    """Return (P(yes), P(no)) from the first-token top_logprobs."""
    try:
        choice = response_json["choices"][0]
        logprobs_field = choice.get("logprobs") or {}
        content = logprobs_field.get("content") or []
        if not content:
            return 0.0, 0.0
        top = content[0].get("top_logprobs") or []
    except (KeyError, IndexError, TypeError):
        return 0.0, 0.0

    p_yes = 0.0
    p_no = 0.0
    for entry in top:
        token = (entry.get("token") or "").strip().lower()
        logprob = entry.get("logprob")
        if logprob is None:
            continue
        if token == "yes":
            p_yes += math.exp(logprob)
        elif token == "no":
            p_no += math.exp(logprob)
    return p_yes, p_no


def is_correct_by_margin(
    p_yes: float,
    p_no: float,
    threshold: float,
) -> bool:
    return (p_yes - p_no) > threshold


def is_filterable_easy(n_confident_yes: int, n_total: int) -> bool:
    return n_total > 0 and n_confident_yes == n_total


def is_filterable_hard(n_confident_yes: int, n_total: int) -> bool:
    return n_total > 0 and n_confident_yes == 0


# ── judge prompt builders ────────────────────────────────────────────────────

def get_qa_judge_fields(extra_info: dict, ground_truth: str) -> tuple[str, str, str]:
    row = extra_info.get("source_row", {})
    if not isinstance(row, dict):
        row = {}

    content = str(row.get("content", extra_info.get("content", "")) or "").strip()
    question = str(row.get("question", extra_info.get("question", "")) or "").strip()
    reference = str(
        extra_info.get(
            "reference_answer",
            row.get("answer", extra_info.get("reference", ground_truth)),
        )
        or ""
    ).strip()

    return content, question, reference


def build_qa_judge_prompt(
    content: str,
    question: str,
    reference: str,
    candidate_output: str,
) -> list[dict]:
    user_content = (
        "Task: QA answer grading\n\n"
        f"Context:\n{content}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Candidate answer:\n{candidate_output}\n\n"
        "Is the candidate answer semantically correct given the context, "
        "question, and reference answer? "
        "Answer with a single word: yes or no."
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a strict QA evaluator. "
                "Respond with exactly one word: either 'yes' or 'no'. "
                "Do not emit any other text, punctuation, or explanation."
            ),
        },
        {"role": "user", "content": user_content},
    ]


# ── API callers ──────────────────────────────────────────────────────────────

def _make_session():
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=256, pool_maxsize=256)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def generate_responses(session, base_url, model_name, messages, n, temperature, top_p, top_k, min_p, max_tokens, seed=None):
    body = {
        "model": model_name,
        "messages": messages,
        "n": n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if seed is not None:
        # vLLM honours `seed` to fix the sampler RNG (covers stochastic
        # decoding and any tie-break draws under greedy).
        body["seed"] = int(seed)
    resp = session.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        timeout=300,
    )
    resp.raise_for_status()
    return [c["message"]["content"] for c in resp.json()["choices"]]


def call_judge(session, judge_base_url, judge_model_name, messages, seed=None):
    """Return the judge response JSON with first-token yes/no logprobs."""
    body = {
        "model": judge_model_name,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if seed is not None:
        body["seed"] = int(seed)
    resp = session.post(
        f"{judge_base_url}/v1/chat/completions",
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ── per-row evaluation ───────────────────────────────────────────────────────

def resolve_task_type(data_source, extra_info):
    task_type = str(extra_info.get("task_type", "")).strip().lower()
    if task_type == "qa":
        return task_type
    if data_source in {"pragrest_qa", "bk_link_qa"}:
        return "qa"
    return "qa"


def _log_judge_call(row_idx, rollout_idx, task_type, messages, candidate, reference,
                    judge_token, judge_result, margin_threshold, confident):
    if _judge_log_path is None:
        return
    record = {
        "row_idx": row_idx,
        "rollout_idx": rollout_idx,
        "task_type": task_type,
        "candidate": candidate,
        "reference": reference,
        "judge_prompt": messages[1]["content"],
        "judge_token": judge_token,
        "p_yes": judge_result.get("p_yes"),
        "p_no": judge_result.get("p_no"),
        "margin": judge_result.get("margin"),
        "margin_threshold": margin_threshold,
        "confident_yes": confident,
    }
    with _judge_log_lock:
        with open(_judge_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def judge_one_response(session, judge_base_url, judge_model_name, task_type, extra_info,
                       ground_truth, response_text, margin_threshold,
                       row_idx=-1, rollout_idx=-1, verbose=False, seed=None):
    """Return margin-judge result for one QA rollout."""
    candidate = extract_final_response(response_text)
    if candidate is None:
        return {
            "judge_token": None,
            "p_yes": 0.0,
            "p_no": 0.0,
            "margin": 0.0,
            "confident_yes": False,
            "format_ok": False,
        }

    content, question, reference = get_qa_judge_fields(extra_info, ground_truth)
    messages = build_qa_judge_prompt(content, question, reference, candidate)

    judge_result = {
        "judge_token": None,
        "p_yes": 0.0,
        "p_no": 0.0,
        "margin": 0.0,
    }
    try:
        response_json = call_judge(session, judge_base_url, judge_model_name, messages, seed=seed)
        p_yes, p_no = extract_yes_no_probs(response_json)
        try:
            judge_token = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            judge_token = None
        judge_result = {
            "judge_token": judge_token,
            "p_yes": p_yes,
            "p_no": p_no,
            "margin": p_yes - p_no,
        }
    except Exception as e:
        with _print_lock:
            print(f"  [Judge error] row {row_idx} rollout {rollout_idx}: {e}", flush=True)

    confident = is_correct_by_margin(
        judge_result["p_yes"], judge_result["p_no"], margin_threshold,
    )

    if verbose:
        with _print_lock:
            print(f"    [Judge] row={row_idx} rollout={rollout_idx} task={task_type}\n"
                  f"      context: {content!r}\n"
                  f"      question: {question!r}\n"
                  f"      candidate: {candidate!r}\n"
                  f"      reference: {reference!r}\n"
                  f"      judge_token={judge_result['judge_token']!r} "
                  f"p_yes={judge_result['p_yes']:.6f} "
                  f"p_no={judge_result['p_no']:.6f} "
                  f"margin={judge_result['margin']:+.6f} "
                  f"threshold={margin_threshold} confident_yes={confident}",
                  flush=True)

    _log_judge_call(row_idx, rollout_idx, task_type, messages, candidate, reference,
                    judge_result["judge_token"], judge_result, margin_threshold, confident)

    return {
        **judge_result,
        "confident_yes": confident,
        "format_ok": True,
    }


def judge_response_chunk(session, judge_base_url, judge_model_name, task_type, extra_info,
                         ground_truth, responses, margin_threshold,
                         row_idx=-1, rollout_offset=0, verbose=False, max_workers=1):
    if max_workers <= 1 or len(responses) <= 1:
        return [
            (
                rollout_offset + i,
                judge_one_response(
                    session, judge_base_url, judge_model_name,
                    task_type, extra_info, ground_truth, response_text,
                    margin_threshold,
                    row_idx=row_idx, rollout_idx=rollout_offset + i, verbose=verbose,
                ),
            )
            for i, response_text in enumerate(responses)
        ]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(responses))) as pool:
        futures = {
            pool.submit(
                judge_one_response,
                session, judge_base_url, judge_model_name,
                task_type, extra_info, ground_truth, response_text,
                margin_threshold,
                row_idx, rollout_offset + i, verbose,
            ): rollout_offset + i
            for i, response_text in enumerate(responses)
        }
        return [(futures[fut], fut.result()) for fut in as_completed(futures)]


def to_native(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj


def evaluate_row(row_idx, row, session, base_url, model_name,
                 judge_base_url, judge_model_name, args):
    """Generate n samples for one row.

    For easy-only filtering, stop once any rollout is not correct. When hard
    filtering is enabled, continue until the row is known to be mixed or all
    sampled rollouts have been judged.
    """
    prompt = row["prompt"]
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    prompt = to_native(prompt)

    extra_info = row.get("extra_info", {})
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    extra_info = to_native(extra_info)

    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)
    reward_model = to_native(reward_model)
    ground_truth = reward_model.get("ground_truth", "")

    data_source = row.get("data_source", "")
    task_type = resolve_task_type(data_source, extra_info)

    n_confident_yes = 0
    n_judged = 0
    n_generated = 0
    while n_generated < args.n_samples:
        chunk_n = min(args.rollout_chunk_size, args.n_samples - n_generated)
        try:
            responses = generate_responses(
                session, base_url, model_name, prompt,
                n=chunk_n,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            print(f"  [Gen error] row {row_idx}: {e}", flush=True)
            return row_idx, -1, args.n_samples  # keep row on gen failure
        if not responses:
            print(f"  [Gen error] row {row_idx}: empty response chunk", flush=True)
            return row_idx, -1, args.n_samples

        judged = judge_response_chunk(
            session, judge_base_url, judge_model_name,
            task_type, extra_info, ground_truth, responses,
            args.judge_margin_threshold,
            row_idx=row_idx,
            rollout_offset=n_generated,
            verbose=args.verbose,
            max_workers=args.judge_chunk_workers,
        )
        n_generated += len(responses)

        chunk_correct = 0
        for _, judge_result in judged:
            n_judged += 1
            if judge_result["confident_yes"]:
                n_confident_yes += 1
                chunk_correct += 1

        has_success = n_confident_yes > 0
        has_failure = n_judged > n_confident_yes
        if args.filter_hard:
            if has_success and has_failure:
                if args.verbose:
                    with _print_lock:
                        print(f"  [Early stop] row {row_idx}: mixed outcomes seen after "
                              f"{n_judged}/{args.n_samples} judged "
                              f"({n_confident_yes}/{n_judged} correct)",
                              flush=True)
                return row_idx, n_confident_yes, n_judged
        elif chunk_correct < len(judged):
            if args.verbose:
                with _print_lock:
                    print(f"  [Early stop] row {row_idx}: not all rollouts correct after "
                          f"{n_judged}/{args.n_samples} judged "
                          f"({n_confident_yes}/{n_judged} correct)",
                          flush=True)
            return row_idx, n_confident_yes, n_judged
    return row_idx, n_confident_yes, n_judged


def main():
    parser = argparse.ArgumentParser(description="Filter QA prompts for GRPO")
    parser.add_argument("--input", required=True, help="Path to train.parquet")
    parser.add_argument("--output", required=True, help="Path to filtered output parquet")
    parser.add_argument("--n_samples", type=int, default=8, help="Rollouts per prompt")
    parser.add_argument("--rollout_chunk_size", type=int, default=4,
                        help="Generate and judge this many rollouts at a time, allowing early stop before all rollouts are generated.")
    parser.add_argument("--judge_chunk_workers", type=int, default=1,
                        help="Parallel judge calls per rollout chunk. Total judge concurrency is roughly max_workers * judge_chunk_workers.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--base_url", default="http://localhost:8200",
                        help="vLLM endpoint serving the rollout/policy model.")
    parser.add_argument("--model_name", default="policy",
                        help="served-model-name for the rollout/policy model.")
    parser.add_argument("--judge_base_url", default=None,
                        help="vLLM endpoint serving the judge model. "
                             "Defaults to --base_url if omitted.")
    parser.add_argument("--judge_model_name", default=None,
                        help="served-model-name for the judge model. "
                             "Defaults to --model_name if omitted.")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Parallel rows to evaluate at once")
    parser.add_argument("--judge_margin_threshold", type=float, default=0.8,
                        help="A rollout is correct when P(yes) - P(no) is greater than this value.")
    parser.add_argument("--confidence_threshold", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--filter_hard", action="store_true",
                        help="Also filter prompts with 0 correct rollouts.")
    parser.add_argument("--margin_threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true",
                        help="Print every rollout/judge decision. Disabled by default because it can bottleneck filtering.")
    args = parser.parse_args()

    if not -1.0 <= args.judge_margin_threshold <= 1.0:
        parser.error("--judge_margin_threshold must be between -1.0 and 1.0")
    if args.n_samples < 1:
        parser.error("--n_samples must be at least 1")
    if args.rollout_chunk_size < 1:
        parser.error("--rollout_chunk_size must be at least 1")
    if args.judge_chunk_workers < 1:
        parser.error("--judge_chunk_workers must be at least 1")

    global _judge_log_path
    _judge_log_path = args.output.replace(".parquet", "_judge_log.jsonl")

    judge_base_url = args.judge_base_url or args.base_url
    judge_model_name = args.judge_model_name or args.model_name

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Judge log will be written to {_judge_log_path}")
    print(f"Policy endpoint: {args.base_url}  (model={args.model_name})")
    print(f"Judge  endpoint: {judge_base_url}  (model={judge_model_name})")
    if args.margin_threshold is not None:
        print("Ignoring deprecated --margin_threshold; use --judge_margin_threshold.")
    if args.confidence_threshold is not None:
        print("Ignoring deprecated --confidence_threshold; margin filtering is used now.")
    print(f"Rollout chunk size: {args.rollout_chunk_size}")
    print(f"Judge chunk workers: {args.judge_chunk_workers}")
    print(f"Judge margin threshold: {args.judge_margin_threshold}")
    print(f"Filter hard prompts: {args.filter_hard}")

    session = _make_session()

    results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        for idx, row in df.iterrows():
            fut = pool.submit(
                evaluate_row, idx, row,
                session, args.base_url, args.model_name,
                judge_base_url, judge_model_name,
                args,
            )
            futures[fut] = idx

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            row_idx, n_confident_yes, n_total = fut.result()
            rate = n_confident_yes / n_total if n_total > 0 else -1
            results[row_idx] = (n_confident_yes, n_total, rate)
            ds = df.loc[row_idx, "data_source"] if row_idx in df.index else "?"
            if args.verbose:
                tqdm.write(
                    f"  row {row_idx:>4d} [{ds}] "
                    f"{n_confident_yes}/{n_total} correct = {rate:.0%}"
                )

    # ── stats ────────────────────────────────────────────────────────────────
    n_filterable_easy = sum(
        1 for ncy, nt, _ in results.values()
        if is_filterable_easy(ncy, nt)
    )
    n_filterable_hard = sum(
        1 for ncy, nt, _ in results.values()
        if is_filterable_hard(ncy, nt)
    )
    n_error = sum(1 for ncy, _, _ in results.values() if ncy == -1)
    n_filtered_hard = n_filterable_hard if args.filter_hard else 0
    n_kept = len(results) - n_filterable_easy - n_filtered_hard

    print(f"\n{'='*60}")
    print(f"Total prompts:  {len(df)}")
    print(f"  Easy (all rollouts P(yes)-P(no) > {args.judge_margin_threshold}): "
          f"{n_filterable_easy}  <- filtered out")
    hard_action = "filtered out" if args.filter_hard else "kept"
    print(f"  Hard (0 correct): {n_filterable_hard}  <- {hard_action}")
    print(f"  Other kept:   {n_kept - (0 if args.filter_hard else n_filterable_hard) - n_error}")
    print(f"  Gen errors:   {n_error}  <- kept (benefit of doubt)")
    print(f"{'='*60}")

    # ── filter ───────────────────────────────────────────────────────────────
    keep_indices = [
        idx for idx, (ncy, nt, _) in results.items()
        if not is_filterable_easy(ncy, nt)
        and not (args.filter_hard and is_filterable_hard(ncy, nt))
    ]
    df_filtered = df.loc[keep_indices].reset_index(drop=True)
    df_filtered.to_parquet(args.output)
    print(f"Saved {len(df_filtered)} rows to {args.output}")

    # ── optional: save full stats for analysis ───────────────────────────────
    stats_path = args.output.replace(".parquet", "_stats.json")
    stats = []
    for idx, (ncy, nt, r) in sorted(results.items()):
        row = df.loc[idx]
        extra = row.get("extra_info", {})
        if isinstance(extra, str):
            extra = json.loads(extra)
        if hasattr(extra, 'tolist'):
            extra = dict(extra)
        stats.append({
            "row_idx": int(idx),
            "data_source": str(row.get("data_source", "")),
            "task_type": str(extra.get("task_type", "")),
            "section": str(extra.get("section", "")),
            "n_confident_yes": int(ncy),
            "n_correct": int(ncy),
            "n_total": int(nt),
            "confident_yes_rate": float(r),
            "correct_rate": float(r),
            "filtered_out": bool(
                is_filterable_easy(ncy, nt)
                or (args.filter_hard and is_filterable_hard(ncy, nt))
            ),
            "filter_reason": (
                "easy" if is_filterable_easy(ncy, nt)
                else "hard" if args.filter_hard and is_filterable_hard(ncy, nt)
                else ""
            ),
        })
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved per-prompt stats to {stats_path}")


if __name__ == "__main__":
    main()
