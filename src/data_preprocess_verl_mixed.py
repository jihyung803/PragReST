"""
Build a PragReST QA VERL dataset.

This script:
1) creates train/val splits by QA section,
2) converts each example into VERL rows,
3) adds task labels for downstream reward/metrics dispatch,
4) writes train.parquet / val.parquet or all.parquet.

Usage:
  python src/data_preprocess_verl_mixed.py \
      --qa_input data/processed/pragmaticQA_dataset_Qwen14b_new.jsonl \
      --output data/verl-mixed-qwen-8b

  python src/data_preprocess_verl_mixed.py \
      --qa_input data/processed/pragmaticQA_dataset_Qwen14b_new.jsonl \
      --output data/verl-mixed-qwen-8b \
      --no_split
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict

import datasets
from datasets import interleave_datasets

from data_preprocess_verl_qa import make_qa_train_row, make_qa_val_row


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_explanation(assistant_content: str) -> str:
    """Extract the explanation text between </think> and \\boxed{...}."""
    idx = assistant_content.rfind("</think>")
    if idx == -1:
        return ""
    after_think = assistant_content[idx + len("</think>"):].strip()
    boxed_match = re.search(r"\\boxed\{", after_think)
    if boxed_match:
        explanation = after_think[:boxed_match.start()].strip()
    else:
        explanation = after_think.strip()
    return explanation if len(explanation) > 10 else ""


def _build_sft_explanation_lookup(sft_path: str) -> dict[tuple, str]:
    """Build a lookup from SFT data mapping QA key fields to explanations."""
    lookup = {}
    for item in _load_jsonl(sft_path):
        if item.get("task_type", "") != "qa":
            continue

        messages = item.get("messages", [])
        if not messages:
            continue
        assistant_content = messages[-1].get("content", "")
        explanation = _extract_explanation(assistant_content)
        if not explanation:
            continue

        src = item.get("metadata", {}).get("source_row", {})
        key = ("qa", src.get("content", ""), src.get("question", ""))
        lookup[key] = explanation

    print(f"[SFT lookup] Loaded {len(lookup)} QA explanations from {sft_path}")
    return lookup


def _split_qa(raw: list[dict], eval_per_section: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    section_to_exs: dict[str, list] = defaultdict(list)
    for ex in raw:
        section_to_exs[ex.get("section", "unknown")].append(ex)

    raw_train, raw_val = [], []
    for section in sorted(section_to_exs.keys()):
        exs = section_to_exs[section]
        rng.shuffle(exs)
        n_hold = min(eval_per_section, len(exs))
        raw_val.extend(exs[:n_hold])
        raw_train.extend(exs[n_hold:])
    return raw_train, raw_val


def _annotate_rows(
    rows: list[dict],
    task_type: str,
    sft_lookup: dict[tuple, str] | None = None,
) -> list[dict]:
    annotated = []
    n_matched = 0
    for row in rows:
        extra = dict(row.get("extra_info", {}))
        extra["task_type"] = task_type
        # Keep trainer metric compatibility with existing conv_type-based logging.
        extra["conv_type"] = task_type

        if sft_lookup:
            key = ("qa", extra.get("content", ""), extra.get("question", ""))
            explanation = sft_lookup.get(key, "")
            extra["sft_explanation"] = explanation
            if explanation:
                n_matched += 1

        row = dict(row)
        row["extra_info"] = extra
        annotated.append(row)

    if sft_lookup:
        print(f"[SFT match] {task_type}: {n_matched}/{len(rows)} rows matched")
    return annotated


def _roundrobin_by_section(
    rows: list[dict],
    rng: random.Random,
) -> datasets.Dataset:
    """Deterministic round-robin interleave that preserves all rows."""
    if not rows:
        return datasets.Dataset.from_list([])

    group_map: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group_map[row["extra_info"]["section"]].append(row)

    group_keys = sorted(group_map.keys())
    print(f"[RoundRobin] {len(group_keys)} groups: "
          + ", ".join(f"{k}({len(group_map[k])})" for k in group_keys))

    queues = [rng.sample(group_map[k], len(group_map[k])) for k in group_keys]

    result = []
    while any(queues):
        for q in queues:
            if q:
                result.append(q.pop(0))

    print(f"[RoundRobin] Result size: {len(result)} rows")
    return datasets.Dataset.from_list(result)


def _interleave_by_section(
    rows: list[dict],
    rng: random.Random,
    seed: int = 42,
    stopping_strategy: str = "all_exhausted",
) -> datasets.Dataset:
    if not rows:
        return datasets.Dataset.from_list([])

    group_map: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group_map[row["extra_info"]["section"]].append(row)

    group_keys = sorted(group_map.keys())
    print(f"[Interleave] {len(group_keys)} groups: "
          + ", ".join(f"{k}({len(group_map[k])})" for k in group_keys))

    group_datasets = []
    for k in group_keys:
        shuffled = rng.sample(group_map[k], len(group_map[k]))
        group_datasets.append(datasets.Dataset.from_list(shuffled))

    n = len(group_datasets)
    interleaved = interleave_datasets(
        group_datasets,
        probabilities=[1.0 / n] * n,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    print(f"[Interleave] Interleaved dataset size: {len(interleaved)} rows")
    return interleaved


def _to_dataset(rows: list[dict]) -> datasets.Dataset:
    if not rows:
        return datasets.Dataset.from_list([])
    return datasets.Dataset.from_list(rows)


def _print_source_counts(name: str, ds: datasets.Dataset):
    counts = defaultdict(int)
    for src in ds["data_source"] if len(ds) > 0 else []:
        counts[src] += 1
    stats = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts else "empty"
    print(f"[{name}] size={len(ds)} | by_data_source: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess QA JSONL -> VERL Parquet")
    parser.add_argument("--qa_input", required=True, help="Path to QA JSONL")
    parser.add_argument("--output", default="data/verl-mixed-qwen-8b", help="Output directory")
    parser.add_argument("--qa_eval_per_section", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft_input", default=None, help="Path to SFT JSONL for explanation lookup")
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Skip train/val split; output a single all.parquet for downstream filtering",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = random.Random(args.seed)

    sft_lookup = _build_sft_explanation_lookup(args.sft_input) if args.sft_input else None

    qa_raw = _load_jsonl(args.qa_input)
    print(f"Loaded QA={len(qa_raw)} from {args.qa_input}")

    if args.no_split:
        # Use train row constructors with system prompt variations for all rows.
        qa_all_rows = _annotate_rows(
            [make_qa_train_row(ex, rng) for ex in qa_raw],
            task_type="qa",
            sft_lookup=sft_lookup,
        )

        print("[QA all]", end=" ")
        all_ds = _roundrobin_by_section(qa_all_rows, rng)

        all_path = os.path.join(args.output, "all.parquet")
        all_ds.to_parquet(all_path)
        print(f"Saved all samples (no split) -> {all_path}")
        _print_source_counts("All", all_ds)
        return

    qa_train_raw, qa_val_raw = _split_qa(qa_raw, args.qa_eval_per_section, rng)
    print(f"Split QA: train={len(qa_train_raw)} val={len(qa_val_raw)}")

    qa_train_rows = _annotate_rows(
        [make_qa_train_row(ex, rng) for ex in qa_train_raw],
        task_type="qa",
        sft_lookup=sft_lookup,
    )
    qa_val_rows = _annotate_rows(
        [make_qa_val_row(ex) for ex in qa_val_raw],
        task_type="qa",
        sft_lookup=sft_lookup,
    )

    print("[QA train]", end=" ")
    train_ds = _interleave_by_section(
        qa_train_rows,
        rng,
        seed=args.seed,
        stopping_strategy="all_exhausted",
    )

    print("[QA val]", end=" ")
    qa_val_rows_sorted = sorted(qa_val_rows, key=lambda r: r["extra_info"]["section"])
    val_ds = _to_dataset(qa_val_rows_sorted)
    print(f"Validation dataset size: {len(val_ds)} rows")

    train_path = os.path.join(args.output, "train.parquet")
    val_path = os.path.join(args.output, "val.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"Saved QA dataset:\n  train -> {train_path}\n  val   -> {val_path}")
    _print_source_counts("Train", train_ds)
    _print_source_counts("Val", val_ds)

    if len(train_ds) > 0:
        sample = train_ds[0]
        print("\n[SAMPLE QA TRAIN ROW]")
        print(f"  data_source: {sample.get('data_source')}")
        extra = sample.get("extra_info", {})
        print(f"  task_type:   {extra.get('task_type')}")
        print(f"  conv_type:   {extra.get('conv_type')}")
        prompt = sample.get("prompt", [])
        if len(prompt) > 1:
            print(f"  user_prompt (truncated): {prompt[1].get('content', '')[:220]}")


if __name__ == "__main__":
    main()
