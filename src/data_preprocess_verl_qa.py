"""
Convert PragReST pragmatic QA JSONL to VERL Parquet.

Input schema (per line):
  section  : str  – pragmatics category
  content  : str  – context paragraph
  question : str  – pragmatic question
  answer   : str  – ground-truth answer

Output schema (per row):
  data_source  : "pragrest_qa"
  prompt       : list  – chat messages [{role, content}, ...]
  reward_model : dict  – {"style": "model", "ground_truth": <answer>}
  extra_info   : dict  – content, question, reference, section

Each raw example produces ONE Parquet row.

Usage:
  python src/data_preprocess_verl_qa.py \
      --input  data/processed/pragmaticQA_dataset_Qwen14b_new.jsonl \
      --output data/verl-qa \
      --eval_per_section 1
"""

import argparse
import json
import os
import random
from collections import defaultdict

import datasets
from datasets import interleave_datasets

# ── prompt templates ──────────────────────────────────────────────────────────

# Match SFT neutral prompt style.
SYSTEM_PROMPTS = [
    "You answer the given question from the provided context. ",
]

_FORMAT_INSTRUCTION = "\n\nWrite your final response in \\boxed{} on the last line."


def _make_qa_user_msg(content: str, question: str) -> str:
    return (
        "Task: QA.\n"
        f"Context:\n{content}\n\n"
        f"Question:\n{question}\n\n"
        "Answer the question based on the context.\n"
        "Output only the final answer text.\n"
        f"{_FORMAT_INSTRUCTION}"
    )


def make_qa_train_row(example, rng: random.Random) -> dict:
    """Convert one QA example to a single VERL-schema row (SFT-aligned prompt)."""
    section   = example.get("section", "unknown")
    content   = example.get("content", "")
    question  = example.get("question", "")
    reference = example.get("answer", "")

    return {
        "data_source": "pragrest_qa",
        "prompt": [
            {"role": "system", "content": rng.choice(SYSTEM_PROMPTS)},
            {"role": "user",   "content": _make_qa_user_msg(content, question)},
        ],
        "reward_model": {"style": "model", "ground_truth": reference},
        "extra_info": {
            "content":   content,
            "question":  question,
            "reference": reference,
            "section":   section,
        },
    }


def make_qa_val_row(example) -> dict:
    """Convert one QA example to a deterministic VERL-schema row for validation."""
    section   = example.get("section", "unknown")
    content   = example.get("content", "")
    question  = example.get("question", "")
    reference = example.get("answer", "")

    return {
        "data_source": "pragrest_qa",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPTS[0]},
            {"role": "user",   "content": _make_qa_user_msg(content, question)},
        ],
        "reward_model": {"style": "model", "ground_truth": reference},
        "extra_info": {
            "content":   content,
            "question":  question,
            "reference": reference,
            "section":   section,
        },
    }


def _interleave_by_section(
    rows: list[dict],
    rng: random.Random,
    seed: int = 42,
    stopping_strategy: str = "all_exhausted",
) -> datasets.Dataset:
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


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PragReST pragmatic QA JSONL to VERL Parquet"
    )
    parser.add_argument("--input",  required=True, help="Path to QA JSONL file")
    parser.add_argument("--output", default="data/verl-qa", help="Output directory")
    parser.add_argument("--eval_per_section", type=int, default=1,
                        help="Number of examples held out per section for validation")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--no_interleave", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(raw)} raw examples from {args.input}")

    rng = random.Random(args.seed)

    # Stratified split by section
    section_to_exs: dict[str, list] = defaultdict(list)
    for ex in raw:
        section_to_exs[ex.get("section", "unknown")].append(ex)

    raw_train, raw_val = [], []
    for section in sorted(section_to_exs.keys()):
        exs = section_to_exs[section]
        rng.shuffle(exs)
        n_hold = min(args.eval_per_section, len(exs))
        raw_val.extend(exs[:n_hold])
        raw_train.extend(exs[n_hold:])

    print(f"Split: {len(raw_train)} train, {len(raw_val)} val "
          f"({args.eval_per_section} held out per section "
          f"× {len(section_to_exs)} sections)")

    train_rows = [make_qa_train_row(ex, rng) for ex in raw_train]
    val_rows   = [make_qa_val_row(ex)        for ex in raw_val]

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows:   {len(val_rows)}")

    if not args.no_interleave:
        train_ds = _interleave_by_section(train_rows, rng, seed=args.seed,
                                          stopping_strategy="all_exhausted")
        val_ds   = _interleave_by_section(val_rows,   rng, seed=args.seed,
                                          stopping_strategy="first_exhausted")
    else:
        train_ds = datasets.Dataset.from_list(train_rows)
        val_ds   = datasets.Dataset.from_list(val_rows)

    train_path = os.path.join(args.output, "train.parquet")
    val_path   = os.path.join(args.output, "val.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    print(f"Saved:\n  train → {train_path}\n  val   → {val_path}")

    if train_rows:
        s = train_rows[0]
        print("\n[SAMPLE TRAIN ROW]")
        print(f"  data_source: {s['data_source']}")
        print(f"  section:     {s['extra_info']['section']}")
        print(f"  question:    {s['extra_info']['question'][:80]}")
        print(f"  reference:   {s['extra_info']['reference'][:80]}")
        print(f"  prompt[1] (truncated): {s['prompt'][1]['content'][:300]}")


if __name__ == "__main__":
    main()
