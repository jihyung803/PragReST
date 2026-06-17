"""Split a filtered VERL parquet into train.parquet and val.parquet."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any

import pandas as pd


def _extra_info_value(value: Any) -> dict:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, dict) else {}
    return {}


def _section(row: pd.Series) -> str:
    extra = _extra_info_value(row.get("extra_info", {}))
    return str(extra.get("section", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a filtered VERL parquet by section."
    )
    parser.add_argument("--input", required=True, help="Filtered parquet file")
    parser.add_argument("--output", required=True, help="Directory for train/val parquet")
    parser.add_argument("--eval_per_section", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    rng = random.Random(args.seed)

    section_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        section_to_indices[_section(row)].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for section in sorted(section_to_indices):
        indices = list(section_to_indices[section])
        rng.shuffle(indices)
        n_holdout = min(args.eval_per_section, len(indices))
        val_indices.extend(indices[:n_holdout])
        train_indices.extend(indices[n_holdout:])

    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train_filtered.parquet")
    val_path = os.path.join(args.output, "val_filtered.parquet")
    df.loc[train_indices].reset_index(drop=True).to_parquet(train_path)
    df.loc[val_indices].reset_index(drop=True).to_parquet(val_path)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Sections: {len(section_to_indices)}")
    print(f"Saved train rows: {len(train_indices)} -> {train_path}")
    print(f"Saved val rows:   {len(val_indices)} -> {val_path}")


if __name__ == "__main__":
    main()
