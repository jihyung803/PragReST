"""Derive an easy-only filtered parquet from filter_prompts.py stats.

The input stats may come from a filtering run that dropped both easy and hard
prompts. This helper removes only rows marked as easy and keeps hard, mixed,
and generation-error rows for a separate training variant.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a parquet file that drops only easy prompts."
    )
    parser.add_argument("--input", required=True, help="Original parquet file")
    parser.add_argument("--stats", required=True, help="Stats JSON from filter_prompts.py")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    with open(args.stats, "r", encoding="utf-8") as f:
        stats = json.load(f)

    easy_indices = {
        int(row["row_idx"])
        for row in stats
        if str(row.get("filter_reason", "")).lower() == "easy"
    }
    keep_indices = [idx for idx in df.index if int(idx) not in easy_indices]
    filtered = df.loc[keep_indices].reset_index(drop=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    filtered.to_parquet(args.output)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Dropped easy rows: {len(easy_indices)}")
    print(f"Saved {len(filtered)} rows to {args.output}")


if __name__ == "__main__":
    main()
