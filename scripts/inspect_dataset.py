"""Audit a Hugging Face dataset before training.

Mirrors `hf_inspect_dataset` from ml-intern: you NEVER train on a dataset
without auditing it. Surfaces:
  - splits and row counts
  - column schema and dtypes
  - 3 sample rows
  - missing-value counts
  - duplicate count on the largest text/prompt column
  - basic distribution stats on string-length and label columns

Usage:
    python scripts/inspect_dataset.py trl-lib/Capybara
    python scripts/inspect_dataset.py trl-lib/Capybara --split train --n 5
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()


def _string_length_stats(values: list[str]) -> dict[str, float]:
    lens = [len(v) for v in values if isinstance(v, str)]
    if not lens:
        return {}
    lens.sort()
    return {
        "n": len(lens),
        "min": lens[0], "max": lens[-1],
        "mean": sum(lens) / len(lens),
        "p50": lens[len(lens) // 2],
        "p95": lens[int(len(lens) * 0.95)],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="HF Hub dataset id, e.g. trl-lib/Capybara")
    p.add_argument("--split", default=None, help="Split to load (default: first available)")
    p.add_argument("--config", default=None, help="Dataset config (subset) name")
    p.add_argument("--n", type=int, default=3, help="Sample rows to print")
    p.add_argument("--probe-rows", type=int, default=2000,
                   help="Rows to scan for length/dup stats (-1 = all)")
    args = p.parse_args()

    # Available configs / splits
    try:
        configs = get_dataset_config_names(args.dataset) or [None]
    except Exception:
        configs = [None]
    config = args.config or configs[0]

    splits = get_dataset_split_names(args.dataset, config_name=config)
    split = args.split or splits[0]
    console.print(f"[bold]{args.dataset}[/]  config=[cyan]{config}[/]  splits={splits}")

    ds = load_dataset(args.dataset, name=config, split=split)
    console.print(f"split=[green]{split}[/] rows=[bold]{len(ds):,}[/]")

    # Schema
    table = Table(title="schema")
    table.add_column("column"); table.add_column("dtype"); table.add_column("non-null")
    n = len(ds)
    probe_n = n if args.probe_rows == -1 else min(args.probe_rows, n)
    sample = ds.select(range(probe_n)) if probe_n < n else ds
    for col in ds.column_names:
        non_null = sum(1 for v in sample[col] if v is not None and v != "")
        table.add_row(col, str(ds.features[col]), f"{non_null}/{probe_n}")
    console.print(table)

    # Likely text/prompt column for length + duplicate stats
    text_candidates = [
        c for c in ds.column_names
        if c.lower() in {"text", "prompt", "messages", "chosen", "rejected", "completion", "input", "instruction"}
    ]
    for col in text_candidates:
        values = sample[col]
        # `messages` is conversational — flatten to a string for stats
        if values and isinstance(values[0], list):
            values = [json.dumps(v, ensure_ascii=False) for v in values]
        stats = _string_length_stats(values)
        if stats:
            console.print(f"[yellow]length stats[/] for `{col}`: {stats}")
            counts = Counter(values)
            dups = sum(c - 1 for c in counts.values() if c > 1)
            console.print(f"[yellow]duplicates[/] in `{col}` (probe of {probe_n}): {dups}")

    # Label distribution if obvious label column
    for col in ds.column_names:
        if col.lower() in {"label", "labels", "category", "class"}:
            counts = Counter(sample[col])
            console.print(f"[magenta]label distribution[/] for `{col}` (probe of {probe_n}): "
                          f"{dict(counts.most_common(20))}")

    # Sample rows
    console.print(f"\n[bold]sample rows[/] (n={args.n}):")
    for i in range(min(args.n, len(ds))):
        row = ds[i]
        console.print(json.dumps(row, ensure_ascii=False, default=str)[:1200])
        console.print("---")


if __name__ == "__main__":
    main()
