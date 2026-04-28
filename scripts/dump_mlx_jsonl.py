"""Dump a HF dataset to the JSONL format `mlx_lm.lora` expects.

mlx-lm wants a directory containing `train.jsonl` and `valid.jsonl`, each line
being either {"text": "..."} or {"messages": [...]}. This converts a HF
dataset into that layout, with an optional 95/5 valid split.

Usage:
    python scripts/dump_mlx_jsonl.py trl-lib/Capybara data/capybara-mlx
    python scripts/dump_mlx_jsonl.py argilla/distilabel-capybara-dpo-7k-binarized \
        data/capybara-dpo --field chosen --max 5000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="HF dataset repo id")
    p.add_argument("output_dir", type=Path, help="Where to write train.jsonl / valid.jsonl")
    p.add_argument("--split", default="train")
    p.add_argument("--field", default=None,
                   help="Source field. Default: 'messages' if present, else 'text'.")
    p.add_argument("--max", type=int, default=None, help="Cap rows.")
    p.add_argument("--valid-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    field = args.field or ("messages" if "messages" in ds.column_names else "text")
    if field not in ds.column_names:
        raise SystemExit(f"field {field!r} not in dataset columns: {ds.column_names}")

    rows = list(ds.select(range(min(args.max, len(ds)))) if args.max else ds)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_valid = max(1, int(len(rows) * args.valid_frac))
    valid, train = rows[:n_valid], rows[n_valid:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, batch in (("train.jsonl", train), ("valid.jsonl", valid)):
        path = args.output_dir / name
        with path.open("w", encoding="utf-8") as f:
            for r in batch:
                f.write(json.dumps({field: r[field]}, ensure_ascii=False) + "\n")
        print(f"wrote {path}  ({len(batch):,} rows)")


if __name__ == "__main__":
    main()
