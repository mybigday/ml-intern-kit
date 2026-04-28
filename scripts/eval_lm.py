"""Convenience wrapper around `lm-eval-harness` for quick model evaluation.

Usage:
    python scripts/eval_lm.py --model HuggingFaceTB/SmolLM2-360M --tasks arc_easy,hellaswag
    python scripts/eval_lm.py --model outputs/sft-default --tasks gsm8k --batch-size 4
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Hub repo id or local path.")
    p.add_argument("--tasks", required=True, help="Comma-separated list, e.g. arc_easy,hellaswag,gsm8k")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Limit examples per task (smoke).")
    p.add_argument("--output", default="outputs/eval")
    p.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "auto")
    args = p.parse_args()

    if not shutil.which("lm_eval") and not shutil.which("lm-eval"):
        print("lm-eval-harness not installed. Run: pip install 'lm-eval>=0.4.5'", file=sys.stderr)
        sys.exit(2)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model},trust_remote_code=True",
        "--tasks", args.tasks,
        "--batch_size", str(args.batch_size),
        "--num_fewshot", str(args.num_fewshot),
        "--device", args.device,
        "--output_path", args.output,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
