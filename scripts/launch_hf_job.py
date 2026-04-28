"""Submit a training script to Hugging Face Jobs (`hf jobs run`) with the
ml-intern pre-flight discipline baked in.

Usage:
    python scripts/launch_hf_job.py \
        --script scripts/train_sft.py \
        --config configs/sft_default.yaml \
        --hardware a10g-largex2 \
        --timeout 4h \
        --hub-model-id you/your-model

What this enforces (refuses to launch otherwise):
  - `--hub-model-id` is set (job storage is ephemeral; without push_to_hub the
    trained model is permanently lost).
  - `--timeout` ≥ 2h for any training script.
  - The reference script exists.

If `hf` CLI is missing, prints the equivalent `huggingface_hub.run_job(...)`
Python call you can run instead.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GPU_FAMILY_HINT = {
    "a10g-largex2": "1-3B params",
    "a100-large": "7-13B params",
    "l40sx4": "30B+ params",
    "a100x4": "30B+ params",
    "a100x8": "70B+ params",
}


def _parse_timeout_hours(s: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", s.lower())
    if not m:
        raise ValueError(f"timeout must look like '4h', '120m', '2h30m'. Got: {s!r}")
    n, unit = float(m.group(1)), m.group(2)
    return n * {"s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}[unit]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", type=Path, required=True, help="Local training script to upload.")
    p.add_argument("--config", type=Path, required=True, help="YAML config the script will read.")
    p.add_argument("--hardware", default="a10g-largex2",
                   help="HF Jobs hardware flavor. See AGENTS.md for sizing.")
    p.add_argument("--timeout", default="4h",
                   help="Wall-clock budget. ≥2h for any training run. Default 4h.")
    p.add_argument("--hub-model-id", required=True,
                   help="Where to push_to_hub. Required — job storage is ephemeral.")
    p.add_argument("--namespace", default=os.environ.get("HF_HUB_NAMESPACE"),
                   help="Optional org namespace (uses your username if omitted).")
    p.add_argument("--dependencies", nargs="*", default=[
        "transformers>=5.5", "trl>=0.28", "peft>=0.18", "accelerate>=1.12",
        "datasets>=4.7", "trackio>=0.24", "bitsandbytes>=0.49", "hf_transfer",
    ])
    p.add_argument("--extra-dep", action="append", default=[])
    p.add_argument("--env", action="append", default=[],
                   help="Extra env: KEY=VALUE (repeat).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Pre-flight
    errors = []
    if not args.script.exists():
        errors.append(f"script not found: {args.script}")
    if not args.config.exists():
        errors.append(f"config not found: {args.config}")
    try:
        hours = _parse_timeout_hours(args.timeout)
        if hours < 2:
            errors.append(f"timeout {args.timeout!r} < 2h — training jobs need ≥2h. "
                          "Override only if this is data prep / inference.")
    except ValueError as e:
        errors.append(str(e))
    if errors:
        for e in errors:
            print(f"[launch_hf_job] ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    deps = list(args.dependencies) + list(args.extra_dep)

    print("[launch_hf_job] PRE-FLIGHT")
    print(f"  script        : {args.script}")
    print(f"  config        : {args.config}")
    print(f"  hardware      : {args.hardware}  ({GPU_FAMILY_HINT.get(args.hardware, 'check sizing')})")
    print(f"  timeout       : {args.timeout}")
    print(f"  hub_model_id  : {args.hub_model_id}")
    print(f"  dependencies  : {deps}")
    print()

    if args.dry_run:
        print("[launch_hf_job] --dry-run set, exiting before submission.")
        return

    if not shutil.which("hf"):
        print("[launch_hf_job] `hf` CLI not on PATH. Install it: pip install 'huggingface_hub[cli]'")
        print("Or submit programmatically:")
        print("    from huggingface_hub import run_job")
        print(f"    run_job(script='{args.script}', flavor='{args.hardware}', "
              f"timeout='{args.timeout}', namespace='{args.namespace}',")
        print(f"            dependencies={deps},")
        print(f"            script_args=['--config', '{args.config}', "
              f"'--hub-model-id', '{args.hub_model_id}'])")
        sys.exit(1)

    cmd = [
        "hf", "jobs", "run",
        "--flavor", args.hardware,
        "--timeout", args.timeout,
    ]
    if args.namespace:
        cmd += ["--namespace", args.namespace]
    for dep in deps:
        cmd += ["--with", dep]
    for kv in args.env:
        cmd += ["--env", kv]
    cmd += [
        "python:3.11",
        "python", str(args.script),
        "--config", str(args.config),
        "--hub-model-id", args.hub_model_id,
    ]
    print("[launch_hf_job] $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
