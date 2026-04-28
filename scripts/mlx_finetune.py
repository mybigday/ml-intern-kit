"""Apple Silicon native LoRA fine-tune via `mlx-lm`.

Use this on M-series Macs when you want better throughput than torch-MPS.
For everything else (CUDA / ROCm / CPU), use `scripts/train_sft.py`.

Install:
    make env-mlx        # bootstrap with --accel=mps --mlx
    # or, if you already have torch-MPS:
    make mlx

Usage:
    python scripts/mlx_finetune.py --config configs/mlx_default.yaml --iters 200

Under the hood this shells out to `mlx_lm.lora`, which is the supported entry
point in mlx-lm ≥ 0.19. We keep this file thin so it tracks upstream cleanly.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="YAML recipe (see configs/mlx_default.yaml)")
    p.add_argument("--iters", type=int, default=None, help="Override num training iters.")
    p.add_argument("--data", type=Path, default=None, help="Override JSONL data dir (must contain train.jsonl/valid.jsonl).")
    p.add_argument("--adapter-path", type=Path, default=None, help="Override LoRA adapter output dir.")
    p.add_argument("--fuse", action="store_true",
                   help="After training, fuse the LoRA adapter back into the base model and save the merged weights.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    model = cfg["model"]["name_or_path"]
    data = str(args.data or cfg["dataset"]["path"])
    adapter = str(args.adapter_path or cfg.get("output", {}).get("adapter_path", "outputs/mlx-lora"))
    iters = int(args.iters or cfg.get("train", {}).get("iters", 200))
    batch_size = int(cfg.get("train", {}).get("batch_size", 4))
    lr = float(cfg.get("train", {}).get("learning_rate", 1e-5))
    lora_layers = int(cfg.get("peft", {}).get("layers", 16))

    if not shutil.which("mlx_lm.lora"):
        print("mlx-lm not installed. Run: make env-mlx   (or: make mlx)", file=sys.stderr)
        sys.exit(2)

    cmd = [
        "mlx_lm.lora",
        "--train",
        "--model", model,
        "--data", data,
        "--adapter-path", adapter,
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", str(lr),
        "--num-layers", str(lora_layers),
    ]
    print("[mlx_finetune] $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    if args.fuse:
        fused = cfg.get("output", {}).get("fused_path", f"{adapter}-fused")
        fuse_cmd = [
            "mlx_lm.fuse",
            "--model", model,
            "--adapter-path", adapter,
            "--save-path", fused,
        ]
        print("[mlx_finetune] $ " + " ".join(fuse_cmd))
        subprocess.run(fuse_cmd, check=True)


if __name__ == "__main__":
    main()
