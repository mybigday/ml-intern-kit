---
name: train-mlx
description: LoRA fine-tune a causal LM natively on Apple Silicon (M1/M2/M3/M4) via `mlx-lm`. Triggered when the user is on macOS arm64, mentions MLX, or wants better throughput than torch-MPS. Use this instead of `train-sft` when the host is a Mac with no NVIDIA GPU.
---

# train-mlx — Apple MLX LoRA fine-tune

For everything except Apple Silicon, prefer the `train-sft` skill (torch +
TRL). MLX wins on M-series because the unified memory means a 7B model that
spills VRAM on a 24GB consumer NVIDIA card fits comfortably on a 36GB M-series.

## When to fire

- `uname -sm` returns `Darwin arm64`.
- User mentions MLX, M-series, "Mac GPU", or unified memory.
- Torch-MPS keeps OOM-ing or running out of MPS-supported ops.

## Sequence

1. **Install the MLX path** (idempotent):
   ```bash
   make env-mlx        # bootstrap with --accel=mps --mlx
   ```
2. **Get the data into MLX's JSONL layout**:
   ```bash
   python scripts/dump_mlx_jsonl.py trl-lib/Capybara data/capybara-mlx
   ```
   Produces `train.jsonl` + `valid.jsonl` with the `messages` (or `text`)
   field. You can skip this if you already have HF JSONL on disk.
3. **Pick a 4-bit MLX-community base** for fast loading on 16-32GB Macs:
   - `mlx-community/SmolLM2-1.7B-Instruct-4bit`
   - `mlx-community/Llama-3.2-3B-Instruct-4bit`
   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
4. **Smoke run, then scale**:
   ```bash
   python scripts/mlx_finetune.py --config configs/mlx_default.yaml --iters 50
   python scripts/mlx_finetune.py --config configs/mlx_default.yaml --iters 1000 --fuse
   ```
   `--fuse` merges the LoRA back into the base weights so you can load the
   result with plain `mlx_lm.generate` or convert back to safetensors.
5. **Push to Hub** (optional): `mlx_lm.fuse --upload-repo you/your-mlx-model`.

## Memory budget rules of thumb

| Model | Min unified memory | Notes |
|---|---|---|
| ≤1.7B | 16 GB | works on M1/M2 base |
| 3B    | 24 GB | M2 Pro / M3 Pro |
| 7B 4-bit | 36 GB | M3 Max / M4 Pro |
| 13B 4-bit | 64 GB | M2 Ultra / M3 Max |

## What MLX does NOT support

- DPO/GRPO out of the box (mlx-lm has SFT + LoRA + DoRA — for preference data,
  generate offline preferences and do SFT-on-chosen, or use torch-MPS via
  `train-dpo` skill at slower speeds).
- Multi-GPU (single-Mac only — fine, since you have one accelerator anyway).
- `bitsandbytes`-style 8-bit optimizers (MLX has its own quantization).

## Eval

`scripts/eval_lm.py` (lm-eval-harness) loads MLX-fused safetensors fine. For
in-process MLX eval: `mlx_lm.generate --model outputs/mlx-lora-fused
--prompt "..."`.
