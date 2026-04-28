---
name: env-bootstrap
description: Recreate the ml-intern-kit Python environment on a new machine (laptop, rented GPU box, Docker, fresh checkout). Triggered when the user is on a new host, sees ImportError on a core dep (torch/transformers/trl/peft/accelerate), or wants to install flash-attn / unsloth / bitsandbytes after the fact.
---

# env-bootstrap — recreate the env anywhere

## Default path

```bash
bash bootstrap.sh           # auto-detects accelerator (cuda / rocm / mps / cpu)
bash bootstrap.sh --dev     # + pytest, ruff, jupyter
bash bootstrap.sh --eval    # + lm-eval-harness
bash bootstrap.sh --all     # everything except flash-attn / unsloth / mlx
source .venv/bin/activate
make doctor                 # prints torch backend, GPU info, mlx if installed
```

## Force a specific accelerator

```bash
make env-cuda    # NVIDIA CUDA 12.4 wheels
make env-rocm    # AMD ROCm 6.2 wheels (gfx942 / gfx110X dGPUs only)
make env-strix-halo  # AMD Ryzen AI Max+ 395 / gfx1151 (TheRock nightly)
make env-mps     # macOS / Apple Silicon, torch-MPS only
make env-mlx     # macOS arm64: torch-MPS + Apple MLX
make env-cpu     # CPU-only wheels
```

`bootstrap.sh` picks the right PyTorch wheel index automatically:
`cu124` for NVIDIA, `rocm6.2` for AMD, default (MPS) for arm64 macOS,
`cpu` everywhere else.

## When `uv` isn't installed

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`bootstrap.sh` will pick it up automatically. If installing uv isn't an option,
the script falls back to `python3.11 -m venv` + `pip install -r requirements.txt`.

## When you need flash-attn or unsloth

These have CUDA / ABI requirements that vary per box, so they are deliberately
**not** in `requirements.txt`. Install after the core stack works:

```bash
make flash      # flash-attn, requires matching CUDA toolkit
make unsloth    # unsloth, Linux only
```

If `make flash` fails with a build error, `attn_implementation="sdpa"` in the
training config is a good fallback — slower than flash-attn-2 but works
everywhere PyTorch does.

## When you need a fully-pinned, identical env (Docker)

```bash
docker build -t ml-intern-kit:cu124 .
docker run --gpus all -it --rm \
    -v "$PWD":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    --env-file .env \
    ml-intern-kit:cu124 bash
```

The Dockerfile pins CUDA 12.4 + Python 3.11 + uv-resolved deps from
`pyproject.toml`. Cache mounts let the host keep model/dataset downloads
across container runs.

## Common breakage on fresh boxes

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False on NVIDIA | Wrong wheel. `make env-cuda` re-pulls from `cu124` index. |
| `torch.cuda.is_available()` is False on AMD | ROCm uses the CUDA API surface — `torch.cuda.is_available()` *should* be True with a `torch.version.hip` set. If False: `make env-rocm`, then `rocminfo`. |
| `bitsandbytes` import error on macOS / ROCm | Expected — gated to CUDA Linux. The training scripts auto-substitute `optim="adamw_torch"` on those backends. |
| `bf16` errors / NaN loss on macOS | `train_sft.py` auto-switches to `fp16` on MPS. If you see this on torch ≤ 2.3, upgrade torch. |
| `flash-attn` build fails | Skip it; the scripts auto-fall-back to `sdpa`. flash-attn is CUDA-only upstream. |
| MPS op-not-implemented error | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall through to CPU for that op, or use the `train-mlx` path instead. |
| HF Hub auth fails | `hf auth login` (or `huggingface-cli login`) and re-run, or set `HF_TOKEN` in `.env`. |
