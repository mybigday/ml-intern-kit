---
name: rocm-strix-halo
description: Set up training and inference on AMD Ryzen AI Max+ 395 / Strix Halo (gfx1151, RDNA 3.5) with TheRock nightly ROCm wheels. Triggered when the host has gfx1151, when `rocminfo` shows Strix Halo, or when the user mentions Strix Halo / Ryzen AI Max / gfx1151 / 128GB unified memory.
---

# rocm-strix-halo — gfx1151 / Ryzen AI Max+ 395

gfx1151 is **not on AMD's official ROCm support matrix** as of this writing
and is **not in upstream PyTorch ROCm wheels** (`download.pytorch.org/whl/rocm6.x`).
Those wheels detect the GPU but die at first compute call with `HIP error:
invalid device function`. The supported path is AMD's **TheRock nightly wheels**.

## Bootstrap

```bash
make env-strix-halo        # equivalent to:
# bash bootstrap.sh --accel=rocm --rocm-nightly --rocm-arch=gfx1151
source .venv/bin/activate
make doctor                # should print: backend=rocm  arch=gfx1151:xnack-
```

Under the hood `bootstrap.sh` installs from
`https://rocm.nightlies.amd.com/v2/gfx1151/` (PEP 503 simple index — published
by AMD in github.com/ROCm/TheRock). Auto-detected if `rocminfo` shows gfx1151.

`bootstrap.sh` also writes one runtime hint to `.venv/bin/activate`:

```sh
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

We deliberately do **NOT** export `PYTORCH_ROCM_ARCH` (build-time only — no
runtime effect on prebuilt wheels) and **NOT** `HSA_OVERRIDE_GFX_VERSION`
(TheRock has native gfx1151 kernels and needs no override). If you fall off
TheRock onto a non-gfx1151 build (Ollama, llama.cpp, stock PyTorch ROCm),
set `HSA_OVERRIDE_GFX_VERSION=11.5.1` — gfx1151 is **RDNA 3.5**, not 3.0
(`11.0.0` is wrong).

## What works out of the box

- `torch`, `torchvision`, `torchaudio`, `triton` — from TheRock nightlies.
- `transformers`, `trl`, `peft`, `accelerate`, `datasets` — pure-Python, work
  unchanged.
- `scripts/train_sft.py`, `scripts/train_dpo.py`, `scripts/train_grpo.py` —
  the platform-overrides branch detects ROCm and drops `flash_attention_2` →
  `sdpa` and `adamw_8bit` → `adamw_torch`. bf16 stays on (gfx1151 supports it).

## What does NOT work out of the box (build-from-source paths)

| Package | State | Workaround |
|---|---|---|
| **bitsandbytes** | 0.49.x source tree lists gfx1151, but no PyPI wheel ships ROCm binaries. TheRock issue #2945 — compilation currently broken on TheRock 7.11. | `cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1151 && make -j`. Stay on `adamw_torch` until that lands. |
| **flash-attention** | No wheel. Stock Triton JIT does not target gfx1151. | Build AOTriton (ahead-of-time Triton kernels) from source — community recipe at github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton. Or just use `attn_implementation="sdpa"`. |
| **vLLM** | PyPI wheel is CUDA-only. ROCm builds work on dGPUs but APU paths (Strix Halo) need guards. | APU-guarded toolbox build (see kyuz0 repo) or stick to `transformers.generate` for now. |

## Memory budget — the actual win on Strix Halo

Strix Halo ships with up to **128 GB unified LPDDR5X** (96 GB allocatable to
GPU under Linux, kernel-tunable). Treat this as one big pool — you do **not**
need LoRA / 4-bit for ≤7B models the way you do on a 24 GB consumer NVIDIA
card.

| Model | Method | Fits comfortably? |
|---|---|---|
| ≤3B | **full SFT** bf16 | yes |
| 7B  | **full SFT** bf16 + grad-checkpoint | yes |
| 13B | full SFT bf16 + grad-checkpoint | tight; LoRA preferred |
| 30B | LoRA bf16 + grad-checkpoint | yes |
| 70B | QLoRA (need bitsandbytes — currently source-build only) | borderline |

If you allocate <96 GB to GPU in BIOS / kernel cmdline, dial back accordingly.
Check actual GPU memory with `make doctor` (it prints `mem=NN GiB`).

## Sanity test after install

```bash
.venv/bin/python -c "
import torch
print('cuda_avail=', torch.cuda.is_available())
print('hip       =', torch.version.hip)
print('arch      =', torch.cuda.get_device_properties(0).gcnArchName)
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
y = x @ x.T
print('matmul ok, dtype=', y.dtype, 'shape=', tuple(y.shape))
"
```

If the matmul works, full-stack training works.

## When to fire

- `make doctor` shows `arch=gfx1151:*`.
- User says "Strix Halo", "Ryzen AI Max+ 395", "AI 395", "gfx1151", "128GB
  unified memory".
- User hits `HIP error: invalid device function` on stock PyTorch ROCm.

## When NOT to use this skill

- gfx1100/1101/1102 (RX 7900/7800/7700 dGPU) — those ARE in upstream PyTorch
  ROCm wheels. Use `make env-rocm` (no nightly).
- gfx942 (MI300) — same: upstream wheels work; use `make env-rocm`.

## References

- TheRock release notes: https://github.com/ROCm/TheRock/blob/main/RELEASES.md
- gfx1151 wheel discussion: https://github.com/ROCm/TheRock/discussions/655
- bitsandbytes ROCm blocker on TheRock 7.11: https://github.com/ROCm/TheRock/issues/2945
- AOTriton + flash-attn for gfx1151: https://github.com/ROCm/ROCm/issues/5404
- Working community recipe: https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton
