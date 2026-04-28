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
  the platform-overrides branch detects ROCm, sets
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, forces
  `attn_implementation="sdpa"` (so SDPA picks the AOTriton flash kernel), and
  drops `adamw_8bit` → `adamw_torch`. bf16 stays on (gfx1151 supports it).
- **AOTriton flash-attention via SDPA** — `libaotriton_v2.so.0.11.2` ships
  inside the TheRock torch wheel (`torch/lib/`). Verified ~20× speedup vs the
  math kernel on this exact host (0.65 ms vs 12.80 ms per
  `(2, 16, 1024, 64)` bf16 attention call), matching ROCm/ROCm#6034.

## Two flash-attention paths — pick the one that works

There are two routes named "flash attention" in HF Transformers, and on
gfx1151 only one of them is real:

| Route | What it imports | Status on gfx1151 (verified 2026-04-28) |
|---|---|---|
| `attn_implementation="sdpa"` + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | torch SDPA → AOTriton 0.11.2b kernel shipped inside the wheel | **Working.** ~20× speedup over math fallback. This is the path our scripts use. |
| `attn_implementation="flash_attention_2"` | upstream `flash_attn` PyPI wheel (Tri Dao) | **Not working.** PyPI wheel is CUDA-only. The ROCm fork (`ROCm/flash-attention`) lists "MI200x, MI250x, MI300x, MI355x, RDNA 3/4" — gfx1151 (RDNA 3.5 APU) is explicitly missing, and even on listed RDNA3 dGPUs the backward kernel is unsupported. Don't go here. |

Without `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` set, PyTorch silently
falls back to the math SDPA kernel and prints
`Flash attention was not compiled for current AMD GPU architecture` — the
warning is misleading, the kernel IS compiled, the env-var gate is just
closed. `bootstrap.sh` writes the export into `.venv/bin/activate` so it
loads automatically; the trainer scripts also set it defensively.

Re: aule-attention (https://github.com/AuleTechnologies/Aule-Attention) — its
Triton backend lists MI300X / MI300A / MI250X / MI250 / MI210 / MI100 and its
Vulkan backend covers RDNA3 dGPUs / RDNA2; **gfx1151 is on neither path**.
README claims `<1e-3` fp16 deviation vs torch SDPA but ships no public
training-loss reproduction or test-suite metrics. Not a recommended training
path on this arch yet.

## What still does NOT work out of the box (build-from-source paths)

| Package | State (verified 2026-04-28) | Workaround |
|---|---|---|
| **bitsandbytes** | Source build on TheRock 7.11 nightly is **confirmed working** (TheRock #2945 closed 2026-02-03). Still no precompiled ROCm wheel on PyPI. | Build from source — see commands below. After that, `optim="adamw_8bit"` works. |
| **vLLM** | PyPI wheel is CUDA-only. ROCm builds work on dGPUs but APU paths (Strix Halo) need guards. | APU-guarded toolbox build (see kyuz0 repo) or stick to `transformers.generate` for now. |

## bf16 numerical caveats on gfx1151 (ROCm/ROCm#6034 — still OPEN)

AMD acknowledged 5 distinct bf16-related bugs that hit training stability on
gfx1151. None of these are AOTriton kernel bugs — they're in the broader
HIP/torch.compile path. Plan around them:

- NaN within ~15 steps when `total_batch_size <= 2**13` (8192 tokens)
- NaN at `head_dim=32` (use 64+)
- NaN at network depth ≥ 12
- NaN with Adam `beta2 < 0.97` (default is fine; check if you tweak)
- Cumulative bf16 drift causing crashes around step ~1000 on long runs

Mitigation: prefer larger device batches, head_dim 64+, depth ≤ 10, default
betas. Checkpoint frequently. If you see drift, restart from the last good
checkpoint or fall back to `attn_implementation="eager"` for diagnosis.

### Build bitsandbytes from source on TheRock 7.11+

Confirmed recipe from TheRock #2945:

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes

source .venv/bin/activate          # the venv with TheRock torch installed
export ROCM_HOME=/opt/rocm
export HIP_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx1151    # build-time only — required here

cmake -DCOMPUTE_BACKEND=hip -S .
make -j$(nproc)
pip install -e . --no-build-isolation

python -c "import bitsandbytes; print('ok')"
```

Note that `PYTORCH_ROCM_ARCH=gfx1151` IS load-bearing at build time (just not
at runtime). The TheRock environment also still requires the TheRock-specific
env vars from issue #1658 to be set in some setups; if `cmake` can't find ROCm
headers, `export ROCM_PATH=$(python -c "from rocm_sdk import path; print(path())")`
or fall back to `/opt/rocm`.

### Enable AOTriton flash-attention on ROCm 7.1+

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Then `attn_implementation="flash_attention_2"` works inside the ROCm 7.1 PyTorch
container. On bare-metal TheRock nightlies, AOTriton must be built locally
(0.11 added gfx1151 support — see ROCm/aotriton commit `0e7d518`).

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
- bitsandbytes source-build recipe (closed 2026-02-03): https://github.com/ROCm/TheRock/issues/2945
- AOTriton 0.11 + gfx1151 (closed 2025-11-01): https://github.com/ROCm/ROCm/issues/5404
- AOTriton gfx1151 commit: https://github.com/ROCm/aotriton/commit/0e7d518e83e6ada6a41cbb8b8b48f50c652fdeeb
- ROCm 7.1 PyTorch install (Radeon / Ryzen AI): https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-pytorch.html
- TheRock env-var blocker (still open): https://github.com/ROCm/TheRock/issues/1658
- Working community recipe: https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton
