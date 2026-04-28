# ml-intern-kit — common commands.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
UV  ?= $(shell command -v uv 2>/dev/null)

.PHONY: env env-dev env-all env-cuda env-rocm env-mps env-cpu env-mlx \
        doctor flash unsloth mlx fmt lint test \
        train-sft train-dpo train-grpo train-mlx eval inspect clean

env:        ## minimal training stack (auto-detect accelerator)
	bash bootstrap.sh

env-dev:    ## + pytest, ruff, jupyter
	bash bootstrap.sh --dev

env-all:    ## + eval + dev
	bash bootstrap.sh --all

env-cuda:   ## force NVIDIA CUDA (cu124) wheels
	bash bootstrap.sh --accel=cuda

env-rocm:   ## force AMD ROCm (rocm6.2) wheels — Linux only
	bash bootstrap.sh --accel=rocm

env-strix-halo: ## AMD Ryzen AI Max+ 395 / gfx1151 (TheRock nightly wheels)
	bash bootstrap.sh --accel=rocm --rocm-nightly --rocm-arch=gfx1151

env-mps:    ## force macOS / Apple Silicon (torch MPS)
	bash bootstrap.sh --accel=mps

env-cpu:    ## force CPU-only wheels (laptops, CI, smoke tests)
	bash bootstrap.sh --accel=cpu

env-mlx:    ## macOS arm64: torch-MPS + Apple MLX
	bash bootstrap.sh --accel=mps --mlx

doctor:     ## verify torch / cuda / rocm / mps / mlx / hf
	@$(PY) scripts/doctor.py

flash:      ## install flash-attn (Linux + correct CUDA only)
	$(PIP) install --no-build-isolation "flash-attn>=2.6"

unsloth:    ## install unsloth (Linux + CUDA only)
	$(PIP) install "unsloth>=2024.10"

mlx:        ## install Apple MLX (macOS arm64 only)
	$(PIP) install "mlx>=0.30" "mlx-lm>=0.30"

fmt:
	$(PY) -m ruff format scripts

lint:
	$(PY) -m ruff check scripts

test:
	$(PY) -m pytest -q

inspect:    ## inspect a dataset: make inspect DS=trl-lib/Capybara
	$(PY) scripts/inspect_dataset.py $(DS)

train-sft:  ## smoke test SFT on the default config (torch — CUDA/ROCm/MPS/CPU)
	$(PY) scripts/train_sft.py --config configs/sft_default.yaml --max-steps 20 --max-samples 256

train-mlx:  ## macOS arm64 native LoRA fine-tune via mlx-lm
	$(PY) scripts/mlx_finetune.py --config configs/mlx_default.yaml --iters 50

train-dpo:
	$(PY) scripts/train_dpo.py --config configs/sft_default.yaml --max-steps 20 --max-samples 256

train-grpo:
	$(PY) scripts/train_grpo.py --config configs/sft_default.yaml --max-steps 20 --max-samples 256

eval:       ## quick lm-eval-harness sanity run: make eval MODEL=... TASKS=arc_easy
	$(PY) scripts/eval_lm.py --model $(MODEL) --tasks $(TASKS)

clean:
	rm -rf outputs/ runs/ wandb/ trackio/ .pytest_cache/ .ruff_cache/

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'
