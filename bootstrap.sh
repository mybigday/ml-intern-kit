#!/usr/bin/env bash
# Bootstrap a reproducible ml-intern-kit environment on any Linux/macOS box.
# Picks `uv` if present, falls back to python -m venv + pip.
#
# Detects the accelerator and installs the matching PyTorch wheel:
#   - NVIDIA CUDA   → torch + cu124 wheels
#   - AMD ROCm      → torch + rocm6.2 wheels (Linux only)
#   - Apple Silicon → torch with MPS (default wheel)  + optional `--mlx`
#   - Plain CPU     → torch CPU wheels
#
# Usage:
#   bash bootstrap.sh                       # auto-detect
#   bash bootstrap.sh --dev                 # + pytest, ruff, jupyter
#   bash bootstrap.sh --eval                # + lm-eval-harness
#   bash bootstrap.sh --all                 # everything except flash-attn / unsloth
#   bash bootstrap.sh --accel cuda|rocm|mps|cpu     # force accelerator
#   bash bootstrap.sh --mlx                 # also install Apple MLX (arm64 macOS)
set -euo pipefail

cd "$(dirname "$0")"

EXTRAS=""
ACCEL=""
WANT_MLX=0
ROCM_ARCH=""              # e.g. gfx1151, gfx1100, gfx942 — auto-detected
ROCM_NIGHTLY=0            # use AMD TheRock nightly wheels (needed for gfx1151)
for arg in "$@"; do
    case "$arg" in
        --dev)   EXTRAS="${EXTRAS},dev" ;;
        --eval)  EXTRAS="${EXTRAS},eval" ;;
        --all)   EXTRAS="${EXTRAS},dev,eval" ;;
        --mlx)   WANT_MLX=1 ;;
        --rocm-nightly) ROCM_NIGHTLY=1 ;;
        --rocm-arch=*) ROCM_ARCH="${arg#*=}" ;;
        --accel=*) ACCEL="${arg#*=}" ;;
        --accel)  shift; ACCEL="${1:-}";;
        cuda|rocm|mps|cpu) ACCEL="$arg" ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done
EXTRAS="${EXTRAS#,}"

PYTHON_VERSION="$(cat .python-version 2>/dev/null || echo 3.11)"
OS="$(uname -s)"
ARCH="$(uname -m)"

# ---- accelerator detection -------------------------------------------------
if [ -z "$ACCEL" ]; then
    if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
        ACCEL="mps"
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        ACCEL="cuda"
    elif command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then
        ACCEL="rocm"
    else
        ACCEL="cpu"
    fi
fi

# Detect ROCm GPU arch (gfx1151 = Strix Halo / Ryzen AI Max+ 395, RDNA 3.5)
if [ "$ACCEL" = "rocm" ] && [ -z "$ROCM_ARCH" ] && command -v rocminfo >/dev/null 2>&1; then
    ROCM_ARCH="$(rocminfo 2>/dev/null | awk -F': *' '/Name: *gfx[0-9]+/ {print $2; exit}' || true)"
fi

# gfx1151 needs the AMD TheRock nightly wheels — upstream PyTorch ROCm wheels
# don't ship binaries for it yet. Auto-flip the flag.
if [ "$ROCM_ARCH" = "gfx1151" ] && [ "$ROCM_NIGHTLY" = "0" ]; then
    echo "[bootstrap] detected gfx1151 (Strix Halo / Ryzen AI Max+ 395) — switching to TheRock ROCm nightly"
    ROCM_NIGHTLY=1
fi

echo "[bootstrap] OS=$OS arch=$ARCH accel=$ACCEL python=$PYTHON_VERSION rocm_arch=${ROCM_ARCH:-n/a} rocm_nightly=$ROCM_NIGHTLY"

case "$ACCEL" in
    cuda) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    rocm)
        # Upstream PyTorch ROCm wheels (rocm6.x) build for gfx900/906/908/90a/942/1030/1100/1101/1102.
        # gfx1151 (Strix Halo) is NOT in those wheels — it'll detect the GPU but
        # die with "HIP error: invalid device function" at first compute call.
        # AMD's TheRock project publishes per-arch nightly wheels at rocm.nightlies.amd.com/v2/<arch>/.
        # Source: https://github.com/ROCm/TheRock/blob/main/RELEASES.md
        if [ "$ROCM_NIGHTLY" = "1" ] && [ -n "$ROCM_ARCH" ]; then
            TORCH_INDEX="https://rocm.nightlies.amd.com/v2/${ROCM_ARCH}/"
        elif [ "$ROCM_NIGHTLY" = "1" ]; then
            # No arch given — fall back to the broad RDNA3 dGPU channel.
            TORCH_INDEX="https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"
        fi
        ;;
    cpu)  TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
    mps)  TORCH_INDEX="" ;;        # default wheel on macOS arm64 already supports MPS
    *) echo "[bootstrap] unknown accel: $ACCEL"; exit 2 ;;
esac

# ---- install ---------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    echo "[bootstrap] using uv ($(uv --version))"
    uv python install "${PYTHON_VERSION}"
    uv venv --python "${PYTHON_VERSION}"
    if [ -n "$TORCH_INDEX" ]; then
        # Install torch first from the accel-specific index — its wheel name and
        # version differ from PyPI (e.g. TheRock ships 2.7.0a0+rocm7.x).
        uv pip install --index-url "$TORCH_INDEX" "torch>=2.4"
    fi
    # Install the rest from PyPI. Use `uv pip install -r requirements.txt`
    # rather than `uv sync` — sync resolves ALL pyproject extras together
    # (unsloth pins old transformers, vllm conflicts on ROCm, etc.) and fails
    # on perfectly fine target platforms. requirements.txt is the curated
    # cross-platform core.
    PIP_EXCLUDES=()
    if [ "$ACCEL" = "rocm" ] || [ "$ACCEL" = "mps" ] || [ "$ACCEL" = "cpu" ]; then
        # bitsandbytes PyPI wheels are CUDA-only — skip it; trainers fall back to adamw_torch.
        PIP_EXCLUDES+=("bitsandbytes")
    fi
    if [ ${#PIP_EXCLUDES[@]} -gt 0 ]; then
        grep -vE "^($(IFS=\|; echo "${PIP_EXCLUDES[*]}"))(\b|>|=|<)" requirements.txt > /tmp/ml-intern-requirements.txt
        REQ_FILE=/tmp/ml-intern-requirements.txt
    else
        REQ_FILE=requirements.txt
    fi
    uv pip install -r "$REQ_FILE"
    if [[ "$EXTRAS" == *dev* ]]; then
        uv pip install -r requirements-dev.txt
    fi
    if [[ "$EXTRAS" == *eval* ]]; then
        uv pip install "lm-eval>=0.4.10" "inspect-ai>=0.3.149"
    fi
    VENV=".venv"
else
    echo "[bootstrap] uv not found — falling back to python -m venv + pip"
    if ! command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
        echo "Need python${PYTHON_VERSION}. Install it (pyenv / apt / brew) or install uv:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    "python${PYTHON_VERSION}" -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip wheel
    if [ -n "$TORCH_INDEX" ]; then
        pip install --index-url "$TORCH_INDEX" "torch>=2.4"
    fi
    if [[ "$EXTRAS" == *dev* ]]; then
        pip install -r requirements-dev.txt
    else
        pip install -r requirements.txt
    fi
    if [[ "$EXTRAS" == *eval* ]]; then
        pip install "lm-eval>=0.4.5" "inspect-ai>=0.3.149"
    fi
    VENV=".venv"
fi

# Apple MLX (optional, arm64 macOS only)
if [ "$WANT_MLX" = "1" ] || ([ "$ACCEL" = "mps" ] && [ "${SLM_AUTO_MLX:-0}" = "1" ]); then
    if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
        echo "[bootstrap] installing Apple MLX"
        "$VENV/bin/pip" install "mlx>=0.18" "mlx-lm>=0.19"
    else
        echo "[bootstrap] --mlx ignored: requires arm64 macOS"
    fi
fi

# Hub login if HF_TOKEN is in env or .env
if [ -f .env ]; then
    set -o allexport; . ./.env; set +o allexport
fi
if [ -n "${HF_TOKEN:-}" ]; then
    echo "[bootstrap] logging into HF Hub"
    "$VENV/bin/python" -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
fi

# Persist runtime ROCm hints so big models fit unified LPDDR5X cleanly.
# Written into .venv/bin/activate so they auto-load on `source .venv/bin/activate`.
#
# Note: PYTORCH_ROCM_ARCH is build-time only (no runtime effect on prebuilt
# wheels), so we deliberately do NOT export it.
#
# HSA_OVERRIDE_GFX_VERSION: TheRock nightlies have native gfx1151 kernels and
# need NO override. We leave it unset by default. Some downstream tools
# (Ollama, llama.cpp builds, certain non-TheRock PyTorch wheels) still need
# `11.5.1` (gfx1151 = RDNA 3.5) — NOT 11.0.0 — set it manually if you hit
# "invalid device function".
if [ "$ACCEL" = "rocm" ] && [ "$ROCM_ARCH" = "gfx1151" ]; then
    ACTIVATE="$VENV/bin/activate"
    if [ -f "$ACTIVATE" ] && ! grep -q "# ml-intern-kit rocm hints" "$ACTIVATE"; then
        {
            echo ""
            echo "# ml-intern-kit rocm hints — gfx1151 / Strix Halo (added by bootstrap.sh)"
            echo "export PYTORCH_HIP_ALLOC_CONF=\${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
            # AOTriton flash/efficient SDPA kernels for gfx1151 ship inside the
            # TheRock torch wheel (libaotriton_v2.so.0.11.2+) but PyTorch keeps
            # them behind this gate. Without it, sdpa silently falls back to
            # the math kernel (~20x slower). HF Transformers' "flash_attention_2"
            # backend imports the upstream `flash_attn` PyPI package which is
            # CUDA-only — keep `attn_implementation="sdpa"` and let SDPA pick
            # the AOTriton flash backend.
            echo "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=\${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
            # Uncomment if you fall off TheRock onto a non-gfx1151 build:
            echo "# export HSA_OVERRIDE_GFX_VERSION=11.5.1"
        } >> "$ACTIVATE"
        echo "[bootstrap] wrote ROCm env hints to $ACTIVATE"
    fi
fi

echo
echo "[bootstrap] done.  accel=$ACCEL  rocm_arch=${ROCM_ARCH:-n/a}"
echo "[bootstrap] activate with:  source $VENV/bin/activate"
echo "[bootstrap] sanity check:   make doctor"
if [ "$ROCM_ARCH" = "gfx1151" ]; then
    echo "[bootstrap] Strix Halo: see .claude/skills/rocm-strix-halo/SKILL.md for the full recipe"
fi
