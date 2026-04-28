# Reproducible CUDA training image. Mirrors the local env from pyproject.toml
# via uv sync. Build:
#
#   docker build -t ml-intern-kit:cu124 .
#   docker run --gpus all -it --rm \
#       -v "$PWD":/workspace -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
#       --env-file .env ml-intern-kit:cu124 bash

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    UV_LINK_MODE=copy \
    PATH="/workspace/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
        python3.11 python3.11-dev python3.11-venv \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official image (matches ml-intern's Dockerfile pattern)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Resolve and install deps first so this layer caches well.
COPY pyproject.toml requirements.txt .python-version ./
RUN uv venv --python 3.11 && uv pip install -r requirements.txt

# Copy the rest of the project (scripts, configs, agent rules)
COPY . .

CMD ["bash"]
