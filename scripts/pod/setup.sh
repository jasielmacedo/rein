#!/usr/bin/env bash
# One-time environment setup on a RunPod Pod. Everything heavy lives on the /workspace volume.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/rein"
FLA_VERSION="${FLA_VERSION:-0.6.0}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.7.0}"

export HF_HOME="$WORKSPACE/hf-cache"
export UV_CACHE_DIR="$WORKSPACE/uv-cache"
export UV_PROJECT_ENVIRONMENT="$WORKSPACE/venv"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
uv sync --extra train --frozen
uv pip install "flash-linear-attention==$FLA_VERSION"
uv pip install --only-binary=:all: "causal-conv1d==$CAUSAL_CONV1D_VERSION" \
  || echo "causal-conv1d has no prebuilt wheel here; the torch conv fallback is used"

uv run python -c "import torch, fla; assert torch.cuda.is_available(); print('cuda', torch.cuda.get_device_name(0), 'fla', fla.__version__)"
