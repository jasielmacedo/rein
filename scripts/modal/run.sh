#!/usr/bin/env bash
# Runs the Modal CLI with UTF-8 output; the Windows console codepage crashes it otherwise.
# Usage: scripts/modal/run.sh run modal_apps/train_app.py --action kernels
set -euo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
exec uv run modal "$@"
