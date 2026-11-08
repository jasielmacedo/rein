#!/usr/bin/env bash
# Copies a finished run (checkpoint, metrics, reports) from the Modal volume to runs/checkpoints/<run>.
# Usage: scripts/modal/pull_run.sh rein-0.8b-r1-L14
set -euo pipefail
RUN="$1"
export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p runs/checkpoints
uv run modal volume get --force rein-data "/runs/$RUN" runs/checkpoints/
du -sh "runs/checkpoints/$RUN"
