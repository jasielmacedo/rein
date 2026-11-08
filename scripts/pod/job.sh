#!/usr/bin/env bash
# Runs one paid job end to end and stops the Pod afterwards, on success and on failure.
# Usage: scripts/pod/job.sh <job-name> <command...>
set -euo pipefail

JOB_NAME="$1"
shift
WORKSPACE="${WORKSPACE:-/workspace}"
LOG_DIR="$WORKSPACE/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$JOB_NAME.log"

export HF_HOME="$WORKSPACE/hf-cache"
export UV_CACHE_DIR="$WORKSPACE/uv-cache"
export UV_PROJECT_ENVIRONMENT="$WORKSPACE/venv"
export PATH="$HOME/.local/bin:$PATH"

stop_pod() {
  local status=$?
  echo "job $JOB_NAME finished with status $status" | tee -a "$LOG_FILE"
  echo "$status" > "$LOG_DIR/$JOB_NAME.status"
  if [[ "${KEEP_POD:-0}" != "1" && -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    runpodctl stop pod "$RUNPOD_POD_ID" || echo "could not stop the Pod from inside; stop it from outside"
  fi
}
trap stop_pod EXIT

cd "$WORKSPACE/rein"
echo "job $JOB_NAME started $(date -u +%FT%TZ)" | tee "$LOG_FILE"
uv run pytest -q tests/test_gpu_sync.py tests/test_acceptance.py 2>&1 | tee -a "$LOG_FILE"
"$@" 2>&1 | tee -a "$LOG_FILE"
