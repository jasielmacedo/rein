#!/usr/bin/env bash
# Copies the committed code (and optional data files) to the Pod over SSH. Run from Git Bash on the laptop.
# Usage: scripts/pod/sync.sh <user@host> <port> [data files...]
set -euo pipefail

TARGET="$1"
PORT="$2"
shift 2
KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/rein}"
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new)

git archive --format=tar HEAD | "${SSH[@]}" "$TARGET" "mkdir -p $REMOTE_DIR && tar -x -C $REMOTE_DIR"
for file in "$@"; do
  "${SSH[@]}" "$TARGET" "mkdir -p $REMOTE_DIR/$(dirname "$file")"
  scp -i "$KEY" -P "$PORT" "$file" "$TARGET:$REMOTE_DIR/$file"
done
echo "synced $(git rev-parse --short HEAD) to $TARGET:$REMOTE_DIR"
