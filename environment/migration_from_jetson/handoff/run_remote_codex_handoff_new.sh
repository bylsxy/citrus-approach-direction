#!/usr/bin/env bash

set -euo pipefail

HANDOFF_ROOT=/home/catas/migration_from_jetson/handoff
LOG=$HANDOFF_ROOT/remote-codex-new-events.jsonl
LAST=$HANDOFF_ROOT/remote-codex-new-last-message.txt

mkdir -p "$HANDOFF_ROOT"
exec /home/catas/.local/bin/codex exec \
    --skip-git-repo-check \
    --json \
    -o "$LAST" \
    - \
    <"$HANDOFF_ROOT/REMOTE_CODEX_PROMPT.txt" \
    >>"$LOG" 2>&1
