#!/usr/bin/env bash

set -euo pipefail

HANDOFF_ROOT=/home/catas/migration_from_jetson/handoff
SESSION_ID=019fb082-00d4-7c50-99a6-0cf7d5287475
LOG=$HANDOFF_ROOT/remote-codex-events.jsonl
LAST=$HANDOFF_ROOT/remote-codex-last-message.txt

mkdir -p "$HANDOFF_ROOT"
exec /home/catas/.local/bin/codex exec resume \
    --skip-git-repo-check \
    --json \
    -o "$LAST" \
    "$SESSION_ID" - \
    <"$HANDOFF_ROOT/REMOTE_CODEX_PROMPT.txt" \
    >>"$LOG" 2>&1
