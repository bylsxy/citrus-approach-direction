#!/usr/bin/env bash

set -euo pipefail

REMOTE=catas@192.168.201.133
DEST=/var/tmp/catas-robotics/migrated/ros2_ws-full-jetson

ssh -o BatchMode=yes "$REMOTE" \
    "mkdir -p '$DEST'"

rsync -aH \
    --no-owner \
    --no-group \
    --partial \
    --human-readable \
    -e 'ssh -o BatchMode=yes' \
    /home/jetson/ros2_ws/ \
    "$REMOTE:$DEST/"

local_bytes=$(du -sb /home/jetson/ros2_ws | awk '{print $1}')
remote_bytes=$(ssh -o BatchMode=yes "$REMOTE" \
    "du -sb '$DEST' | awk '{print \\$1}'")

if [[ "$local_bytes" != "$remote_bytes" ]]; then
    printf 'Size mismatch: local=%s remote=%s\n' \
        "$local_bytes" "$remote_bytes" >&2
    exit 1
fi

ssh -o BatchMode=yes "$REMOTE" \
    "printf 'ROS2_FULL_TRANSFER_OK\\nbytes=%s\\n' '$remote_bytes' >'/var/tmp/catas-robotics/migrated/ROS2_FULL_TRANSFER_OK.txt'"

printf 'ROS2_FULL_TRANSFER_OK bytes=%s\n' "$remote_bytes"
