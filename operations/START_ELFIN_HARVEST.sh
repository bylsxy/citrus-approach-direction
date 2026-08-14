#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -r /opt/ros/noetic/setup.bash ]]; then
    CHROOT_WRAPPER="$SCRIPT_DIR/migration_from_jetson/real_runtime/run_noetic_userns_chroot.sh"
    if [[ ! -x "$CHROOT_WRAPPER" ]]; then
        echo "Noetic GUI wrapper is missing: $CHROOT_WRAPPER" >&2
        exit 69
    fi
    exec "$CHROOT_WRAPPER" "$0" "$@"
fi

exec "${HOME}/ros_ws/src/elfin_vision/scripts/start_elfin_harvest.sh" "$@"
