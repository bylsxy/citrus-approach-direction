#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -r /opt/ros/noetic/setup.bash ]]; then
    CHROOT_WRAPPER="$SCRIPT_DIR/migration_from_jetson/real_runtime/run_noetic_privileged_chroot.sh"
    if [[ ! -x "$CHROOT_WRAPPER" ]]; then
        echo "Noetic hardware wrapper is missing: $CHROOT_WRAPPER" >&2
        exit 69
    fi
    exec sudo -E "$CHROOT_WRAPPER" "$0" "$@"
fi

exec "$SCRIPT_DIR/ros_ws/src/elfin_robot/elfin_robot_bringup/script/start_elfin5_hardware.sh" "$@"
