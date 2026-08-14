#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -r /opt/ros/noetic/setup.bash ]]; then
    CHROOT_WRAPPER="$SCRIPT_DIR/migration_from_jetson/real_runtime/run_noetic_userns_chroot.sh"
    for argument in "$@"; do
        if [[ "$argument" == --replace-existing ]]; then
            CHROOT_WRAPPER="$SCRIPT_DIR/migration_from_jetson/real_runtime/run_noetic_privileged_chroot.sh"
            break
        fi
    done
    if [[ ! -x "$CHROOT_WRAPPER" ]]; then
        echo "Noetic GUI wrapper is missing: $CHROOT_WRAPPER" >&2
        exit 69
    fi
    if [[ "$CHROOT_WRAPPER" == *run_noetic_privileged_chroot.sh ]]; then
        exec sudo -E "$CHROOT_WRAPPER" "$0" "$@"
    fi
    exec "$CHROOT_WRAPPER" "$0" "$@"
fi

exec "$SCRIPT_DIR/ros_ws/src/elfin_robot/elfin_robot_bringup/script/start_elfin5_panel.sh" "$@"
