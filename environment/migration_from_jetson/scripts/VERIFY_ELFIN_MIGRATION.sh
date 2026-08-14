#!/usr/bin/env bash
#
# Static, no-motion verification for the copied Elfin work.

set -euo pipefail

HOME_ROOT=/home/catas
ROS_WS=$HOME_ROOT/ros_ws
RS_WS=$HOME_ROOT/ros2_ws
failures=0

check_path() {
    local path=$1
    if [[ -e "$path" || -L "$path" ]]; then
        printf 'PASS path: %s\n' "$path"
    else
        printf 'FAIL missing: %s\n' "$path" >&2
        failures=$((failures + 1))
    fi
}

printf '[1/5] Required migrated files\n'
for path in \
    "$ROS_WS/src/elfin_robot" \
    "$ROS_WS/src/elfin_vision" \
    "$RS_WS/src/realsense-ros" \
    "$HOME_ROOT/elfin_citrus_data" \
    "$HOME_ROOT/.config/elfin_vision/remote_inference.token" \
    "$ROS_WS/src/elfin_vision/config/camera_to_robot.yaml" \
    "$HOME_ROOT/START_ELFIN_HARDWARE.sh" \
    "$HOME_ROOT/START_ELFIN_PANEL.sh" \
    "$HOME_ROOT/START_CITRUS_VISION.sh" \
    "$HOME_ROOT/STOP_ELFIN.sh"; do
    check_path "$path"
done

printf '[2/5] Shell syntax (does not execute any launcher)\n'
while IFS= read -r -d '' script; do
    bash -n "$script"
done < <(
    find \
        "$HOME_ROOT" -maxdepth 1 -type f -name '*.sh' -print0
    find \
        "$ROS_WS/src/elfin_robot" \
        "$ROS_WS/src/elfin_vision" \
        -type f -name '*.sh' -print0
)

printf '[3/5] Python syntax (cache stays in /tmp)\n'
export PYTHONPYCACHEPREFIX=/tmp/catas-elfin-migration-pycache
while IFS= read -r -d '' source; do
    /usr/bin/python3 -m py_compile "$source"
done < <(
    find \
        "$ROS_WS/src/elfin_robot" \
        "$ROS_WS/src/elfin_vision" \
        -type f -name '*.py' -print0
)

printf '[4/5] No old machine path remains in active source\n'
if grep -RInI \
    --exclude-dir=.git \
    --exclude-dir=build \
    --exclude-dir=devel \
    --exclude-dir=install \
    --exclude-dir=__pycache__ \
    '/home/jetson' \
    "$ROS_WS/src" "$RS_WS/src" "$HOME_ROOT"/*.sh; then
    printf 'FAIL old /home/jetson paths remain above\n' >&2
    failures=$((failures + 1))
else
    printf 'PASS path adaptation\n'
fi

printf '[5/5] ROS Noetic readiness gate\n'
if [[ -r /opt/ros/noetic/setup.bash ]]; then
    printf 'PASS ROS Noetic setup exists\n'
else
    printf 'BLOCKED ROS Noetic is not installed on this host.\n' >&2
    printf 'Do not start hardware; use the planned Ubuntu 20.04/Noetic environment.\n' >&2
    failures=$((failures + 1))
fi

if ((failures)); then
    printf 'STATIC_CHECK_INCOMPLETE failures=%d (no hardware command was sent)\n' "$failures" >&2
    exit 1
fi

printf 'STATIC_CHECK_OK (no hardware command was sent)\n'
