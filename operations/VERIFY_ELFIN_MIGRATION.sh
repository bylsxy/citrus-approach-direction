#!/usr/bin/env bash
#
# Static, no-motion verification for the copied Elfin work.

set -euo pipefail

HOME_ROOT=/home/catas
ROS_WS=$HOME_ROOT/ros_ws
RS_WS=$HOME_ROOT/ros2_ws
failures=0
TOP_LAUNCHERS=(
    DETECT_ELFIN.sh
    RECOVER_ELFIN_COLLISION.sh
    START_CITRUS_VISION.sh
    START_ELFIN_CAMERA_CALIBRATION.sh
    START_ELFIN_DEMO.sh
    START_ELFIN_FREEDRIVE_SIM.sh
    START_ELFIN_HARDWARE.sh
    START_ELFIN_HARVEST.sh
    START_ELFIN_PANEL.sh
    STOP_ELFIN.sh
    STOP_ELFIN_DEMO.sh
    TEST_ELFIN_OFFLINE.sh
    VERIFY_ELFIN_MIGRATION.sh
)

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
for launcher in "${TOP_LAUNCHERS[@]}"; do
    bash -n "$HOME_ROOT/$launcher"
done
while IFS= read -r -d '' script; do
    bash -n "$script"
done < <(find \
    "$ROS_WS/src/elfin_robot" \
    "$ROS_WS/src/elfin_vision" \
    -type f -name '*.sh' -print0)

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
old_path_found=false
while IFS= read -r -d '' active_file; do
    if grep -nI '/home/jetson' "$active_file"; then
        old_path_found=true
    fi
done < <(find \
    "$ROS_WS/src" \
    "$RS_WS/src" \
    "$HOME_ROOT/elfin_citrus_data" \
    -type f \
    -not -path '*/.git/*' \
    -not -path '*/build/*' \
    -not -path '*/devel/*' \
    -not -path '*/install/*' \
    -not -path '*/__pycache__/*' \
    -print0)
for launcher in "${TOP_LAUNCHERS[@]}"; do
    [[ "$launcher" == VERIFY_ELFIN_MIGRATION.sh ]] && continue
    if grep -nI '/home/jetson' "$HOME_ROOT/$launcher"; then
        old_path_found=true
    fi
done
if [[ "$old_path_found" == true ]]; then
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
