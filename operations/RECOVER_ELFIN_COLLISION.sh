#!/usr/bin/env bash
set -euo pipefail

exec "${HOME}/ros_ws/src/elfin_robot/elfin_robot_bringup/script/elfin_collision_recovery.sh" "$@"
