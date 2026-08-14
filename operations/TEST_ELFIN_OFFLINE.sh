#!/usr/bin/env bash
set -euo pipefail

exec "${HOME}/ros_ws/src/elfin_robot/elfin_robot_bringup/script/elfin_offline_smoke_test.sh" "$@"
