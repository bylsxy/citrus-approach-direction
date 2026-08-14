#!/usr/bin/env bash
set -euo pipefail

exec sudo "${HOME}/ros_ws/src/elfin_robot/elfin_robot_bringup/script/detect_elfin_ethercat_interface.sh" "$@"
