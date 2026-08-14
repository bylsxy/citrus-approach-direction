#!/usr/bin/env bash
#
# Capture the Jetson-side software and source state needed to rebuild the
# migrated robotics work on an x86_64 machine.  This script is read-only apart
# from files below migration_to_catas/manifests.

set -euo pipefail

OUT_DIR=/home/jetson/migration_to_catas/manifests
TRACK_CONDA=/home/jetson/anaconda3/bin/conda
TRACK_PYTHON=/home/jetson/anaconda3/envs/track/bin/python

mkdir -p "$OUT_DIR"

{
    date --iso-8601=seconds
    uname -a
    lsb_release -a 2>/dev/null || true
    printf '\nROS distributions:\n'
    find /opt/ros -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort
    printf '\nCUDA compiler:\n'
    nvcc --version 2>/dev/null || true
    printf '\nGPU runtime:\n'
    nvidia-smi 2>&1 || true
} >"$OUT_DIR/jetson-platform.txt"

dpkg-query -W -f='${binary:Package}\t${Version}\t${Architecture}\n' \
    >"$OUT_DIR/jetson-dpkg-packages.tsv"
apt-mark showmanual >"$OUT_DIR/jetson-apt-manual.txt"

"$TRACK_CONDA" list -n track --explicit >"$OUT_DIR/track-conda-explicit-linux-aarch64.txt"
"$TRACK_CONDA" env export -n track \
    --file "$OUT_DIR/track-conda-environment-linux-aarch64.yml"
"$TRACK_CONDA" list -n track --json >"$OUT_DIR/track-conda-list.json"
"$TRACK_PYTHON" -m pip freeze >"$OUT_DIR/track-pip-freeze.txt"

for repo in \
    /home/jetson/ros_ws/src/elfin_robot \
    /home/jetson/ros_ws/src/elfin_vision \
    /home/jetson/ros2_ws/src/realsense-ros; do
    name=$(basename "$repo")
    {
        printf 'path=%s\n' "$repo"
        git -C "$repo" rev-parse HEAD
        git -C "$repo" status --short
        git -C "$repo" remote -v
        git -C "$repo" submodule status --recursive 2>/dev/null || true
    } >"$OUT_DIR/git-${name}.txt"
done

{
    printf 'Jetson source SHA-256 manifest generated at '
    date --iso-8601=seconds
    find \
        /home/jetson/ros_ws/src \
        /home/jetson/ros2_ws/src/realsense-ros \
        /home/jetson/elfin_citrus_data \
        -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum
} >"$OUT_DIR/robotics-active-files.sha256"

{
    printf 'Sensitive files are intentionally listed without their contents.\n'
    stat -c '%a\t%U:%G\t%s\t%n' \
        /home/jetson/.config/elfin_vision/remote_inference.token \
        /home/jetson/ros_ws/src/elfin_vision/config/camera_to_robot.yaml \
        /home/jetson/.ros/elfin_freedrive_points.yaml \
        /home/jetson/.ros/elfin_freedrive_payload.yaml \
        2>/dev/null || true
} >"$OUT_DIR/robotics-sensitive-file-metadata.txt"

printf 'Manifests written to %s\n' "$OUT_DIR"
