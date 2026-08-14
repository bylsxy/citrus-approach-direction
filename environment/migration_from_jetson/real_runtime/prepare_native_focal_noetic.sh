#!/usr/bin/env bash
#
# Prepare a separate native Ubuntu 20.04 installation for Noetic.
# It intentionally refuses to run on the current Ubuntu 18.04 installation.
# Default mode is audit-only.

set -euo pipefail

mode=${1:---audit}
case "$mode" in
    --audit|--apply) ;;
    *)
        printf 'Usage: %s [--audit|--apply]\n' "$0" >&2
        exit 64
        ;;
esac

source /etc/os-release
if [[ ${ID:-} != ubuntu || ${VERSION_ID:-} != 20.04 ]]; then
    printf 'Refusing: boot the dedicated Ubuntu 20.04 system first. Current system is %s %s.\n' \
        "${ID:-unknown}" "${VERSION_ID:-unknown}" >&2
    exit 78
fi
if [[ $(dpkg --print-architecture) != amd64 ]]; then
    printf 'Refusing: expected amd64.\n' >&2
    exit 78
fi

packages=(
    build-essential
    cmake
    ethtool
    fonts-noto-cjk
    git
    iproute2
    pkg-config
    python3-catkin-tools
    python3-numpy
    python3-opencv
    python3-pip
    python3-rosdep
    python3-rospkg
    python3-scipy
    python3-yaml
    python3-wxgtk4.0
    libcanberra-gtk-module
    libcanberra-gtk3-module
    locales
    ros-noetic-desktop-full
    ros-noetic-moveit
    ros-noetic-moveit-servo
    ros-noetic-realsense2-camera
    ros-noetic-soem
    rt-tests
    usbutils
)

printf 'Native Focal package plan (%d exact package names):\n' "${#packages[@]}"
printf '  %s\n' "${packages[@]}"
printf 'No apt upgrade, Melodic removal, NVIDIA runfile, bootloader edit, or robot launch is included.\n'

kernel_config="/boot/config-$(uname -r)"
if [[ -r "$kernel_config" ]]; then
    grep -E '^(CONFIG_PREEMPT|CONFIG_PREEMPT_RT|CONFIG_RT_GROUP_SCHED)=' \
        "$kernel_config" || true
    grep -E '^# (CONFIG_PREEMPT|CONFIG_PREEMPT_RT|CONFIG_RT_GROUP_SCHED) is not set' \
        "$kernel_config" || true
fi

if [[ "$mode" == --audit ]]; then
    if [[ -r /opt/ros/noetic/setup.bash ]]; then
        printf 'ROS_NOETIC_ALREADY_PRESENT\n'
    else
        printf 'AUDIT_ONLY_ROS_NOETIC_ABSENT\n'
    fi
    exit 0
fi

if ((EUID != 0)); then
    printf 'Run the reviewed apply mode locally with:\n' >&2
    printf '  sudo %q --apply\n' "$0" >&2
    exit 77
fi

read -r -p 'Type exactly PREPARE_NATIVE_FOCAL_NOETIC to continue: ' confirmation
if [[ "$confirmation" != PREPARE_NATIVE_FOCAL_NOETIC ]]; then
    printf 'Confirmation did not match; nothing was installed.\n' >&2
    exit 77
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg2

keyring=/usr/share/keyrings/ros-archive-keyring.gpg
temporary_key=$(mktemp)
trap 'rm -f "$temporary_key"' EXIT
curl -fL --retry 4 --retry-delay 2 \
    -o "$temporary_key" \
    https://raw.githubusercontent.com/ros/rosdistro/master/ros.key
gpg --dearmor --yes --output "$keyring" "$temporary_key"
printf '%s\n' \
    'deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main' \
    >/etc/apt/sources.list.d/ros1.list

apt-get update
apt-get install -y --no-install-recommends "${packages[@]}"
locale-gen en_US.UTF-8 zh_CN.UTF-8
rosdep init 2>/dev/null || \
    test -f /etc/ros/rosdep/sources.list.d/20-default.list

printf 'NATIVE_FOCAL_NOETIC_PACKAGES_INSTALLED\n'
printf 'No robot node was started. Run rosdep update as the normal desktop user, then build and complete timing/NIC gates.\n'
