#!/usr/bin/env bash

# Run one privileged ROS Noetic command in the existing Focal rootfs while
# retaining the host network/PID namespaces. A private mount namespace keeps
# all bind mounts temporary and makes raw EtherCAT access possible without
# installing Noetic into the Bionic host.

set -euo pipefail

ROOTFS=/var/tmp/catas-robotics/noetic/rootfs

if (( EUID != 0 )); then
    echo "Run this wrapper through the system authorization dialog." >&2
    exit 77
fi
if (( $# == 0 )); then
    echo "Usage: run_noetic_privileged_chroot.sh COMMAND [ARG ...]" >&2
    exit 64
fi
for path in /dev /proc /sys /run /home/catas; do
    if [[ ! -e "$path" || ! -e "$ROOTFS$path" ]]; then
        echo "Required host/rootfs path is missing: $path" >&2
        exit 69
    fi
done
for command_name in chroot mount unshare; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required host command is missing: $command_name" >&2
        exit 69
    fi
done

exec unshare --mount --propagation private /bin/bash -c '
    set -euo pipefail
    rootfs=$1
    shift

    for path in /dev /proc /sys /run /home/catas; do
        mount --rbind "$path" "$rootfs$path"
        mount --make-rslave "$rootfs$path"
    done
    mount --bind /etc/hosts "$rootfs/etc/hosts"
    mount --bind /etc/resolv.conf "$rootfs/etc/resolv.conf"

    exec chroot "$rootfs" /usr/bin/env -i \
        HOME=/home/catas \
        USER=root \
        LOGNAME=root \
        SHELL=/bin/bash \
        LANG=zh_CN.UTF-8 \
        LC_ALL=zh_CN.UTF-8 \
        CATAS_NOETIC_ENV=chroot \
        ROS_DISTRO=noetic \
        ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}" \
        ROS_IP="${ROS_IP:-127.0.0.1}" \
        DISPLAY="${DISPLAY:-:0}" \
        XAUTHORITY=/home/catas/.Xauthority \
        QT_X11_NO_MITSHM=1 \
        PATH=/opt/ros/noetic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        PYTHONNOUSERSITE=1 \
        "$@"
' noetic-chroot "$ROOTFS" "$@"
