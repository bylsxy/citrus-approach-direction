#!/usr/bin/env bash

# Enter the Noetic rootfs natively through an unprivileged user/mount namespace.
# This is suitable for ROS clients and CPU vision workers. It intentionally
# provides no host-root, realtime, cgroup or raw-EtherCAT authority.

set -euo pipefail

ROOTFS=/var/tmp/catas-robotics/noetic/rootfs

if (( $# == 0 )); then
    echo "Usage: run_noetic_userns_chroot.sh COMMAND [ARG ...]" >&2
    exit 64
fi
if [[ ! -x "$ROOTFS/bin/bash" || ! -d "$ROOTFS/home/catas" ]]; then
    echo "Noetic rootfs is unavailable: $ROOTFS" >&2
    exit 69
fi
for command_name in chroot mount unshare; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Required host command is missing: $command_name" >&2
        exit 69
    }
done

exec unshare --user --map-root-user --mount --propagation private \
    /bin/bash -c '
        set -euo pipefail
        rootfs=$1
        shift
        mount --rbind /dev "$rootfs/dev"
        mount --make-rslave "$rootfs/dev"
        mount --rbind /proc "$rootfs/proc"
        mount --make-rslave "$rootfs/proc"
        # librealsense needs the live sysfs USB topology as well as /dev.
        # Without this bind, a camera plugged in after the visual terminal
        # starts creates video/USB device nodes but rs-enumerate-devices still
        # reports "No device detected".
        mount --rbind /sys "$rootfs/sys"
        mount --make-rslave "$rootfs/sys"
        mount --bind /home/catas "$rootfs/home/catas"
        if [[ -d /tmp/.X11-unix && -d "$rootfs/tmp/.X11-unix" ]]; then
            mount --bind /tmp/.X11-unix "$rootfs/tmp/.X11-unix"
        fi
        mount --bind /etc/hosts "$rootfs/etc/hosts"
        mount --bind /etc/resolv.conf "$rootfs/etc/resolv.conf"
        exec chroot "$rootfs" /usr/bin/env -i \
            HOME=/home/catas \
            USER=catas \
            LOGNAME=catas \
            SHELL=/bin/bash \
            LANG=zh_CN.UTF-8 \
            LC_ALL=zh_CN.UTF-8 \
            CATAS_NOETIC_ENV=userns-chroot \
            ROS_DISTRO=noetic \
            ROS_MASTER_URI=http://127.0.0.1:11311 \
            ROS_IP=127.0.0.1 \
            DISPLAY="${DISPLAY:-:0}" \
            XAUTHORITY=/home/catas/.Xauthority \
            QT_X11_NO_MITSHM=1 \
            PATH=/opt/ros/noetic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
            PYTHONNOUSERSITE=1 \
            "$@"
    ' noetic-userns-chroot "$ROOTFS" "$@"
