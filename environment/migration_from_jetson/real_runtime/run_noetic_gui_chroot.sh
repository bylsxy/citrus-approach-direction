#!/usr/bin/env bash

# Run a local Noetic GUI/ROS command in the Focal rootfs without PRoot's
# ptrace emulation. The private mount namespace disappears with the command.

set -euo pipefail

ROOTFS=/var/tmp/catas-robotics/noetic/rootfs
DISPLAY_NUMBER=${ELFIN_DISPLAY:-:0}
XAUTHORITY_FILE=/home/catas/.Xauthority
X11_SOCKET_DIR=/tmp/.X11-unix

if (( EUID != 0 )); then
    echo "Run this wrapper through the system authorization dialog." >&2
    exit 77
fi
if (( $# == 0 )); then
    echo "Usage: run_noetic_gui_chroot.sh COMMAND [ARG ...]" >&2
    exit 64
fi
if [[ ! "$DISPLAY_NUMBER" =~ ^:[0-9]+$ ]]; then
    echo "Invalid local X11 display: $DISPLAY_NUMBER" >&2
    exit 64
fi
x11_socket="$X11_SOCKET_DIR/X${DISPLAY_NUMBER#:}"
if [[ ! -S "$x11_socket" || ! -r "$XAUTHORITY_FILE" ]]; then
    echo "The local X11 desktop or its authority cookie is unavailable." >&2
    exit 69
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

install -d -m 1777 "$ROOTFS/tmp" "$ROOTFS$X11_SOCKET_DIR"

exec unshare --mount --propagation private /bin/bash -c '
    set -euo pipefail
    rootfs=$1
    display_number=$2
    xauthority_file=$3
    x11_socket_dir=$4
    shift 4

    for path in /dev /proc /sys /run /home/catas; do
        mount --rbind "$path" "$rootfs$path"
        mount --make-rslave "$rootfs$path"
    done
    mount --bind "$x11_socket_dir" "$rootfs$x11_socket_dir"
    mount --make-slave "$rootfs$x11_socket_dir"
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
        ROS_MASTER_URI=http://127.0.0.1:11311 \
        ROS_IP=127.0.0.1 \
        DISPLAY="$display_number" \
        XAUTHORITY="$xauthority_file" \
        XDG_RUNTIME_DIR=/run/user/1000 \
        DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
        NO_AT_BRIDGE=1 \
        GTK_MODULES= \
        GDK_BACKEND=x11 \
        XLIB_SKIP_ARGB_VISUALS=1 \
        QT_X11_NO_MITSHM=1 \
        PATH=/opt/ros/noetic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        PYTHONNOUSERSITE=1 \
        "$@"
' noetic-gui-chroot "$ROOTFS" "$DISPLAY_NUMBER" "$XAUTHORITY_FILE" \
    "$X11_SOCKET_DIR" "$@"
