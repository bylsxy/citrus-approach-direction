#!/usr/bin/env bash
#
# Read-only preflight for the migrated Elfin/RealSense host.
# It never runs slaveinfo, roslaunch, a ROS service, or an actuator command.

set -euo pipefail

STRICT=false
if [[ ${1:-} == --strict ]]; then
    STRICT=true
elif (($#)); then
    printf 'Usage: %s [--strict]\n' "$0" >&2
    exit 64
fi

blockers=0

section() {
    printf '\n[%s]\n' "$1"
}

block() {
    printf 'BLOCKED: %s\n' "$1"
    blockers=$((blockers + 1))
}

pass() {
    printf 'PASS: %s\n' "$1"
}

section "identity"
date --iso-8601=seconds
uname -a
if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -a 2>/dev/null || true
fi
printf 'architecture=%s\n' "$(dpkg --print-architecture 2>/dev/null || uname -m)"
id

section "ROS installations"
find /opt/ros -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    2>/dev/null | sort || true
if [[ -r /opt/ros/noetic/setup.bash ]]; then
    pass "native ROS Noetic exists"
else
    block "native ROS Noetic is absent on this boot"
fi

section "GPU and CUDA"
if command -v lspci >/dev/null 2>&1; then
    lspci -nnk | sed -n '/VGA compatible controller/,+4p;/3D controller/,+4p'
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi; then
    pass "NVIDIA kernel driver and userspace client communicate"
else
    block "nvidia-smi cannot communicate with an NVIDIA driver"
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    printf 'nvcc=absent\n'
fi

section "realtime kernel gate"
kernel_config="/boot/config-$(uname -r)"
if [[ -r "$kernel_config" ]]; then
    grep -E '^(CONFIG_PREEMPT|CONFIG_PREEMPT_RT|CONFIG_RT_GROUP_SCHED)=' \
        "$kernel_config" || true
    grep -E '^# (CONFIG_PREEMPT|CONFIG_PREEMPT_RT|CONFIG_RT_GROUP_SCHED) is not set' \
        "$kernel_config" || true
else
    printf 'kernel_config=unreadable:%s\n' "$kernel_config"
fi
for path in \
    /proc/sys/kernel/sched_rt_period_us \
    /proc/sys/kernel/sched_rt_runtime_us; do
    if [[ -r "$path" ]]; then
        printf '%s=%s\n' "${path##*/}" "$(<"$path")"
    else
        printf '%s=unavailable\n' "${path##*/}"
    fi
done
rt_root=/sys/fs/cgroup/cpu,cpuacct
if [[ -r "$rt_root/cpu.rt_runtime_us" && -w "$rt_root/tasks" ]]; then
    pass "cgroup-v1 realtime budget files are available"
else
    block "the current kernel/cgroup does not expose writable cgroup-v1 realtime budget files required by the existing hardware launcher"
fi
printf 'user_rtprio=%s\n' "$(ulimit -r)"
printf 'user_memlock_kib=%s\n' "$(ulimit -l)"

section "network interfaces (no EtherCAT probe)"
if command -v ip >/dev/null 2>&1; then
    ip -br link || true
    ip -br address || true
    ip route || true
fi
safe_candidates=()
for interface_path in /sys/class/net/*; do
    interface=${interface_path##*/}
    [[ "$interface" == lo ]] && continue
    [[ -e "$interface_path/device" ]] || continue
    [[ -d "$interface_path/wireless" ]] && continue
    [[ -r "$interface_path/type" && "$(<"$interface_path/type")" == 1 ]] || continue
    [[ -r "$interface_path/carrier" && "$(<"$interface_path/carrier")" == 1 ]] || continue
    if command -v ip >/dev/null 2>&1; then
        if ip -o -4 address show dev "$interface" scope global | grep -q .; then
            continue
        fi
        if ip -4 route show default dev "$interface" | grep -q .; then
            continue
        fi
    fi
    safe_candidates+=("$interface")
done
if ((${#safe_candidates[@]} == 1)); then
    pass "one linked, non-routed Ethernet candidate exists: ${safe_candidates[0]}"
elif ((${#safe_candidates[@]} == 0)); then
    block "no linked, non-routed dedicated Ethernet interface exists; no raw EtherCAT probe was attempted"
else
    block "multiple non-routed Ethernet candidates require explicit physical identification: ${safe_candidates[*]}"
fi

section "USB and udev (enumeration only)"
if command -v lsusb >/dev/null 2>&1; then
    lsusb || true
fi
find /etc/udev/rules.d /lib/udev/rules.d -maxdepth 1 -type f \
    \( -iname '*realsense*' -o -iname '*elfin*' -o -iname '*ethercat*' \) \
    -printf '%p\n' 2>/dev/null | sort || true

section "container runtime"
if command -v docker >/dev/null 2>&1; then
    docker --version || true
else
    block "Docker is absent"
fi
if command -v podman >/dev/null 2>&1; then
    podman --version || true
fi

section "Ling archive disk"
if command -v lsblk >/dev/null 2>&1; then
    lsblk -o NAME,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINT,MODEL,TRAN || true
fi
ling_target=$(
    findmnt -rn -S UUID=D09A-E492 -o TARGET 2>/dev/null | head -n 1 || true
)
if [[ -n "$ling_target" ]]; then
    pass "Ling is mounted at $ling_target"
else
    block "Ling UUID D09A-E492 is not mounted"
fi

section "migration completion markers"
find /var/tmp/catas-robotics/noetic/markers -maxdepth 1 -type f \
    -printf '%f\n' 2>/dev/null | sort || true
if [[ -r /var/tmp/catas-robotics/migrated/ROS2_FULL_TRANSFER_OK.txt ]]; then
    sed -n '1,20p' \
        /var/tmp/catas-robotics/migrated/ROS2_FULL_TRANSFER_OK.txt
else
    block "the full ros2_ws transfer marker is absent"
fi

printf '\nNO_MOTION_HOST_AUDIT blockers=%d\n' "$blockers"
printf 'No slaveinfo, roslaunch, ROS service, EtherCAT socket, camera stream, or actuator command was used.\n'

if [[ "$STRICT" == true && $blockers -ne 0 ]]; then
    exit 1
fi
