#!/usr/bin/env bash
#
# Install only the locally recommended Ubuntu NVIDIA driver branch for the
# detected PCI device.  Default mode is simulation; the script never reboots.

set -euo pipefail

mode=${1:---simulate}
case "$mode" in
    --simulate|--apply) ;;
    *)
        printf 'Usage: %s [--simulate|--apply]\n' "$0" >&2
        exit 64
        ;;
esac

source /etc/os-release
if [[ ${ID:-} != ubuntu || ${VERSION_ID:-} != 18.04 ]]; then
    printf 'Refusing: this reviewed driver branch applies only to the current Ubuntu 18.04 host.\n' >&2
    exit 78
fi
if [[ $(dpkg --print-architecture) != amd64 ]]; then
    printf 'Refusing: expected amd64.\n' >&2
    exit 78
fi
if ! lspci -Dnnd 10de:28e0 | grep -q '10de:28e0'; then
    printf 'Refusing: expected NVIDIA PCI ID 10de:28e0 was not found.\n' >&2
    exit 78
fi
if command -v mokutil >/dev/null 2>&1 \
        && ! mokutil --sb-state 2>&1 | grep -qi 'disabled'; then
    printf 'Refusing: Secure Boot is not confirmed disabled.\n' >&2
    exit 78
fi

kernel_headers="linux-headers-$(uname -r)"
packages=("$kernel_headers" nvidia-driver-530)

printf 'Detected GPU: '
lspci -Dnnd 10de:28e0
printf 'Planned exact packages: %s\n' "${packages[*]}"
printf 'CUDA toolkit files are already present; this script does not install a CUDA runfile.\n'
printf 'No apt upgrade, Docker install, Melodic removal, or automatic reboot is included.\n'

if [[ "$mode" == --simulate ]]; then
    apt-get -s install --no-install-recommends "${packages[@]}"
    printf 'SIMULATION_ONLY\n'
    exit 0
fi

if ((EUID != 0)); then
    printf 'Run the reviewed apply mode locally with:\n' >&2
    printf '  sudo %q --apply\n' "$0" >&2
    exit 77
fi

for lock in /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock; do
    if command -v fuser >/dev/null 2>&1 && fuser "$lock" >/dev/null 2>&1; then
        printf 'Host apt/dpkg lock is busy: %s\n' "$lock" >&2
        exit 75
    fi
done

read -r -p 'Type exactly INSTALL_NVIDIA_530 to continue: ' confirmation
if [[ "$confirmation" != INSTALL_NVIDIA_530 ]]; then
    printf 'Confirmation did not match; nothing was installed.\n' >&2
    exit 77
fi

apt-get update
apt-get install -y --no-install-recommends "${packages[@]}"

printf 'NVIDIA_DRIVER_PACKAGES_INSTALLED_REBOOT_REQUIRED\n'
printf 'This script did not reboot. After a deliberate reboot, verify nvidia-smi before any CUDA/TensorRT build.\n'
