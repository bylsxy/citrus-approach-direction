#!/usr/bin/env bash
#
# Exact, minimal host package action needed to mount the existing Ling exFAT
# filesystem on Ubuntu 18.04.  Default mode is simulation.

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
    printf 'Refusing: this reviewed package set is only for Ubuntu 18.04.\n' >&2
    exit 78
fi
if [[ $(dpkg --print-architecture) != amd64 ]]; then
    printf 'Refusing: expected amd64.\n' >&2
    exit 78
fi

packages=(exfat-fuse exfat-utils)

printf 'Planned exact packages: %s\n' "${packages[*]}"
printf 'No apt upgrade, filesystem format, fsck repair, or source deletion is included.\n'

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

read -r -p 'Type exactly INSTALL_EXFAT_SUPPORT to continue: ' confirmation
if [[ "$confirmation" != INSTALL_EXFAT_SUPPORT ]]; then
    printf 'Confirmation did not match; nothing was installed.\n' >&2
    exit 77
fi

apt-get update
apt-get install -y --no-install-recommends "${packages[@]}"

printf 'EXFAT_SUPPORT_INSTALLED\n'
printf 'Next non-destructive command: udisksctl mount -b /dev/sda1\n'
