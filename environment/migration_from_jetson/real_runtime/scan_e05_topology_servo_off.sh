#!/usr/bin/env bash

# Identify the E05 EtherCAT chain from the dedicated link without starting the
# ROS hardware driver or requesting Servo On.

set -euo pipefail

INTERFACE=${1:-enx207bd51a34e1}
ROOTFS=/var/tmp/catas-robotics/noetic/rootfs
SLAVEINFO=/opt/ros/noetic/bin/slaveinfo
LOCK_FILE=/run/lock/elfin5-hardware.lock
PID_FILE=/run/elfin5-hardware.pid
EVIDENCE_DIR=/home/catas/elfin_evidence
EXPECTED_INTERFACE=enx207bd51a34e1
EXPECTED_HANS_ID='Man: 0000001a ID: 50440200 Rev: 05132016'
EXPECTED_IO_ID='Man: 00201911 ID: 10003201 Rev: 00000001'

if (( EUID != 0 )); then
    echo "This topology scan must run as root so SOEM can open a raw socket." >&2
    exit 77
fi

if [[ "$INTERFACE" != "$EXPECTED_INTERFACE" ]]; then
    echo "Refusing unexpected interface: $INTERFACE" >&2
    exit 64
fi

for command_name in chroot flock ip timeout; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command is missing: $command_name" >&2
        exit 69
    fi
done

if [[ ! -x "$ROOTFS$SLAVEINFO" ]]; then
    echo "Noetic slaveinfo is missing: $ROOTFS$SLAVEINFO" >&2
    exit 69
fi

if [[ ! -e "/sys/class/net/$INTERFACE/device" ||
      ! -r "/sys/class/net/$INTERFACE/type" ||
      "$(<"/sys/class/net/$INTERFACE/type")" != 1 ]]; then
    echo "$INTERFACE is not a physical Ethernet interface." >&2
    exit 69
fi
if [[ ! -r "/sys/class/net/$INTERFACE/carrier" ||
      "$(<"/sys/class/net/$INTERFACE/carrier")" != 1 ]]; then
    echo "$INTERFACE has no carrier; check robot power and the dedicated cable." >&2
    exit 69
fi
if ip -o -4 address show dev "$INTERFACE" scope global | grep -q .; then
    echo "Refusing $INTERFACE because it carries a global IPv4 address." >&2
    exit 69
fi
if ip -4 route show default dev "$INTERFACE" | grep -q .; then
    echo "Refusing $INTERFACE because it carries a default route." >&2
    exit 69
fi

exec 8>"$LOCK_FILE"
if ! flock -n 8; then
    echo "The Elfin hardware lock is already held; refusing a second SOEM master." >&2
    exit 75
fi
if [[ -r "$PID_FILE" ]]; then
    driver_pid=$(<"$PID_FILE")
    if [[ "$driver_pid" =~ ^[0-9]+$ ]] && kill -0 "$driver_pid" 2>/dev/null; then
        echo "The Elfin hardware launch is already active as PID $driver_pid." >&2
        exit 75
    fi
fi

echo "Interface: $INTERFACE"
echo "Carrier: up"
echo "IPv4/default route: none"
echo "ROS hardware driver: not started"
echo "Servo On request: not sent"
echo "Scanning EtherCAT identities for at most 12 seconds..."

set +e
output=$(timeout --signal=INT --kill-after=2s 12s \
    chroot "$ROOTFS" "$SLAVEINFO" "$INTERFACE" 2>&1)
scan_status=$?
set -e
printf '%s\n' "$output"

install -d -o catas -g catas -m 0755 "$EVIDENCE_DIR"
stamp=$(date +%Y%m%d-%H%M%S)
log_file="$EVIDENCE_DIR/e05-topology-$stamp.log"
{
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'interface=%s\n' "$INTERFACE"
    printf 'scan_status=%s\n' "$scan_status"
    printf '%s\n' "$output"
} >"$log_file"
chown catas:catas "$log_file"
chmod 0644 "$log_file"

configured_count=$(sed -n \
    's/^\([0-9][0-9]*\) slaves found and configured\.$/\1/p' <<<"$output")
slave_count=$(grep -c '^Slave:' <<<"$output" || true)
hans_names=$(grep -c '^ Name:Hans Robot$' <<<"$output" || true)
hans_ids=$(grep -c "^ $EXPECTED_HANS_ID$" <<<"$output" || true)
io_names=$(grep -c '^ Name:F2838x CPU1 EtherCAT Slave$' <<<"$output" || true)
io_ids=$(grep -c "^ $EXPECTED_IO_ID$" <<<"$output" || true)

if [[ "$scan_status" != 0 || "$configured_count" != 4 ||
      "$slave_count" != 4 || "$hans_names" != 3 || "$hans_ids" != 3 ||
      "$io_names" != 1 || "$io_ids" != 1 ]]; then
    echo "E05_TOPOLOGY_REJECTED" >&2
    echo "Evidence: $log_file" >&2
    exit 69
fi

echo "E05_TOPOLOGY_OK interface=$INTERFACE slaves=4 hans=3 io=1"
echo "Evidence: $log_file"
