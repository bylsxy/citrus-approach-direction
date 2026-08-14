#!/usr/bin/env bash

# Bound and record a 1 kHz RR10 latency test while hackbench creates normal
# scheduler load. This does not access ROS, EtherCAT, cameras, or motors.

set -euo pipefail

RT_CPU=${ELFIN_RT_CPU:-14}
DURATION_SECONDS=${ELFIN_RT_AUDIT_SECONDS:-60}
MAX_LATENCY_US=${ELFIN_RT_MAX_LATENCY_US:-100}
EVIDENCE_DIR=/home/catas/elfin_evidence

if (( EUID != 0 )); then
    echo "Run this audit through the system authorization dialog." >&2
    exit 77
fi
for command_name in cyclictest hackbench taskset timeout; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command is missing: $command_name" >&2
        exit 69
    fi
done
if ! [[ "$RT_CPU" =~ ^[0-9]+$ && "$DURATION_SECONDS" =~ ^[0-9]+$ &&
        "$MAX_LATENCY_US" =~ ^[0-9]+$ ]]; then
    echo "Realtime audit settings must be non-negative integers." >&2
    exit 64
fi

install -d -o catas -g catas -m 0755 "$EVIDENCE_DIR"
stamp=$(date +%Y%m%d-%H%M%S)
log_file="$EVIDENCE_DIR/realtime-audit-$stamp.log"
load_pid=""

cleanup() {
    if [[ -n "$load_pid" ]] && kill -0 "$load_pid" 2>/dev/null; then
        kill -TERM "$load_pid" 2>/dev/null || true
        wait "$load_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

(
    while true; do
        hackbench --threads --groups 2 --loops 1000 >/dev/null 2>&1 || exit
    done
) &
load_pid=$!

echo "Running ${DURATION_SECONDS}s latency audit on CPU $RT_CPU with hackbench load..."
set +e
output=$(timeout --signal=TERM --kill-after=1s "$((DURATION_SECONDS + 5))s" \
    cyclictest --policy=rr --priority=10 --mlockall --nanosleep \
    --interval=1000 --affinity="$RT_CPU" --threads=1 \
    --duration="${DURATION_SECONDS}s" --quiet 2>&1)
test_status=$?
set -e
cleanup
load_pid=""
printf '%s\n' "$output"

{
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'kernel=%s\n' "$(uname -r)"
    printf 'cpu=%s\n' "$RT_CPU"
    printf 'duration_seconds=%s\n' "$DURATION_SECONDS"
    printf 'test_status=%s\n' "$test_status"
    printf '%s\n' "$output"
} >"$log_file"
chown catas:catas "$log_file"
chmod 0644 "$log_file"

maximum=$(sed -n 's/.*Max:[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
    <<<"$output" | tail -n 1)
if [[ "$test_status" != 0 || -z "$maximum" || "$maximum" -gt "$MAX_LATENCY_US" ]]; then
    echo "REALTIME_AUDIT_REJECTED max=${maximum:-unknown}us limit=${MAX_LATENCY_US}us" >&2
    echo "Evidence: $log_file" >&2
    exit 69
fi

echo "REALTIME_AUDIT_OK max=${maximum}us limit=${MAX_LATENCY_US}us"
echo "Evidence: $log_file"
