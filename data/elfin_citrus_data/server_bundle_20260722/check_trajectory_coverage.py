#!/usr/bin/env python3
import argparse
from pathlib import Path


def read_timestamps(path):
    ts = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.strip().replace(",", " ").split()
        if not parts:
            continue
        try:
            ts.append(float(parts[0]))
        except ValueError:
            pass
    return ts


def main():
    parser = argparse.ArgumentParser(description="Check trajectory time coverage against a rosbag time range.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--bag-start", type=float, required=True)
    parser.add_argument("--bag-end", type=float, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.98)
    args = parser.parse_args()

    ts = read_timestamps(args.trajectory)
    if not ts:
        raise SystemExit("No trajectory timestamps found.")
    traj_start, traj_end = min(ts), max(ts)
    bag_duration = args.bag_end - args.bag_start
    overlap = max(0.0, min(traj_end, args.bag_end) - max(traj_start, args.bag_start))
    coverage = overlap / bag_duration if bag_duration > 0 else 0.0
    print(f"trajectory={args.trajectory}")
    print(f"trajectory_count={len(ts)}")
    print(f"bag_start={args.bag_start:.9f}")
    print(f"bag_end={args.bag_end:.9f}")
    print(f"traj_start={traj_start:.9f}")
    print(f"traj_end={traj_end:.9f}")
    print(f"coverage={coverage:.6f}")
    if coverage < args.min_coverage:
        raise SystemExit(f"Coverage {coverage:.6f} is below required {args.min_coverage:.6f}.")


if __name__ == "__main__":
    main()
