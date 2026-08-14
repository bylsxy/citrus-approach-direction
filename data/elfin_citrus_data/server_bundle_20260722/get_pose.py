#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def quat_to_rot_wxyz(q):
    w, x, y, z = q
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("zero quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_trajectory(path: Path, quat_order: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 8:
                continue
            t = vals[0]
            p = np.array(vals[1:4], dtype=np.float64)
            q_raw = np.array(vals[4:8], dtype=np.float64)
            if quat_order == "wxyz":
                q = q_raw
            elif quat_order == "xyzw":
                q = np.array([q_raw[3], q_raw[0], q_raw[1], q_raw[2]], dtype=np.float64)
            else:
                raise ValueError(f"Unsupported quaternion order: {quat_order}")
            rows.append((t, p, q))
    if not rows:
        raise RuntimeError(f"No poses parsed from {path}")
    return rows


def nearest_pose(rows, timestamp):
    times = np.array([r[0] for r in rows], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - timestamp)))
    return rows[idx], abs(times[idx] - timestamp)


def main():
    parser = argparse.ArgumentParser(description="Convert VINS trajectory to pose CSV with T_world_camera.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-index", default="")
    parser.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    parser.add_argument("--max-dt", type=float, default=0.2)
    args = parser.parse_args()

    traj = read_trajectory(Path(args.trajectory), args.quat_order)
    out_rows = []

    if args.frames_index:
        with open(args.frames_index, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                frame = row["frame"]
                ts = float(row["t_rgb"])
                (pose_t, p, q), dt = nearest_pose(traj, ts)
                valid = dt <= args.max_dt
                out_rows.append((frame, ts, pose_t, dt, valid, p, q))
    else:
        for pose_t, p, q in traj:
            out_rows.append(("", pose_t, pose_t, 0.0, True, p, q))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "timestamp",
                "pose_timestamp",
                "pose_dt",
                "pose_valid",
                "tx",
                "ty",
                "tz",
                "qw",
                "qx",
                "qy",
                "qz",
                "T00",
                "T01",
                "T02",
                "T03",
                "T10",
                "T11",
                "T12",
                "T13",
                "T20",
                "T21",
                "T22",
                "T23",
                "T30",
                "T31",
                "T32",
                "T33",
            ]
        )
        for frame, ts, pose_t, dt, valid, p, q in out_rows:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = quat_to_rot_wxyz(q)
            T[:3, 3] = p
            writer.writerow(
                [
                    frame,
                    f"{ts:.9f}",
                    f"{pose_t:.9f}",
                    f"{dt:.9f}",
                    int(valid),
                    f"{p[0]:.9f}",
                    f"{p[1]:.9f}",
                    f"{p[2]:.9f}",
                    f"{q[0]:.9f}",
                    f"{q[1]:.9f}",
                    f"{q[2]:.9f}",
                    f"{q[3]:.9f}",
                ]
                + [f"{x:.9f}" for x in T.reshape(-1)]
            )
    valid_count = sum(1 for r in out_rows if r[4])
    print(f"poses={len(out_rows)} valid={valid_count} output={args.output}")


if __name__ == "__main__":
    main()
