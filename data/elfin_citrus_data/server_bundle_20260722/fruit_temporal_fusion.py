#!/usr/bin/env python3
import argparse
import csv
from glob import glob
from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_instances(paths):
    rows = []
    for frame_idx, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "frame": frame_idx,
                        "source": path,
                        "center": np.array([float(row["center_x"]), float(row["center_y"]), float(row["center_z"])]),
                        "point_count": int(float(row["point_count"])),
                        "diameter": float(row.get("diameter_bbox_m", 0) or 0),
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fuse frame-level fruit instances into global instances.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = []
    for pattern in args.inputs:
        paths.extend(sorted(glob(pattern)))
    paths = sorted(set(paths))
    rows = read_instances(paths)
    dist_th = float(cfg["fusion"]["distance_threshold_m"])
    dia_th = float(cfg["fusion"]["diameter_threshold_m"])

    globals_ = []
    for row in rows:
        best = None
        best_dist = None
        for g in globals_:
            dist = float(np.linalg.norm(row["center"] - g["center"]))
            dia_ok = (row["diameter"] <= 0 or g["diameter"] <= 0 or abs(row["diameter"] - g["diameter"]) <= dia_th)
            if dist < dist_th and dia_ok and (best_dist is None or dist < best_dist):
                best = g
                best_dist = dist
        if best is None:
            globals_.append(
                {
                    "center": row["center"].copy(),
                    "diameter": row["diameter"],
                    "observations": 1,
                    "point_count": row["point_count"],
                }
            )
        else:
            n = best["observations"]
            best["center"] = (best["center"] * n + row["center"]) / (n + 1)
            best["diameter"] = (best["diameter"] * n + row["diameter"]) / (n + 1)
            best["observations"] += 1
            best["point_count"] += row["point_count"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit_id", "center_x", "center_y", "center_z", "observations", "point_count", "mean_diameter_bbox_m"])
        for i, g in enumerate(globals_):
            c = g["center"]
            writer.writerow([i, f"{c[0]:.6f}", f"{c[1]:.6f}", f"{c[2]:.6f}", g["observations"], g["point_count"], f"{g['diameter']:.6f}"])
    print(f"global_instances={len(globals_)} output={args.output}")


if __name__ == "__main__":
    main()
