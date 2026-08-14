#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_instances(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate fruit instances in world coordinates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    rows = read_instances(Path(args.instances))
    dist_th = float(cfg["fusion"]["distance_threshold_m"])
    dia_th = float(cfg["fusion"]["diameter_threshold_m"])
    merged = []
    for row in rows:
        center = np.array([float(row["center_x"]), float(row["center_y"]), float(row["center_z"])], dtype=np.float64)
        diameter = float(row.get("diameter_bbox_m", 0) or 0)
        point_count = int(float(row.get("point_count", 0) or 0))
        observations = int(float(row.get("observations", 1) or 1))
        best = None
        best_dist = 1e9
        for g in merged:
            dist = float(np.linalg.norm(center - g["center"]))
            size_ok = diameter <= 0 or g["diameter"] <= 0 or abs(diameter - g["diameter"]) < dia_th
            if dist < dist_th and size_ok and dist < best_dist:
                best = g
                best_dist = dist
        if best is None:
            merged.append({"center": center, "diameter": diameter, "point_count": point_count, "observations": observations})
        else:
            w0 = max(best["point_count"], 1)
            w1 = max(point_count, 1)
            best["center"] = (best["center"] * w0 + center * w1) / (w0 + w1)
            best["diameter"] = (best["diameter"] * best["observations"] + diameter * observations) / (best["observations"] + observations)
            best["point_count"] += point_count
            best["observations"] += observations

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit_id", "center_x", "center_y", "center_z", "observations", "point_count", "mean_diameter_bbox_m"])
        for i, g in enumerate(merged):
            c = g["center"]
            writer.writerow([i, f"{c[0]:.6f}", f"{c[1]:.6f}", f"{c[2]:.6f}", g["observations"], g["point_count"], f"{g['diameter']:.6f}"])
    print(f"global_fruit_instances={len(merged)}")


if __name__ == "__main__":
    main()
