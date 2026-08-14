#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def read_clustered_points(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    points = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float32)
    labels = np.array([int(r["fruit_id"]) for r in rows], dtype=np.int32)
    return points, labels


def estimate_axes(points: np.ndarray):
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    proj = centered @ vecs
    extents = proj.max(axis=0) - proj.min(axis=0)
    axes = np.maximum(extents / 2.0, 1e-6)
    return float(axes[0]), float(axes[1]), float(axes[2])


def main():
    parser = argparse.ArgumentParser(description="Estimate fruit size from clustered 3D points.")
    parser.add_argument("--clustered-points", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    points, labels = read_clustered_points(Path(args.clustered_points))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit_id", "diameter_m", "axis_x_m", "axis_y_m", "axis_z_m", "volume_m3", "method", "point_count"])
        for label in sorted([x for x in np.unique(labels) if x >= 0]):
            cluster = points[labels == label]
            a, b, c = estimate_axes(cluster)
            volume = 4.0 / 3.0 * np.pi * a * b * c
            equiv_diameter = 2.0 * (a * b * c) ** (1.0 / 3.0)
            writer.writerow([int(label), f"{equiv_diameter:.6f}", f"{2*a:.6f}", f"{2*b:.6f}", f"{2*c:.6f}", f"{volume:.9f}", "ellipsoid_pca", len(cluster)])
    print(f"size_output={args.output}")


if __name__ == "__main__":
    main()
