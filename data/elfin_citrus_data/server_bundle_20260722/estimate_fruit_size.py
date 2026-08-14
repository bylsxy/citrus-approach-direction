#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def read_points(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["fruit_id"]) >= 0:
                rows.append(row)
    pts = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float32)
    labels = np.array([int(r["fruit_id"]) for r in rows], dtype=np.int32)
    return pts, labels


def sphere_fit(points):
    center = points.mean(axis=0)
    radius = float(np.median(np.linalg.norm(points - center, axis=1)))
    volume = 4.0 / 3.0 * np.pi * radius**3
    return center, radius, volume


def ellipsoid_fit(points):
    centered = points - points.mean(axis=0)
    if len(points) < 3:
        return (0.0, 0.0, 0.0), 0.0
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    proj = centered @ vecs
    axes = np.maximum((proj.max(axis=0) - proj.min(axis=0)) / 2.0, 1e-6)
    volume = 4.0 / 3.0 * np.pi * axes[0] * axes[1] * axes[2]
    return axes, volume


def main():
    parser = argparse.ArgumentParser(description="Estimate fruit size from world fruit instances.")
    parser.add_argument("--clustered-points", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pts, labels = read_points(Path(args.clustered_points))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit_id", "diameter_sphere_m", "diameter_ellipsoid_m", "axis_x_m", "axis_y_m", "axis_z_m", "volume_sphere_m3", "volume_ellipsoid_m3", "point_count"])
        for label in sorted(np.unique(labels)):
            cluster = pts[labels == label]
            _, radius, vol_s = sphere_fit(cluster)
            axes, vol_e = ellipsoid_fit(cluster)
            d_e = 2.0 * float((axes[0] * axes[1] * axes[2]) ** (1.0 / 3.0))
            writer.writerow([int(label), f"{2*radius:.6f}", f"{d_e:.6f}", f"{2*axes[0]:.6f}", f"{2*axes[1]:.6f}", f"{2*axes[2]:.6f}", f"{vol_s:.9f}", f"{vol_e:.9f}", len(cluster)])
    print(f"fruit_size={args.output}")


if __name__ == "__main__":
    main()
