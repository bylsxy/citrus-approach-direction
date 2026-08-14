#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def voxel_downsample(points, voxel):
    if voxel <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(keep)]


def robust_filter(points, quantile=0.90):
    if len(points) < 8:
        return points
    center = np.median(points, axis=0)
    d = np.linalg.norm(points - center, axis=1)
    cutoff = np.quantile(d, quantile)
    keep = d <= cutoff
    if keep.sum() < 8:
        return points
    return points[keep]


def fit_sphere(points):
    a = np.column_stack([points[:, 0], points[:, 1], points[:, 2], np.ones(len(points))])
    b = -(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = -coef[:3] / 2.0
    r2 = float(np.sum(center**2) - coef[3])
    radius = np.sqrt(max(r2, 0.0))
    residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    return center, float(radius), float(np.mean(residual))


def pca_ellipsoid(points):
    centered = points - points.mean(axis=0)
    if len(points) < 3:
        return np.zeros(3), 0.0
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    proj = centered @ vecs
    axes = np.maximum((proj.max(axis=0) - proj.min(axis=0)) / 2.0, 1e-6)
    volume = 4.0 / 3.0 * np.pi * axes[0] * axes[1] * axes[2]
    return axes, float(volume)


def read_clustered_points(path):
    clusters = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = int(row["fruit_id"])
            if fid < 0:
                continue
            clusters.setdefault(fid, []).append([float(row["x"]), float(row["y"]), float(row["z"])])
    return {fid: np.asarray(pts, dtype=np.float64) for fid, pts in clusters.items()}


def main():
    parser = argparse.ArgumentParser(description="Fit sphere/ellipsoid fruit shape models from clustered world fruit points.")
    parser.add_argument("--clustered-points", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.015)
    parser.add_argument("--outlier-quantile", type=float, default=0.90)
    parser.add_argument("--min-points", type=int, default=12)
    parser.add_argument("--ellipsoid-error-ratio", type=float, default=0.25)
    args = parser.parse_args()

    clusters = read_clustered_points(Path(args.clustered_points))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    success = 0
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fruit_id",
                "center_x",
                "center_y",
                "center_z",
                "radius",
                "diameter",
                "volume",
                "fit_error",
                "point_count",
                "filtered_point_count",
                "model",
                "axis_x",
                "axis_y",
                "axis_z",
            ]
        )
        for fid in sorted(clusters):
            raw = clusters[fid]
            pts = voxel_downsample(raw, args.voxel_size)
            pts = robust_filter(pts, args.outlier_quantile)
            if len(pts) < args.min_points:
                continue
            center, radius, err = fit_sphere(pts)
            axes, ellipsoid_volume = pca_ellipsoid(pts)
            model = "sphere"
            volume = 4.0 / 3.0 * np.pi * radius**3
            diameter = 2.0 * radius
            if radius <= 1e-6 or err / max(radius, 1e-6) > args.ellipsoid_error_ratio:
                model = "ellipsoid"
                diameter = 2.0 * float((axes[0] * axes[1] * axes[2]) ** (1.0 / 3.0))
                radius = diameter / 2.0
                volume = ellipsoid_volume
            writer.writerow(
                [
                    fid,
                    f"{center[0]:.6f}",
                    f"{center[1]:.6f}",
                    f"{center[2]:.6f}",
                    f"{radius:.6f}",
                    f"{diameter:.6f}",
                    f"{volume:.9f}",
                    f"{err:.6f}",
                    len(raw),
                    len(pts),
                    model,
                    f"{2*axes[0]:.6f}",
                    f"{2*axes[1]:.6f}",
                    f"{2*axes[2]:.6f}",
                ]
            )
            success += 1
    print(f"fruit_instance_count={len(clusters)} successful_fit_count={success} output={args.output}")


if __name__ == "__main__":
    main()
