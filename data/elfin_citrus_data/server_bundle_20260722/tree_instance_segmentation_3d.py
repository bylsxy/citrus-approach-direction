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


def load_tree_points(path, stride=1):
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if stride > 1 and i % stride:
                continue
            pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.asarray(pts, dtype=np.float32)


def density_peaks(points, cell_size, min_distance, quantile, max_trees):
    xy = points[:, :2]
    mn = xy.min(axis=0)
    ij = np.floor((xy - mn) / cell_size).astype(np.int32)
    shape = ij.max(axis=0) + 1
    grid = np.zeros((shape[0], shape[1]), dtype=np.float32)
    for i, j in ij:
        grid[i, j] += 1
    smooth = grid.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            smooth += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    threshold = max(5.0, float(np.quantile(smooth[smooth > 0], quantile)) if np.any(smooth > 0) else 5.0)
    candidates = []
    for i in range(1, smooth.shape[0] - 1):
        for j in range(1, smooth.shape[1] - 1):
            val = smooth[i, j]
            if val < threshold:
                continue
            patch = smooth[i - 1 : i + 2, j - 1 : j + 2]
            if val >= patch.max():
                center = mn + (np.array([i + 0.5, j + 0.5]) * cell_size)
                candidates.append((val, center))
    candidates.sort(key=lambda x: x[0], reverse=True)
    centers = []
    for _, c in candidates:
        if all(np.linalg.norm(c - old) >= min_distance for old in centers):
            centers.append(c)
        if len(centers) >= max_trees:
            break
    if not centers and len(points):
        centers = [xy.mean(axis=0)]
    return np.asarray(centers, dtype=np.float32)


def assign_points(points, centers_xy, max_radius):
    if len(centers_xy) == 0:
        return np.full(len(points), -1, dtype=np.int32)
    d = np.linalg.norm(points[:, None, :2] - centers_xy[None, :, :], axis=2)
    labels = np.argmin(d, axis=1).astype(np.int32)
    min_d = d[np.arange(len(points)), labels]
    labels[min_d > max_radius] = -1
    return labels


def pack_rgb(r, g, b):
    return int(r) << 16 | int(g) << 8 | int(b)


def write_pcd(path, points, labels):
    palette = np.array(
        [[40, 160, 70], [70, 190, 150], [150, 180, 60], [80, 130, 220], [180, 120, 70], [130, 80, 180]],
        dtype=np.uint8,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n")
        f.write("FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        for p, label in zip(points, labels):
            color = np.array([120, 120, 120], dtype=np.uint8) if label < 0 else palette[label % len(palette)]
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {pack_rgb(color[0], color[1], color[2])}\n")


def main():
    parser = argparse.ArgumentParser(description="Geometry-based 3D tree instance segmentation.")
    parser.add_argument("--tree-points", required=True)
    parser.add_argument("--output-points", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-pcd", required=True)
    parser.add_argument("--read-stride", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.15)
    parser.add_argument("--ground-quantile", type=float, default=0.05)
    parser.add_argument("--ground-offset", type=float, default=0.35)
    parser.add_argument("--cell-size", type=float, default=0.45)
    parser.add_argument("--peak-min-distance", type=float, default=2.2)
    parser.add_argument("--peak-quantile", type=float, default=0.92)
    parser.add_argument("--max-trees", type=int, default=30)
    parser.add_argument("--assignment-radius", type=float, default=2.6)
    parser.add_argument("--min-tree-points", type=int, default=30)
    args = parser.parse_args()

    raw = load_tree_points(Path(args.tree_points), args.read_stride)
    if len(raw) == 0:
        raise RuntimeError("No tree points loaded.")
    z_ground = float(np.quantile(raw[:, 2], args.ground_quantile))
    filtered = raw[raw[:, 2] >= z_ground + args.ground_offset]
    ds = voxel_downsample(filtered, args.voxel_size)
    centers_xy = density_peaks(ds, args.cell_size, args.peak_min_distance, args.peak_quantile, args.max_trees)
    labels = assign_points(ds, centers_xy, args.assignment_radius)

    keep_labels = []
    for label in sorted([x for x in np.unique(labels) if x >= 0]):
        if int(np.sum(labels == label)) >= args.min_tree_points:
            keep_labels.append(label)
    remap = {old: new for new, old in enumerate(keep_labels)}
    new_labels = np.full_like(labels, -1)
    for old, new in remap.items():
        new_labels[labels == old] = new

    Path(args.output_points).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_points, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tree_id", "point_index", "x", "y", "z"])
        for idx, (p, label) in enumerate(zip(ds, new_labels)):
            if label >= 0:
                writer.writerow([int(label), idx, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])

    with open(args.summary, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tree_id", "center_x", "center_y", "center_z", "height", "width", "point_count", "min_z", "max_z"])
        for label in sorted([x for x in np.unique(new_labels) if x >= 0]):
            pts = ds[new_labels == label]
            center = pts.mean(axis=0)
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            width = float(max(mx[0] - mn[0], mx[1] - mn[1]))
            writer.writerow(
                [
                    int(label),
                    f"{center[0]:.6f}",
                    f"{center[1]:.6f}",
                    f"{center[2]:.6f}",
                    f"{mx[2]-mn[2]:.6f}",
                    f"{width:.6f}",
                    len(pts),
                    f"{mn[2]:.6f}",
                    f"{mx[2]:.6f}",
                ]
            )
    write_pcd(args.output_pcd, ds, new_labels)
    print(f"tree_instance_count={len([x for x in np.unique(new_labels) if x >= 0])} tree_points={len(ds)} raw_points={len(raw)}")


if __name__ == "__main__":
    main()
