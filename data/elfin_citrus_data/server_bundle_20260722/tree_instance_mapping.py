#!/usr/bin/env python3
import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np


def read_points(path, class_id=1):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["class_id"]) == class_id:
                rows.append(row)
    pts = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float32)
    return pts


def dbscan(points, eps, min_points):
    n = len(points)
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    cell = np.floor(points / eps).astype(np.int32)
    buckets = {}
    for i, k in enumerate(map(tuple, cell)):
        buckets.setdefault(k, []).append(i)

    def neigh(i):
        c = cell[i]
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out.extend(buckets.get(tuple(c + np.array([dx, dy, dz])), []))
        if not out:
            return []
        cand = np.array(out, dtype=np.int32)
        d = np.linalg.norm(points[cand] - points[i], axis=1)
        return cand[d <= eps].tolist()

    cid = 0
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        ns = neigh(i)
        if len(ns) < min_points:
            continue
        labels[i] = cid
        q = deque(ns)
        while q:
            j = q.popleft()
            if not visited[j]:
                visited[j] = True
                js = neigh(j)
                if len(js) >= min_points:
                    q.extend(js)
            if labels[j] < 0:
                labels[j] = cid
        cid += 1
    return labels


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(keep)]


def pack_rgb(r, g, b):
    return int(r) << 16 | int(g) << 8 | int(b)


def write_pcd(path, points, labels):
    palette = np.array([[0, 150, 0], [0, 210, 120], [80, 180, 80], [120, 220, 120]], dtype=np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        for p, label in zip(points, labels):
            c = np.array([90, 90, 90], dtype=np.uint8) if label < 0 else palette[label % len(palette)]
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {pack_rgb(c[2], c[1], c[0])}\n")


def main():
    parser = argparse.ArgumentParser(description="Build tree instance map from world tree points.")
    parser.add_argument("--points", required=True)
    parser.add_argument("--output-instances", required=True)
    parser.add_argument("--output-pcd", required=True)
    parser.add_argument("--eps", type=float, default=0.35)
    parser.add_argument("--min-points", type=int, default=500)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    args = parser.parse_args()
    raw_pts = read_points(Path(args.points), class_id=1)
    pts = voxel_downsample(raw_pts, args.voxel_size)
    labels = dbscan(pts, args.eps, args.min_points)
    Path(args.output_instances).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_instances, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tree_id", "center_x", "center_y", "center_z", "point_count", "height_m", "crown_size_x_m", "crown_size_y_m", "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"])
        for label in sorted([x for x in np.unique(labels) if x >= 0]):
            cpts = pts[labels == label]
            center = cpts.mean(axis=0)
            mn = cpts.min(axis=0)
            mx = cpts.max(axis=0)
            writer.writerow([int(label), f"{center[0]:.6f}", f"{center[1]:.6f}", f"{center[2]:.6f}", len(cpts), f"{mx[2]-mn[2]:.6f}", f"{mx[0]-mn[0]:.6f}", f"{mx[1]-mn[1]:.6f}", f"{mn[0]:.6f}", f"{mx[0]:.6f}", f"{mn[1]:.6f}", f"{mx[1]:.6f}", f"{mn[2]:.6f}", f"{mx[2]:.6f}"])
    write_pcd(args.output_pcd, pts, labels)
    print(f"tree_instances={len([x for x in np.unique(labels) if x >= 0])} tree_points={len(pts)} raw_tree_points={len(raw_pts)}")


if __name__ == "__main__":
    main()
