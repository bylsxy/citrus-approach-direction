#!/usr/bin/env python3
import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_semantic_points(path, class_id):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["class_id"]) == class_id:
                rows.append(row)
    pts = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float32)
    frames = [r.get("frame", "") for r in rows]
    conf = np.array([float(r.get("confidence", 1.0)) for r in rows], dtype=np.float32)
    return pts, frames, conf


def grid_dbscan(points, eps, min_points):
    n = len(points)
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    cell = np.floor(points / eps).astype(np.int32)
    buckets = {}
    for i, key in enumerate(map(tuple, cell)):
        buckets.setdefault(key, []).append(i)

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

    visited = np.zeros(n, dtype=bool)
    cid = 0
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


def voxel_downsample(points, frames, conf, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return points, frames, conf
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    return points[keep], [frames[i] for i in keep], conf[keep]


def pack_rgb(r, g, b):
    return int(r) << 16 | int(g) << 8 | int(b)


def write_cluster_pcd(path, points, labels):
    palette = np.array([[0, 0, 180], [0, 165, 255], [255, 120, 0], [80, 190, 80], [160, 80, 220], [220, 220, 80]], dtype=np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        for p, label in zip(points, labels):
            c = np.array([80, 80, 80], dtype=np.uint8) if label < 0 else palette[label % len(palette)]
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {pack_rgb(c[2], c[1], c[0])}\n")


def main():
    parser = argparse.ArgumentParser(description="Build fruit instance map from world citrus points.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--output-instances", required=True)
    parser.add_argument("--output-points", required=True)
    parser.add_argument("--output-pcd", required=True)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    eps = float(args.eps if args.eps is not None else cfg["clustering"]["eps_m"])
    min_points = int(args.min_points if args.min_points is not None else cfg["clustering"]["min_points"])
    raw_pts, raw_frames, raw_conf = read_semantic_points(Path(args.points), class_id=0)
    pts, frames, conf = voxel_downsample(raw_pts, raw_frames, raw_conf, args.voxel_size)
    labels = grid_dbscan(pts, eps, min_points)

    Path(args.output_instances).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_instances, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit_id", "center_x", "center_y", "center_z", "point_count", "observations", "diameter_bbox_m"])
        for label in sorted([x for x in np.unique(labels) if x >= 0]):
            cluster = pts[labels == label]
            obs = len(set([fr for fr, lb in zip(frames, labels) if lb == label and fr]))
            center = cluster.mean(axis=0)
            bbox = cluster.max(axis=0) - cluster.min(axis=0)
            writer.writerow([int(label), f"{center[0]:.6f}", f"{center[1]:.6f}", f"{center[2]:.6f}", len(cluster), obs, f"{float(np.max(bbox)):.6f}"])

    with open(args.output_points, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z", "class_id", "confidence", "fruit_id"])
        for p, fr, score, label in zip(pts, frames, conf, labels):
            writer.writerow([fr, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}", 0, f"{score:.6f}", int(label)])
    write_cluster_pcd(args.output_pcd, pts, labels)
    print(f"fruit_instances={len([x for x in np.unique(labels) if x >= 0])} points={len(pts)} raw_points={len(raw_pts)}")


if __name__ == "__main__":
    main()
