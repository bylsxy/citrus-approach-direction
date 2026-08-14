#!/usr/bin/env python3
import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import yaml


PALETTE_BGR = np.array(
    [
        [0, 0, 180],
        [0, 140, 255],
        [255, 120, 0],
        [80, 180, 80],
        [180, 80, 180],
        [220, 220, 60],
        [60, 200, 220],
        [120, 80, 220],
    ],
    dtype=np.uint8,
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_points_csv(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    pts = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float32)
    conf = np.array([float(r.get("confidence", 1.0)) for r in rows], dtype=np.float32)
    return pts, conf


def grid_dbscan(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    n = len(points)
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    cell = np.floor(points / eps).astype(np.int32)
    buckets = {}
    for i, key in enumerate(map(tuple, cell)):
        buckets.setdefault(key, []).append(i)

    def neighbors(i):
        c = cell[i]
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out.extend(buckets.get(tuple(c + np.array([dx, dy, dz])), []))
        if not out:
            return []
        cand = np.array(out, dtype=np.int32)
        dist = np.linalg.norm(points[cand] - points[i], axis=1)
        return cand[dist <= eps].tolist()

    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbs = neighbors(i)
        if len(nbs) < min_points:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = deque(nbs)
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                jn = neighbors(j)
                if len(jn) >= min_points:
                    queue.extend(jn)
            if labels[j] < 0:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def pack_rgb(r, g, b):
    return int(r) << 16 | int(g) << 8 | int(b)


def write_pcd(path: Path, points: np.ndarray, labels: np.ndarray):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        for p, label in zip(points, labels):
            color = np.array([80, 80, 80], dtype=np.uint8) if label < 0 else PALETTE_BGR[label % len(PALETTE_BGR)]
            rgb = pack_rgb(color[2], color[1], color[0])
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {rgb}\n")


def main():
    parser = argparse.ArgumentParser(description="Cluster fruit points into per-fruit instances.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--instances-csv", required=True)
    parser.add_argument("--clustered-points-csv", required=True)
    parser.add_argument("--output-pcd", required=True)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--min-points", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    eps = float(args.eps if args.eps is not None else cfg["clustering"]["eps_m"])
    min_points = int(args.min_points if args.min_points is not None else cfg["clustering"]["min_points"])
    points, conf = read_points_csv(Path(args.input))
    labels = grid_dbscan(points, eps, min_points)

    ensure_dir(Path(args.instances_csv).parent)
    with open(args.instances_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit_id", "center_x", "center_y", "center_z", "point_count", "diameter_bbox_m"])
        for label in sorted([x for x in np.unique(labels) if x >= 0]):
            cluster = points[labels == label]
            center = cluster.mean(axis=0)
            bbox = cluster.max(axis=0) - cluster.min(axis=0)
            diameter = float(np.max(bbox))
            writer.writerow([int(label), f"{center[0]:.6f}", f"{center[1]:.6f}", f"{center[2]:.6f}", len(cluster), f"{diameter:.6f}"])

    with open(args.clustered_points_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "class_id", "confidence", "fruit_id"])
        for p, score, label in zip(points, conf, labels):
            writer.writerow([f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}", 0, f"{score:.6f}", int(label)])

    write_pcd(Path(args.output_pcd), points, labels)
    n_clusters = len([x for x in np.unique(labels) if x >= 0])
    print(f"instances={n_clusters} noise_points={int(np.sum(labels < 0))}")


if __name__ == "__main__":
    main()
