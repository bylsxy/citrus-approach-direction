#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = np.array(
    [
        [220, 30, 30],
        [255, 140, 0],
        [60, 170, 220],
        [120, 80, 220],
        [80, 190, 90],
        [230, 210, 70],
        [190, 80, 160],
    ],
    dtype=np.uint8,
)


def pack_rgb(r, g, b):
    return int(r) << 16 | int(g) << 8 | int(b)


def write_pcd(path, points, colors):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n")
        f.write("FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {pack_rgb(c[0], c[1], c[2])}\n")


def sphere_points(center, radius, n_lat=12, n_lon=24):
    pts = []
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            pts.append(
                [
                    center[0] + radius * np.sin(phi) * np.cos(theta),
                    center[1] + radius * np.sin(phi) * np.sin(theta),
                    center[2] + radius * np.cos(phi),
                ]
            )
    return np.asarray(pts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Create fruit sphere model PCD and preview image.")
    parser.add_argument("--fruit-shape", required=True)
    parser.add_argument("--sphere-pcd", required=True)
    parser.add_argument("--center-pcd", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--max-preview-fruits", type=int, default=120)
    args = parser.parse_args()

    shapes = []
    with open(args.fruit_shape, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shapes.append(row)

    sphere_all = []
    sphere_colors = []
    centers = []
    center_colors = []
    for row in shapes:
        fid = int(row["fruit_id"])
        center = np.array([float(row["center_x"]), float(row["center_y"]), float(row["center_z"])], dtype=np.float32)
        radius = float(row["radius"])
        if radius <= 0:
            continue
        color = PALETTE[fid % len(PALETTE)]
        pts = sphere_points(center, radius)
        sphere_all.append(pts)
        sphere_colors.append(np.repeat(color.reshape(1, 3), len(pts), axis=0))
        centers.append(center)
        center_colors.append(color)

    sphere_points_all = np.vstack(sphere_all) if sphere_all else np.empty((0, 3), dtype=np.float32)
    sphere_colors_all = np.vstack(sphere_colors) if sphere_colors else np.empty((0, 3), dtype=np.uint8)
    center_points = np.asarray(centers, dtype=np.float32) if centers else np.empty((0, 3), dtype=np.float32)
    center_colors = np.asarray(center_colors, dtype=np.uint8) if centers else np.empty((0, 3), dtype=np.uint8)
    write_pcd(args.sphere_pcd, sphere_points_all, sphere_colors_all)
    write_pcd(args.center_pcd, center_points, center_colors)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if len(center_points):
        show = min(args.max_preview_fruits, len(center_points))
        idx = np.linspace(0, len(center_points) - 1, show).astype(int)
        ax.scatter(center_points[idx, 0], center_points[idx, 1], center_points[idx, 2], s=18, c=center_colors[idx] / 255.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Fruit sphere model")
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout()
    Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.preview, dpi=220)
    plt.close(fig)
    print(f"fruit_spheres={len(center_points)} sphere_pcd={args.sphere_pcd} center_pcd={args.center_pcd}")


if __name__ == "__main__":
    main()
