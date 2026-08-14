#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_tree_points(path, max_points=60000):
    pts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
            labels.append(int(row["tree_id"]))
    pts = np.asarray(pts, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(pts), max_points, replace=False)
        pts, labels = pts[keep], labels[keep]
    return pts, labels


def read_shapes(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def read_assoc(path):
    assoc = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assoc[int(row["fruit_id"])] = int(row["tree_id"])
    return assoc


def main():
    parser = argparse.ArgumentParser(description="Create compact tree-fruit digital twin preview.")
    parser.add_argument("--tree-points", required=True)
    parser.add_argument("--fruit-shape", required=True)
    parser.add_argument("--associations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-fruits", type=int, default=250)
    args = parser.parse_args()

    tree_pts, tree_labels = read_tree_points(Path(args.tree_points))
    shapes = read_shapes(Path(args.fruit_shape))
    assoc = read_assoc(Path(args.associations))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(tree_pts):
        unique = sorted(np.unique(tree_labels))
        cmap = plt.get_cmap("Greens")
        for label in unique:
            pts = tree_pts[tree_labels == label]
            color = cmap(0.35 + 0.5 * ((label % 6) / 5.0))
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c=[color], alpha=0.22)

    if shapes:
        show_shapes = shapes[: args.max_fruits]
        centers = np.array([[float(r["center_x"]), float(r["center_y"]), float(r["center_z"])] for r in show_shapes], dtype=np.float32)
        radii = np.array([float(r["radius"]) for r in show_shapes], dtype=np.float32)
        colors = []
        for r in show_shapes:
            tid = assoc.get(int(r["fruit_id"]), -1)
            colors.append(plt.get_cmap("tab20")((tid % 20) if tid >= 0 else 19))
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=np.clip(radii * 600, 8, 80), c=colors, alpha=0.92)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Tree-fruit digital twin")
    ax.view_init(elev=35, azim=-55)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=230)
    plt.close(fig)
    print(f"digital_twin_preview={args.output}")


if __name__ == "__main__":
    main()
