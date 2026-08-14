#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def read_instances(path, id_key):
    rows = []
    if not path or not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row[id_key], np.array([float(row["center_x"]), float(row["center_y"]), float(row["center_z"])])))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Open3D visualization for world fruit-tree digital twin.")
    parser.add_argument("--fruit-pcd", required=True)
    parser.add_argument("--tree-pcd", default="")
    parser.add_argument("--fruit-instances", default="")
    parser.add_argument("--tree-instances", default="")
    args = parser.parse_args()

    try:
        import open3d as o3d
    except Exception as exc:
        print("Open3D is not installed. Open the PCD files with pcl_viewer or CloudCompare.")
        print(f"Import error: {exc}")
        return

    geoms = []
    fruit = o3d.io.read_point_cloud(args.fruit_pcd)
    if len(fruit.points) > 300000:
        fruit = fruit.voxel_down_sample(0.02)
    geoms.append(fruit)
    if args.tree_pcd and Path(args.tree_pcd).exists():
        tree = o3d.io.read_point_cloud(args.tree_pcd)
        if len(tree.points) > 300000:
            tree = tree.voxel_down_sample(0.03)
        geoms.append(tree)
    for _, c in read_instances(args.fruit_instances, "fruit_id"):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        mesh.translate(c)
        mesh.paint_uniform_color([0.9, 0.2, 0.1])
        geoms.append(mesh)
    for _, c in read_instances(args.tree_instances, "tree_id"):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        mesh.translate(c)
        mesh.paint_uniform_color([0.1, 0.6, 0.2])
        geoms.append(mesh)
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
