#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Visualize fruit map with Open3D when available.")
    parser.add_argument("--pcd", required=True)
    parser.add_argument("--instances", default="")
    parser.add_argument("--sizes", default="")
    args = parser.parse_args()

    print(f"PCD: {args.pcd}")
    if args.instances and Path(args.instances).exists():
        with open(args.instances, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"Instances: {len(rows)}")
        for row in rows[:20]:
            print(row)

    try:
        import open3d as o3d
    except Exception as exc:
        print("Open3D is not installed in this environment. Use pcl_viewer or CloudCompare to open the PCD.")
        print(f"Import error: {exc}")
        return

    cloud = o3d.io.read_point_cloud(args.pcd)
    if len(cloud.points) > 300000:
        cloud = cloud.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw_geometries([cloud])


if __name__ == "__main__":
    main()
