#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Associate fruit instances to tree instances by nearest tree center with height constraint.")
    parser.add_argument("--fruit-instances", required=True)
    parser.add_argument("--tree-instances", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-xy-distance", type=float, default=3.5)
    parser.add_argument("--height-margin", type=float, default=0.8)
    args = parser.parse_args()
    fruits = read_csv(Path(args.fruit_instances))
    trees = read_csv(Path(args.tree_instances))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit_id", "tree_id", "distance", "height_difference", "method"])
        for fr in fruits:
            fp = np.array([float(fr["center_x"]), float(fr["center_y"]), float(fr["center_z"])])
            candidates = []
            for tr in trees:
                tc = np.array([float(tr["center_x"]), float(tr["center_y"]), float(tr["center_z"])])
                xy_dist = float(np.linalg.norm(fp[:2] - tc[:2]))
                height_diff = float(fp[2] - tc[2])
                if "min_z" in tr and "max_z" in tr:
                    valid_h = float(tr["min_z"]) - args.height_margin <= fp[2] <= float(tr["max_z"]) + args.height_margin
                elif "height" in tr:
                    valid_h = abs(height_diff) <= float(tr["height"]) / 2.0 + args.height_margin
                else:
                    valid_h = True
                if valid_h and xy_dist <= args.max_xy_distance:
                    candidates.append((xy_dist, abs(height_diff), tr, "nearest_valid_tree_center"))
            if candidates:
                dist, _, tr, method = sorted(candidates, key=lambda x: x[0])[0]
                writer.writerow([fr["fruit_id"], tr["tree_id"], f"{dist:.6f}", f"{fp[2]-float(tr['center_z']):.6f}", "nearest_valid_tree_center"])
            elif trees:
                fallback = []
                for tr in trees:
                    tc = np.array([float(tr["center_x"]), float(tr["center_y"]), float(tr["center_z"])])
                    fallback.append((float(np.linalg.norm(fp[:2] - tc[:2])), abs(float(fp[2] - tc[2])), tr))
                dist, _, tr = sorted(fallback, key=lambda x: x[0])[0]
                writer.writerow([fr["fruit_id"], tr["tree_id"], f"{dist:.6f}", f"{fp[2]-float(tr['center_z']):.6f}", "nearest_tree_center_fallback"])
            else:
                writer.writerow([fr["fruit_id"], -1, "nan", "nan", "no_tree_instance"])
    print(f"associations={len(fruits)}")


if __name__ == "__main__":
    main()
