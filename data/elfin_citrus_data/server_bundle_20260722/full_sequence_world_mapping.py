#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pack_rgb(r, g, b):
    return int(r) << 16 | int(g) << 8 | int(b)


def read_frame_index(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_poses(path):
    poses = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = row["frame"]
            T = np.array([float(row[f"T{i}{j}"]) for i in range(4) for j in range(4)], dtype=np.float64).reshape(4, 4)
            poses[frame] = {
                "T": T,
                "valid": int(row["pose_valid"]) == 1,
                "pose_dt": float(row["pose_dt"]),
                "timestamp": float(row["timestamp"]),
            }
    return poses


class PcdStream:
    def __init__(self, path):
        self.path = Path(path)
        self.tmp = self.path.with_suffix(self.path.suffix + ".tmp_points")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.tmp, "w", encoding="utf-8")
        self.count = 0

    def write(self, points, color_bgr):
        if len(points) == 0:
            return
        rgb = pack_rgb(color_bgr[2], color_bgr[1], color_bgr[0])
        for p in points:
            self.f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {rgb}\n")
        self.count += len(points)

    def close(self):
        self.f.close()
        with open(self.path, "w", encoding="utf-8") as out:
            out.write("# .PCD v0.7 - Point Cloud Data file format\n")
            out.write("VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
            out.write(f"WIDTH {self.count}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {self.count}\nDATA ascii\n")
            with open(self.tmp, "r", encoding="utf-8") as inp:
                shutil.copyfileobj(inp, out)
        self.tmp.unlink(missing_ok=True)


def yolo_masks(rgb_bgr, model, conf, class_ids):
    h, w = rgb_bgr.shape[:2]
    masks = {cid: np.zeros((h, w), dtype=np.uint8) for cid in class_ids}
    confs = {cid: np.zeros((h, w), dtype=np.float32) for cid in class_ids}
    res = model.predict(rgb_bgr, conf=conf, verbose=False)[0]
    if res.masks is None or res.boxes is None:
        return masks, confs
    raw_masks = res.masks.data.detach().cpu().numpy()
    classes = res.boxes.cls.detach().cpu().numpy().astype(int)
    scores = res.boxes.conf.detach().cpu().numpy()
    for m, cls_id, score in zip(raw_masks, classes, scores):
        cls_id = int(cls_id)
        if cls_id not in masks:
            continue
        resized = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
        update = resized & (score >= confs[cls_id])
        masks[cls_id][update] = 1
        confs[cls_id][update] = float(score)
    return masks, confs


def reconstruct_points(rgb_shape, depth_raw, mask, score, T_wc, cfg, class_id, max_points):
    h, w = rgb_shape[:2]
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    if depth_raw.shape[:2] != (h, w):
        depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    depth = depth_raw.astype(np.float32) * float(cfg["camera"]["depth_scale"])
    valid = (
        (mask > 0)
        & np.isfinite(depth)
        & (depth >= float(cfg["camera"]["min_depth_m"]))
        & (depth <= float(cfg["camera"]["max_depth_m"]))
    )
    ys, xs = np.where(valid)
    if max_points > 0 and len(xs) > max_points:
        rng = np.random.default_rng(42 + class_id)
        keep = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys = xs[keep], ys[keep]
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    z = depth[ys, xs]
    fx, fy = float(cfg["camera"]["fx"]), float(cfg["camera"]["fy"])
    cx, cy = float(cfg["camera"]["cx"]), float(cfg["camera"]["cy"])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    pc = np.column_stack([x, y, z, np.ones_like(z)]).astype(np.float64)
    pw = (T_wc @ pc.T).T[:, :3].astype(np.float32)
    return pw, score[ys, xs].astype(np.float32)


def write_points_csv_rows(writer, frame, timestamp, points, scores, class_id):
    for p, score in zip(points, scores):
        writer.writerow([frame, f"{timestamp:.9f}", f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}", class_id, f"{score:.6f}"])


def main():
    parser = argparse.ArgumentParser(description="Run full-sequence VINS world-frame fruit/tree mapping.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames-index", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-frames", type=int, default=0, help="Debug only. 0 means all frames.")
    parser.add_argument("--max-fruit-points-per-frame", type=int, default=30000)
    parser.add_argument("--max-tree-points-per-frame", type=int, default=30000)
    parser.add_argument("--save-overlays-every", type=int, default=0)
    parser.add_argument("--run-clustering", action="store_true")
    parser.add_argument("--tree-eps", type=float, default=0.45)
    parser.add_argument("--tree-min-points", type=int, default=80)
    parser.add_argument("--tree-voxel-size", type=float, default=0.05)
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = load_yaml(args.config)
    frames = read_frame_index(args.frames_index)
    if args.limit_frames > 0:
        frames = frames[: args.limit_frames]
    poses = read_poses(args.poses)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    if args.save_overlays_every > 0:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    fruit_color = np.array(cfg["classes"]["citrus"]["bgr"], dtype=np.uint8)
    tree_color = np.array(cfg["classes"]["tree"]["bgr"], dtype=np.uint8)
    model = YOLO(args.model)
    conf_th = float(cfg["reconstruction"].get("segmentation_conf", 0.25))

    fruit_pcd = PcdStream(out_dir / "global_world_fruit_cloud.pcd")
    tree_pcd = PcdStream(out_dir / "global_world_tree_cloud.pcd")
    fruit_csv_path = out_dir / "global_world_fruit_points.csv"
    tree_csv_path = out_dir / "global_world_tree_points.csv"
    frame_stats_path = out_dir / "frame_pose_mapping_stats.csv"
    summary_path = out_dir / "sequence_summary.csv"

    frame_count = len(frames)
    valid_pose_count = 0
    processed_count = 0
    fruit_point_count = 0
    tree_point_count = 0

    with open(fruit_csv_path, "w", encoding="utf-8", newline="") as fruit_f, open(
        tree_csv_path, "w", encoding="utf-8", newline=""
    ) as tree_f, open(frame_stats_path, "w", encoding="utf-8", newline="") as stat_f:
        fruit_writer = csv.writer(fruit_f)
        tree_writer = csv.writer(tree_f)
        stat_writer = csv.writer(stat_f)
        fruit_writer.writerow(["frame", "timestamp", "x", "y", "z", "class_id", "confidence"])
        tree_writer.writerow(["frame", "timestamp", "x", "y", "z", "class_id", "confidence"])
        stat_writer.writerow(
            [
                "frame",
                "timestamp",
                "pose_valid",
                "pose_dt",
                "fruit_points",
                "tree_points",
                "rgb_path",
                "depth_path",
            ]
        )

        for idx, rec in enumerate(frames):
            frame = rec["frame"]
            timestamp = float(rec["t_rgb"])
            pose = poses.get(frame)
            if pose is None or not pose["valid"]:
                stat_writer.writerow([frame, f"{timestamp:.9f}", 0, "nan", 0, 0, rec["rgb_path"], rec["depth_path"]])
                continue
            valid_pose_count += 1
            rgb = cv2.imread(rec["rgb_path"], cv2.IMREAD_COLOR)
            depth = cv2.imread(rec["depth_path"], cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                stat_writer.writerow([frame, f"{timestamp:.9f}", 1, f"{pose['pose_dt']:.9f}", 0, 0, rec["rgb_path"], rec["depth_path"]])
                continue
            masks, confs = yolo_masks(rgb, model, conf_th, [0, 1])
            fruit_pts, fruit_scores = reconstruct_points(
                rgb.shape, depth, masks[0], confs[0], pose["T"], cfg, 0, args.max_fruit_points_per_frame
            )
            tree_pts, tree_scores = reconstruct_points(
                rgb.shape, depth, masks[1], confs[1], pose["T"], cfg, 1, args.max_tree_points_per_frame
            )
            fruit_pcd.write(fruit_pts, fruit_color)
            tree_pcd.write(tree_pts, tree_color)
            write_points_csv_rows(fruit_writer, frame, timestamp, fruit_pts, fruit_scores, 0)
            write_points_csv_rows(tree_writer, frame, timestamp, tree_pts, tree_scores, 1)
            fruit_point_count += len(fruit_pts)
            tree_point_count += len(tree_pts)
            processed_count += 1
            stat_writer.writerow(
                [
                    frame,
                    f"{timestamp:.9f}",
                    1,
                    f"{pose['pose_dt']:.9f}",
                    len(fruit_pts),
                    len(tree_pts),
                    rec["rgb_path"],
                    rec["depth_path"],
                ]
            )
            if args.save_overlays_every > 0 and idx % args.save_overlays_every == 0:
                color_layer = np.zeros_like(rgb)
                color_layer[masks[0] > 0] = fruit_color
                color_layer[masks[1] > 0] = tree_color
                overlay = cv2.addWeighted(rgb, 0.72, color_layer, 0.28, 0)
                cv2.imwrite(str(overlay_dir / f"{frame}_overlay.png"), overlay)
            if (idx + 1) % 100 == 0:
                print(f"processed_frames={idx + 1}/{frame_count} valid_pose={valid_pose_count} fruit_points={fruit_point_count} tree_points={tree_point_count}", flush=True)

    fruit_pcd.close()
    tree_pcd.close()

    pose_match_rate = valid_pose_count / frame_count if frame_count else 0.0
    fruit_candidate_count = ""
    fruit_instances = ""
    tree_instances = ""
    tree_yield_rows = []

    if args.run_clustering:
        scripts = Path(__file__).resolve().parent
        subprocess.check_call(
            [
                "python3",
                str(scripts / "fruit_instance_mapping.py"),
                "--config",
                args.config,
                "--points",
                str(fruit_csv_path),
                "--output-instances",
                str(out_dir / "fruit_instances.csv"),
                "--output-points",
                str(out_dir / "fruit_clustered_points.csv"),
                "--output-pcd",
                str(out_dir / "fruit_instance_map.pcd"),
            ]
        )
        subprocess.check_call(
            [
                "python3",
                str(scripts / "remove_duplicate_fruits.py"),
                "--config",
                args.config,
                "--instances",
                str(out_dir / "fruit_instances.csv"),
                "--output",
                str(out_dir / "global_fruit_instances.csv"),
            ]
        )
        subprocess.check_call(
            [
                "python3",
                str(scripts / "estimate_fruit_size.py"),
                "--clustered-points",
                str(out_dir / "fruit_clustered_points.csv"),
                "--output",
                str(out_dir / "fruit_size.csv"),
            ]
        )
        subprocess.check_call(
            [
                "python3",
                str(scripts / "tree_instance_mapping.py"),
                "--points",
                str(tree_csv_path),
                "--output-instances",
                str(out_dir / "tree_instances.csv"),
                "--output-pcd",
                str(out_dir / "tree_instance_map.pcd"),
                "--eps",
                str(args.tree_eps),
                "--min-points",
                str(args.tree_min_points),
                "--voxel-size",
                str(args.tree_voxel_size),
            ]
        )
        subprocess.check_call(
            [
                "python3",
                str(scripts / "associate_fruit_tree.py"),
                "--fruit-instances",
                str(out_dir / "global_fruit_instances.csv"),
                "--tree-instances",
                str(out_dir / "tree_instances.csv"),
                "--output",
                str(out_dir / "tree_fruit.csv"),
            ]
        )
        subprocess.check_call(
            [
                "python3",
                str(scripts / "tree_yield_estimation.py"),
                "--config",
                args.config,
                "--associations",
                str(out_dir / "tree_fruit.csv"),
                "--fruit-size",
                str(out_dir / "fruit_size.csv"),
                "--output",
                str(out_dir / "tree_yield.csv"),
            ]
        )

        fruit_candidate_count = count_data_rows(out_dir / "fruit_instances.csv")
        fruit_instances = count_data_rows(out_dir / "global_fruit_instances.csv")
        tree_instances = count_data_rows(out_dir / "tree_instances.csv")
        tree_yield_rows = read_tree_yield(out_dir / "tree_yield.csv")

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_count",
                "valid_pose_count",
                "pose_match_rate",
                "processed_frame_count",
                "fruit_point_count",
                "tree_point_count",
                "fruit_candidate_count",
                "fruit_instance_count",
                "tree_instance_count",
            ]
        )
        writer.writerow(
            [
                frame_count,
                valid_pose_count,
                f"{pose_match_rate:.6f}",
                processed_count,
                fruit_point_count,
                tree_point_count,
                fruit_candidate_count,
                fruit_instances,
                tree_instances,
            ]
        )
    if tree_yield_rows:
        with open(out_dir / "tree_yield_summary.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tree_id", "fruit_count", "estimated_weight"])
            for row in tree_yield_rows:
                writer.writerow([row.get("tree_id", ""), row.get("fruit_count", ""), row.get("estimated_mass_kg", "")])

    print(f"frame_count={frame_count}")
    print(f"valid_pose_count={valid_pose_count}")
    print(f"pose_match_rate={pose_match_rate:.6f}")
    print(f"fruit_points={fruit_point_count}")
    print(f"tree_points={tree_point_count}")
    print(f"output_dir={out_dir}")


def count_data_rows(path):
    if not Path(path).exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def read_tree_yield(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
