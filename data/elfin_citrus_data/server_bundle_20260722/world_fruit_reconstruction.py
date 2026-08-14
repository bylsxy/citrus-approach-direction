#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pack_rgb(r, g, b):
    return int(r) << 16 | int(g) << 8 | int(b)


def write_pcd(path, points, colors_bgr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        for p, c in zip(points, colors_bgr):
            rgb = pack_rgb(c[2], c[1], c[0])
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {rgb}\n")


def read_frame_index(path):
    with open(path, "r", encoding="utf-8") as f:
        return {row["frame"]: row for row in csv.DictReader(f)}


def read_poses(path):
    poses = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = row["frame"]
            T = np.array([float(row[f"T{i}{j}"]) for i in range(4) for j in range(4)], dtype=np.float64).reshape(4, 4)
            poses[frame] = {"T": T, "valid": int(row["pose_valid"]) == 1, "timestamp": float(row["timestamp"])}
    return poses


def yolo_masks(rgb_bgr, model, class_ids, conf):
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
        if int(cls_id) not in masks:
            continue
        resized = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
        update = resized & (score >= confs[int(cls_id)])
        masks[int(cls_id)][update] = 1
        confs[int(cls_id)][update] = float(score)
    return masks, confs


def reconstruct_class(rgb_path, depth_path, T_wc, cfg, mask, conf, class_id, max_points):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth_raw is None:
        raise RuntimeError(f"Cannot read RGB/depth: {rgb_path} {depth_path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    h, w = rgb.shape[:2]
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
    if len(xs) > max_points:
        rng = np.random.default_rng(42 + int(class_id))
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
    return pw, conf[ys, xs].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct semantic RGB-D points in VINS world frame.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames-index", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames", nargs="*", default=[])
    parser.add_argument("--class-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--output-pcd", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--debug-dir", default="")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = load_yaml(args.config)
    frame_index = read_frame_index(args.frames_index)
    poses = read_poses(args.poses)
    frames = args.frames or sorted(frame_index.keys())
    model = YOLO(args.model)
    max_points = int(cfg["reconstruction"].get("max_points_per_frame", 120000))
    conf_th = float(cfg["reconstruction"].get("segmentation_conf", 0.25))

    all_points = []
    all_colors = []
    rows = []
    class_colors = {0: cfg["classes"]["citrus"]["bgr"], 1: cfg["classes"]["tree"]["bgr"]}

    for frame in frames:
        if frame not in frame_index or frame not in poses or not poses[frame]["valid"]:
            continue
        rec = frame_index[frame]
        rgb_path = Path(rec["rgb_path"])
        depth_path = Path(rec["depth_path"])
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        masks, confs = yolo_masks(rgb, model, args.class_ids, conf_th)
        if args.debug_dir:
            Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
            overlay = rgb.copy()
            color_layer = np.zeros_like(rgb)
            for cid in args.class_ids:
                color_layer[masks[cid] > 0] = np.array(class_colors.get(cid, [255, 255, 255]), dtype=np.uint8)
            overlay = cv2.addWeighted(overlay, 0.72, color_layer, 0.28, 0)
            cv2.imwrite(str(Path(args.debug_dir) / f"{frame}_semantic_overlay.png"), overlay)
        for cid in args.class_ids:
            pts, scores = reconstruct_class(rgb_path, depth_path, poses[frame]["T"], cfg, masks[cid], confs[cid], cid, max_points)
            if len(pts) == 0:
                continue
            color = np.array(class_colors.get(cid, [255, 255, 255]), dtype=np.uint8)
            all_points.append(pts)
            all_colors.append(np.repeat(color.reshape(1, 3), len(pts), axis=0))
            for p, score in zip(pts, scores):
                rows.append([frame, rec["t_rgb"], f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}", cid, f"{score:.6f}"])

    points = np.vstack(all_points) if all_points else np.empty((0, 3), dtype=np.float32)
    colors = np.vstack(all_colors) if all_colors else np.empty((0, 3), dtype=np.uint8)
    write_pcd(args.output_pcd, points, colors)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp", "x", "y", "z", "class_id", "confidence"])
        writer.writerows(rows)
    print(f"world_points={len(points)} rows={len(rows)} output={args.output_pcd}")


if __name__ == "__main__":
    main()
