#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pack_rgb(r, g, b):
    return int(r) << 16 | int(g) << 8 | int(b)


def write_pcd_xyzrgb(path: Path, points: np.ndarray, colors_bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F U\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for p, c in zip(points, colors_bgr):
            rgb = pack_rgb(c[2], c[1], c[0])
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {rgb}\n")


def mask_from_yolo(rgb_bgr, model_path, conf):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for YOLO mask generation in this mode.") from exc

    model = YOLO(model_path)
    results = model.predict(rgb_bgr, conf=conf, verbose=False)
    h, w = rgb_bgr.shape[:2]
    fruit_mask = np.zeros((h, w), dtype=np.uint8)
    confidence = np.zeros((h, w), dtype=np.float32)
    if not results:
        return fruit_mask, confidence
    res = results[0]
    if res.masks is None or res.boxes is None:
        return fruit_mask, confidence
    masks = res.masks.data.detach().cpu().numpy()
    classes = res.boxes.cls.detach().cpu().numpy().astype(int)
    confs = res.boxes.conf.detach().cpu().numpy()
    for m, cls_id, score in zip(masks, classes, confs):
        if cls_id != 0:
            continue
        resized = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
        update = resized & (score >= confidence)
        fruit_mask[update] = 1
        confidence[update] = float(score)
    return fruit_mask, confidence


def mask_from_color(mask_bgr, citrus_bgr):
    target = np.array(citrus_bgr, dtype=np.int16).reshape(1, 1, 3)
    diff = np.abs(mask_bgr.astype(np.int16) - target)
    fruit_mask = (diff.sum(axis=2) < 35).astype(np.uint8)
    confidence = fruit_mask.astype(np.float32)
    return fruit_mask, confidence


def reconstruct(rgb_path, depth_path, cfg, model_path=None, mask_path=None):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise RuntimeError(f"Cannot read RGB image: {rgb_path}")
    if depth_raw is None:
        raise RuntimeError(f"Cannot read depth image: {depth_path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    h, w = rgb.shape[:2]
    if depth_raw.shape[:2] != (h, w):
        depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    if mask_path:
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise RuntimeError(f"Cannot read mask image: {mask_path}")
        if mask_bgr.shape[:2] != (h, w):
            mask_bgr = cv2.resize(mask_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
        fruit_mask, confidence = mask_from_color(mask_bgr, cfg["classes"]["citrus"]["bgr"])
    else:
        fruit_mask, confidence = mask_from_yolo(
            rgb, model_path or cfg["paths"]["yolo_model"], cfg["reconstruction"]["segmentation_conf"]
        )

    dilate_px = int(cfg["reconstruction"].get("mask_dilate_px", 0))
    if dilate_px > 0:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        fruit_mask = cv2.dilate(fruit_mask, kernel, iterations=1)

    depth = depth_raw.astype(np.float32) * float(cfg["camera"]["depth_scale"])
    valid = (
        (fruit_mask > 0)
        & np.isfinite(depth)
        & (depth >= float(cfg["camera"]["min_depth_m"]))
        & (depth <= float(cfg["camera"]["max_depth_m"]))
    )
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32), rgb, fruit_mask

    max_points = int(cfg["reconstruction"].get("max_points_per_frame", 120000))
    if len(xs) > max_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys = xs[keep], ys[keep]

    z = depth[ys, xs]
    fx, fy = float(cfg["camera"]["fx"]), float(cfg["camera"]["fy"])
    cx, cy = float(cfg["camera"]["cx"]), float(cfg["camera"]["cy"])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.column_stack([x, y, z]).astype(np.float32)
    conf = confidence[ys, xs].astype(np.float32)
    return points, conf, rgb, fruit_mask


def main():
    parser = argparse.ArgumentParser(description="Single-frame RGB-D citrus point reconstruction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--output-pcd", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--debug-mask", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    points, conf, rgb, fruit_mask = reconstruct(
        Path(args.rgb), Path(args.depth), cfg, model_path=args.model or None, mask_path=args.mask or None
    )
    citrus_bgr = np.array(cfg["classes"]["citrus"]["bgr"], dtype=np.uint8)
    colors = np.repeat(citrus_bgr.reshape(1, 3), len(points), axis=0)
    write_pcd_xyzrgb(Path(args.output_pcd), points, colors)

    ensure_dir(Path(args.output_csv).parent)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "class_id", "confidence"])
        for p, score in zip(points, conf):
            writer.writerow([f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}", 0, f"{score:.6f}"])

    if args.debug_mask:
        ensure_dir(Path(args.debug_mask).parent)
        mask_bgr = np.zeros_like(rgb)
        mask_bgr[fruit_mask > 0] = citrus_bgr
        overlay = cv2.addWeighted(rgb, 0.72, mask_bgr, 0.28, 0)
        cv2.imwrite(args.debug_mask, overlay)

    print(f"fruit_points={len(points)} pcd={args.output_pcd} csv={args.output_csv}")


if __name__ == "__main__":
    main()
