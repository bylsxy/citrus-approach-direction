#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import rosbag


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stamp_to_float(msg) -> float:
    return float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs) * 1e-9


def decode_image(msg):
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = data.reshape(msg.height, msg.width, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if enc in ("mono8", "8uc1"):
        return data.reshape(msg.height, msg.width)
    if enc in ("16uc1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    if enc in ("32fc1",):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def main():
    parser = argparse.ArgumentParser(description="Extract synchronized RGB-D frames from a ROS bag.")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=3, help="Maximum synchronized frames to save. Use 0 or negative for all frames.")
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--sync-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.output)
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    ensure_dir(rgb_dir)
    ensure_dir(depth_dir)

    depth_cache = []
    rgb_msgs = []
    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=[args.depth_topic, args.rgb_topic]):
            if topic == args.depth_topic:
                depth_cache.append((stamp_to_float(msg), msg))
            elif topic == args.rgb_topic:
                rgb_msgs.append((stamp_to_float(msg), msg))

    if not rgb_msgs:
        raise RuntimeError(f"No RGB messages found on {args.rgb_topic}")
    if not depth_cache:
        raise RuntimeError(f"No depth messages found on {args.depth_topic}")

    depth_times = np.array([x[0] for x in depth_cache], dtype=np.float64)
    saved = []
    candidates = rgb_msgs[:: max(args.frame_stride, 1)]
    for idx, (t_rgb, rgb_msg) in enumerate(candidates):
        nearest = int(np.argmin(np.abs(depth_times - t_rgb)))
        t_depth, depth_msg = depth_cache[nearest]
        dt = abs(t_depth - t_rgb)
        if dt > args.sync_tolerance:
            continue
        rgb = decode_image(rgb_msg)
        depth = decode_image(depth_msg)
        name = f"frame_{len(saved):06d}"
        rgb_path = rgb_dir / f"{name}_rgb.png"
        depth_path = depth_dir / f"{name}_depth.png"
        cv2.imwrite(str(rgb_path), rgb)
        cv2.imwrite(str(depth_path), depth)
        saved.append((name, t_rgb, t_depth, dt, str(rgb_path), str(depth_path)))
        if args.max_frames > 0 and len(saved) >= args.max_frames:
            break

    index_path = out_dir / "frames_index.csv"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("frame,t_rgb,t_depth,dt_sec,rgb_path,depth_path\n")
        for row in saved:
            f.write(",".join([str(x) for x in row]) + "\n")

    print(f"Saved {len(saved)} synchronized RGB-D frames to {out_dir}")
    if not saved:
        raise RuntimeError("No synchronized frames saved; increase sync tolerance or reduce frame stride.")


if __name__ == "__main__":
    main()
