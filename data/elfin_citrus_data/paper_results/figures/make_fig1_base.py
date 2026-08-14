#!/usr/bin/env python
"""Assemble fig1 base canvas: (a) side view on top row, (b) hand-eye calib
and (c) occlusion illustration side by side on bottom row.

Only image compositing is done here; all annotations (arrows, boxes, extra
text) are added later in Photoshop. This script only draws the (a)/(b)/(c)
panel labels.
"""
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

SRC_DIR = "/home/catas/论文模板写作/我们自己的论文/所有图片/"
OUT_DIR = "/home/catas/elfin_citrus_data/paper_results/figures/"

FILES = {
    "a": "侧视图（暂未标注相机型号、安装高度、俯仰角、基座坐标系、末端坐标系、树冠工作区与预备点距离）.jpg",
    "b": "手眼标定TCP的时候.jpg",
    "c": "一张图体现无遮挡、叶子遮挡、果实遮挡.png",
}

MAX_SIDE = 1600


def load_resized(path, max_side=MAX_SIDE):
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(im), im.size


def add_label(ax, text):
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        color="black",
        family="sans-serif",
    )


def main():
    arr_a, size_a = load_resized(os.path.join(SRC_DIR, FILES["a"]))
    arr_b, size_b = load_resized(os.path.join(SRC_DIR, FILES["b"]))
    arr_c, size_c = load_resized(os.path.join(SRC_DIR, FILES["c"]))

    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[1.35, 1.0],
        width_ratios=[1.0, 1.0],
        wspace=0.10,
        hspace=0.12,
    )
    fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    ax_a.imshow(arr_a, aspect="equal")
    ax_b.imshow(arr_b, aspect="equal")
    ax_c.imshow(arr_c, aspect="equal")

    for ax in (ax_a, ax_b, ax_c):
        ax.axis("off")

    add_label(ax_a, "(a)")
    add_label(ax_b, "(b)")
    add_label(ax_c, "(c)")

    png_path = os.path.join(OUT_DIR, "fig1_platform_base.png")
    svg_path = os.path.join(OUT_DIR, "fig1_platform_base.svg")
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)

    print("figsize_inches:", fig.get_size_inches())
    print("pos_a:", ax_a.get_position())
    print("pos_b:", ax_b.get_position())
    print("pos_c:", ax_c.get_position())
    print("resized_a_px:", size_a)
    print("resized_b_px:", size_b)
    print("resized_c_px:", size_c)
    print("png_path:", png_path)
    print("svg_path:", svg_path)


if __name__ == "__main__":
    main()
