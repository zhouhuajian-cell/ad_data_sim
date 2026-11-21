#!/usr/bin/env python3
"""GND 语义分割训练数据准备。

输入：/data/test_data/gnd_train/<veh_date>/<clip>/
  img_ori/      原图
  seg_lane/     语义掩码（多类颜色）
  obj_mask/     实例掩码（uint16）
  freespace/    可行驶区域

输出：/data/train/gnd_seg/
  images/train/  images/val/
  labels/train/  labels/val/    （单通道索引 PNG）
  classes.json
"""
import glob
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_guard import guard_all_mnt_ops

guard_all_mnt_ops()

SRC = "/data/test_data/gnd_train"
OUT = "/data/train/gnd_seg"

# seg_lane 颜色 → 类别索引
PALETTE = {
    (0, 0, 0): 0,            # 背景
    (0, 255, 0): 1,          # 边线/实线
    (232, 35, 244): 2,       # 内部虚线
    (153, 153, 190): 3,      # 远处/细线
}
CLASSES = ["background", "edge_solid", "lane_dash", "far_lane"]


def seg_to_index(seg_path):
    """seg_lane RGB → 单通道类别索引。"""
    s = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if s is None:
        return None
    if s.ndim == 2:
        s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)

    idx = np.zeros(s.shape[:2], np.uint8)
    for color, k in PALETTE.items():
        m = np.all(s == np.array(color, np.uint8), axis=2)
        idx[m] = k
    return idx


def main():
    clips = []
    for veh_date in sorted(os.listdir(SRC)):
        vp = os.path.join(SRC, veh_date)
        if not os.path.isdir(vp):
            continue
        for clip in sorted(os.listdir(vp)):
            cp = os.path.join(vp, clip)
            if not os.path.isdir(os.path.join(cp, "seg_lane")):
                continue
            clips.append(cp)

    print("clip: %d" % len(clips))
    if not clips:
        raise SystemExit("无数据")

    # 按 clip 划分
    random.seed(42)
    random.shuffle(clips)
    n_val = max(1, int(len(clips) * 0.2))
    val_set = set(clips[:n_val])

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    stats = {"train": 0, "val": 0}
    cls_px = {}

    for cp in clips:
        split = "val" if cp in val_set else "train"
        img_dir = os.path.join(cp, "img_ori")
        seg_dir = os.path.join(cp, "seg_lane")
        if not os.path.isdir(img_dir):
            continue

        imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        for fn in imgs:
            stem = os.path.splitext(fn)[0]
            seg_p = os.path.join(seg_dir, stem + ".png")
            if not os.path.isfile(seg_p):
                continue

            idx = seg_to_index(seg_p)
            if idx is None:
                continue
            # 必须真的有线（非全背景）
            if idx.max() == 0:
                continue

            ban = os.path.basename(os.path.dirname(cp)) + "_" + \
                  os.path.basename(cp).replace(".", "_") + "_" + \
                  stem.replace(".", "_")

            shutil.copy2(os.path.join(img_dir, fn),
                         os.path.join(OUT, "images", split, ban + ".jpg"))
            cv2.imwrite(os.path.join(OUT, "labels", split, ban + ".png"),
                        idx)
            stats[split] += 1

            u, c = np.unique(idx, return_counts=True)
            for k, n in zip(u, c):
                cls_px[int(k)] = cls_px.get(int(k), 0) + int(n)

    # classes.json
    with open(os.path.join(OUT, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({
            "classes": CLASSES,
            "palette": {str(k): v for k, v in PALETTE.items()},
            "n_train": stats["train"],
            "n_val": stats["val"],
            "pixel_stats": cls_px,
            "source": "seg_lane (green=edge, pink=dash)",
        }, f, ensure_ascii=False, indent=1)

    print("\n=== 完成 ===")
    print("训练: %d  验证: %d" % (stats["train"], stats["val"]))
    tot = sum(cls_px.values())
    print("\n类别像素分布:")
    for i, c in enumerate(CLASSES):
        n = cls_px.get(i, 0)
        print("  %-14s %12d  %6.3f%%" % (c, n, n / max(tot, 1) * 100))
    print("\n输出:", OUT)


if __name__ == "__main__":
    main()
