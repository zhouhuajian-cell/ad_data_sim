#!/usr/bin/env python3
"""解析 GND seg_lane 分割掩码的类别定义（只读网盘）。"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
import data_guard
from data_guard import guard_all_mnt_ops

guard_all_mnt_ops()

D = ("/mnt/done_data_new/31_ex_GND/05_maxieye/02_HS5-001/"
     "20260629/1782705797.755")


def main():
    seg_dir = os.path.join(D, "seg_lane")
    img_dir = os.path.join(D, "img_ori")

    segs = sorted(os.listdir(seg_dir))
    imgs = sorted(os.listdir(img_dir))
    print("seg 数: %d   img 数: %d" % (len(segs), len(imgs)))

    # 取一张对齐的
    stem = os.path.splitext(segs[10])[0]
    sp = os.path.join(seg_dir, stem + ".png")
    ip = os.path.join(img_dir, stem + ".jpg")

    seg = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(ip) if os.path.isfile(ip) else None

    print("\n=== seg ===")
    print("  shape:", seg.shape)
    if img is not None:
        print("  img shape:", img.shape)

    # 颜色统计
    flat = seg.reshape(-1, 3)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)
    print("\n=== 调色板（按像素数排序）===")
    print("  %-18s %-12s %s" % ("BGR", "像素数", "占比"))
    for i in order:
        c = uniq[i]
        n = counts[i]
        pct = n / len(flat) * 100
        print("  %-18s %-12d %.2f%%"
              % (str(tuple(int(v) for v in c)), n, pct))

    # 保存可视化
    out = "/tmp/gnd_probe"
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, "seg.png"), seg)
    if img is not None:
        cv2.imwrite(os.path.join(out, "img.jpg"), img)
        # 叠加
        ov = cv2.addWeighted(img, 0.6, seg, 0.4, 0)
        cv2.imwrite(os.path.join(out, "overlay.jpg"), ov)

    # 每个颜色的分布位置
    print("\n=== 各颜色位置分布 ===")
    h, w = seg.shape[:2]
    for i in order:
        c = uniq[i]
        m = np.all(seg == c, axis=2)
        if m.sum() < 100:
            continue
        ys, xs = np.where(m)
        print("  %-18s y=[%d,%d] x=[%d,%d]  占比 %.2f%%"
              % (str(tuple(int(v) for v in c)),
                 ys.min(), ys.max(), xs.min(), xs.max(),
                 m.mean() * 100))

    print("\n产出: /tmp/gnd_probe/")


if __name__ == "__main__":
    main()
