#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：找出标牌尺寸超出原牌的样本，量化 grow 的合适取值。"""
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np


def red_ring_size(patch):
    """在图片块里找红色圆环，返回 (w, h) 或 None。"""
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, (0, 55, 45), (14, 255, 255))
         | cv2.inRange(hsv, (158, 55, 45), (180, 255, 255)))
    if m.sum() < 40:
        return None
    ys, xs = np.where(m > 0)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def main():
    out = "/data/render/tsr_prod_200"
    rows = []
    for a in sorted(glob.glob(out + "/annotations/*.json")):
        try:
            with open(a, encoding="utf-8") as f:
                d = json.load(f)
            o = d["objects"][0]
            if "speed_limit" not in o["class"]:
                continue
            src = d["source_path"]
            comp = os.path.join(out, d["image"])
            if not (os.path.isfile(src) and os.path.isfile(comp)):
                continue
            im = cv2.imread(src)
            ci = cv2.imread(comp)
            if im is None or ci is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
            H, W = im.shape[:2]
            sz = min(x2 - x1, y2 - y1)
            pad = int(max(sz * 0.45, 8))
            a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
            a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
            r_orig = red_ring_size(im[a0:a1, b0:b1])
            r_sim = red_ring_size(ci[a0:a1, b0:b1])
            if not r_orig or not r_sim:
                continue
            ratio = r_sim[0] / float(r_orig[0])
            rows.append((ratio, sz, r_orig[0], r_sim[0], o["class"],
                         a, (a0, b0, a1, b1)))
        except Exception:
            continue

    if not rows:
        print("没有可用样本")
        return 2

    rows.sort(key=lambda r: -r[0])
    print("共 %d 个限速牌样本" % len(rows))
    print()
    print("仿真红圈 / 原牌红圈 的比值分布：")
    ratios = np.array([r[0] for r in rows])
    for q in (5, 25, 50, 75, 90, 95, 99):
        print("  %2d%% 分位: %.2f" % (q, np.percentile(ratios, q)))
    print()
    over = (ratios > 1.25).sum()
    print("比值 > 1.25（明显偏大）: %d 个 (%.1f%%)" % (over, 100.0 * over / len(ratios)))
    print()
    print("最偏大的 8 个：")
    print("  %-6s %-6s %-8s %-8s %-8s %s" % ("比值", "框px", "原红圈", "仿真红圈", "尺寸", "类别"))
    for ratio, sz, ro, rs, cls, a, box in rows[:8]:
        print("  %-6.2f %-6d %-8d %-8d %-8d %s" % (ratio, sz, ro, rs, sz, cls))

    # 按尺寸分组看比值
    print()
    print("按标牌尺寸分组：")
    for lo, hi in ((0, 40), (40, 60), (60, 90), (90, 200)):
        grp = [r for r in rows if lo <= r[1] < hi]
        if grp:
            g = np.array([r[0] for r in grp])
            print("  %3d-%3dpx: %3d 个, 比值中位 %.2f, >1.25 占 %.0f%%"
                  % (lo, hi, len(grp), np.median(g), 100.0 * (g > 1.25).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
