#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_fp.py — 诊断 YOLO 误检与多标牌框问题

从 200 张产出的异常样本看，主要两类问题：

  ① 误检：货车车身的图案被当成标牌（CAVA 那张）
     特征：框内区域纹理杂乱、没有清晰的圆形/矩形结构

  ② 一个框圈住多个标牌：方底板上下两个限速牌，YOLO 框住了整块
     特征：框的宽高比异常（竖长），或框内能分割出两个独立的圆形亮区

这轮先量化这两种各占多少，再决定怎么过滤。
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np

OUT = "/data/render/tsr_prod_200"


def count_inner_signs(roi):
    """在框内数有几个「圆形亮块」——超过 1 个说明框住了多个标牌。"""
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
        return 0
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 14) | 1,
                                                      max(3, h // 14) | 1))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    cnt = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area < (w * h) * 0.04:
            continue
        asp = max(bw, bh) / float(max(1, min(bw, bh)))
        if asp <= 1.8:                    # 接近方形/圆的块
            cnt += 1
    return cnt


def main():
    rows = []
    for a in sorted(glob.glob(OUT + "/annotations/*.json")):
        try:
            with open(a, encoding="utf-8") as f:
                d = json.load(f)
            o = d["objects"][0]
            src = d["source_path"]
            if not os.path.isfile(src):
                continue
            im = cv2.imread(src)
            if im is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
            H, W = im.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            roi = im[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            bw, bh = x2 - x1, y2 - y1
            n_inner = count_inner_signs(roi)
            rows.append({
                "file": d.get("image", ""),
                "cls": o.get("class", ""),
                "w": bw, "h": bh,
                "aspect": (max(bw, bh) / float(max(1, min(bw, bh)))),
                "n_inner": n_inner,
                "score": o.get("score", 0),
            })
        except Exception:
            continue

    print("共 %d 个样本" % len(rows))

    multi = [r for r in rows if r["n_inner"] >= 2]
    tall = [r for r in rows if r["aspect"] > 1.6]
    lowscore = [r for r in rows if r["score"] < 0.45]

    print()
    print("【问题分类】")
    print("  框内疑似多个标牌 (>1 个圆/方块): %d 个 (%.1f%%)"
          % (len(multi), 100.0 * len(multi) / max(len(rows), 1)))
    print("  框长宽比 > 1.6（竖长/横长，可能是多牌或畸形）: %d 个 (%.1f%%)"
          % (len(tall), 100.0 * len(tall) / max(len(rows), 1)))
    print("  置信度 < 0.45: %d 个 (%.1f%%)"
          % (len(lowscore), 100.0 * len(lowscore) / max(len(rows), 1)))

    print()
    print("长宽比分布：")
    asp = np.array([r["aspect"] for r in rows])
    for q in (50, 75, 90, 95, 99):
        print("  %2d%%: %.2f" % (q, np.percentile(asp, q)))

    print()
    print("框内标牌数分布：")
    from collections import Counter
    c = Counter(r["n_inner"] for r in rows)
    for k in sorted(c):
        print("  %d 个: %d 帧" % (k, c[k]))

    if tall:
        print()
        print("竖长框样例（应过滤）：")
        for r in sorted(tall, key=lambda x: -x["aspect"])[:6]:
            print("  %-46s %dx%d 比%.1f 内%d个 %s"
                  % (r["file"][:46], r["w"], r["h"], r["aspect"],
                     r["n_inner"], r["cls"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
