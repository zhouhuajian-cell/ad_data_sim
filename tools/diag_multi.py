#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化「框内多标牌」的样本，判断是真多牌还是误判。"""
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np

OUT = "/data/render/tsr_prod_200"


def count_inner(roi):
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
        if max(bw, bh) / float(max(1, min(bw, bh))) <= 1.8:
            cnt += 1
    return cnt


def main():
    multi = []
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
            n_in = count_inner(roi)
            if n_in >= 2:
                multi.append((n_in, src, d, o))
        except Exception:
            continue

    print("框内多标牌的样本: %d 个" % len(multi))
    tiles = []
    for n_in, src, d, o in multi[:9]:
        im = cv2.imread(src)
        comp = cv2.imread(os.path.join(OUT, d["image"]))
        x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
        H, W = im.shape[:2]
        pad = int(max((x2 - x1) * 1.5, 24))
        a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
        a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
        S = 190
        ca = cv2.resize(im[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)
        cb = cv2.resize(comp[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)
        # 在图上画出检测框
        def draw(img, ox, oy, scale, col):
            cv2.rectangle(img,
                          (int((x1 - ox) * scale), int((y1 - oy) * scale)),
                          (int((x2 - ox) * scale), int((y2 - oy) * scale)),
                          col, 1)
            return img
        sc = S / float(max(1, b1 - b0))
        ca = draw(ca, b0, a0, sc, (0, 255, 0))
        cb = draw(cb, b0, a0, sc, (0, 200, 255))
        cv2.putText(ca, "O n=%d" % n_in, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1)
        cv2.putText(cb, o["class"][:12], (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        tiles.append(np.hstack([ca, np.full((S, 3, 3), 255, np.uint8),
                                cb, np.full((S, 10, 3), 40, np.uint8)]))

    if tiles:
        while len(tiles) % 3:
            tiles.append(np.full((190, 393, 3), 40, np.uint8))
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        cv2.imwrite("/tmp/multisign.jpg", np.vstack(rows))
        print("可视化 /tmp/multisign.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
