#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""找出比值异常的样本，输出原始画面，确认到底是"贴太大"还是"原牌不可测"。"""
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np

OUT = "/data/render/tsr_prod_200"


def red_ring_size(patch):
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
    rows = []
    for a in sorted(glob.glob(OUT + "/annotations/*.json")):
        try:
            with open(a, encoding="utf-8") as f:
                d = json.load(f)
            o = d["objects"][0]
            if "speed_limit" not in o["class"]:
                continue
            src, comp = d["source_path"], os.path.join(OUT, d["image"])
            if not (os.path.isfile(src) and os.path.isfile(comp)):
                continue
            im, ci = cv2.imread(src), cv2.imread(comp)
            if im is None or ci is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
            H, W = im.shape[:2]
            sz = min(x2 - x1, y2 - y1)
            pad = int(max(sz * 0.45, 8))
            a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
            a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
            ro = red_ring_size(im[a0:a1, b0:b1])
            rs = red_ring_size(ci[a0:a1, b0:b1])
            if not ro or not rs:
                continue
            rows.append((rs[0] / float(ro[0]), sz, ro[0], rs[0], o["class"],
                         src, comp, o["bbox"]))
        except Exception:
            continue

    rows.sort(key=lambda r: -r[0])
    bad = [r for r in rows if r[0] > 1.25]
    print("比值 > 1.25 的样本: %d 个" % len(bad))

    tiles = []
    for ratio, sz, ro, rs, cls, src, comp, bb in bad[:8]:
        im, ci = cv2.imread(src), cv2.imread(comp)
        x1, y1, x2, y2 = [int(v) for v in bb]
        H, W = im.shape[:2]
        pad = int(max(sz * 1.2, 20))       # 看更大范围，确认原牌到底是什么
        a0, b0 = max(0, x1 - pad), max(0, x1 - pad)
        a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
        a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
        S = 200
        ca = cv2.resize(im[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)
        cb = cv2.resize(ci[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)
        cv2.putText(ca, "O %dpx" % ro, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(cb, "S %dpx x%.1f" % (rs, ratio), (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 255), 1)
        tiles.append(np.hstack([ca, np.full((S, 3, 3), 255, np.uint8),
                                cb, np.full((S, 10, 3), 40, np.uint8)]))
        print("  比值 %.1f  框%d  原红圈%d  仿真红圈%d  %s" % (ratio, sz, ro, rs, cls))

    if tiles:
        rows_img = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        mw = max(r.shape[1] for r in rows_img)
        rows_img = [np.hstack([r, np.full((200, mw - r.shape[1], 3), 40, np.uint8)])
                    for r in rows_img]
        cv2.imwrite("/tmp/badsize.jpg", np.vstack(rows_img))
        print("\n对比图 /tmp/badsize.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
