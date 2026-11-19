#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 SD 精修效果：挑三张不同尺寸的已有产出，对比精修前后。"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np

from sd_refine import make_sign_mask, refine, strength_for_size


def main():
    anns = sorted(glob.glob("/data/render/tsr_100/annotations/*.json"))
    tests = []
    for a in anns:
        try:
            with open(a, encoding="utf-8") as f:
                d = json.load(f)
            o = d["objects"][0]
            bb = o["bbox"]
            sz = min(bb[2] - bb[0], bb[3] - bb[1])
            tests.append((a, d, o, sz))
        except Exception:
            continue
    if not tests:
        print("没有可用样本")
        return 2

    tests.sort(key=lambda x: x[3])
    n = len(tests)
    picked = [tests[max(0, n // 10)], tests[n // 2], tests[min(n - 1, n - 1 - n // 10)]]
    print("选中 %d 张，尺寸: %s" % (len(picked), [int(t[3]) for t in picked]))

    results = []
    for a, d, o, sz in picked:
        comp_path = os.path.join("/data/render/tsr_100", d["image"])
        im = cv2.imread(comp_path)
        if im is None:
            continue
        mask = make_sign_mask(im.shape, o["bbox"])
        st = strength_for_size(sz)
        t0 = time.time()
        out, info = refine(im, mask, strength=st, model_dir="/data/models/sd15-inpaint")
        dt = time.time() - t0
        ok = info.get("ok", False)
        print("  %-24s %3dpx strength=%.2f  %.1fs  %s"
              % (o["class"][:24], sz, st, dt, "OK" if ok else info.get("reason")))
        if not ok:
            continue

        bb = o["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bb]
        pad = max(10, int((x2 - x1) * 0.6))
        H, W = im.shape[:2]
        a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
        a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
        ca = im[a0:a1, b0:b1]
        cb = out[a0:a1, b0:b1]
        S = 220
        ca = cv2.resize(ca, (S, S), interpolation=cv2.INTER_NEAREST)
        cb = cv2.resize(cb, (S, S), interpolation=cv2.INTER_NEAREST)
        cv2.putText(ca, "BEFORE", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(cb, "SD %.2f" % st, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        results.append(np.hstack([ca, np.full((S, 2, 3), 255, np.uint8),
                                  cb, np.full((S, 10, 3), 40, np.uint8)]))

    if results:
        mw = max(r.shape[1] for r in results)
        results = [np.hstack([r, np.full((220, mw - r.shape[1], 3), 40, np.uint8)])
                   for r in results]
        out_img = np.vstack(results)
        cv2.imwrite("/tmp/sd_compare.jpg", out_img)
        print("对比图 /tmp/sd_compare.jpg %s" % (out_img.shape,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
