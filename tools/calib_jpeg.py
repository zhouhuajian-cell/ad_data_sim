#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校准 JPEG 质量阈值：量真实标牌区域的拉普拉斯方差分布。"""
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np


def main():
    vals = []
    for outdir in ("/data/render/tsr_prod_200", "/data/render/tsr_v7_30"):
        for a in sorted(glob.glob(outdir + "/annotations/*.json"))[:150]:
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
                patch = im[y1:y2, x1:x2]
                if patch.shape[0] < 8 or patch.shape[1] < 8:
                    continue
                g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                vals.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            except Exception:
                continue

    if not vals:
        print("没有样本")
        return 2
    v = np.array(vals)
    print("标牌区域拉普拉斯方差（%d 个样本）：" % len(v))
    for q in (5, 10, 25, 50, 75, 90, 95):
        print("  %2d%% 分位: %8.1f" % (q, np.percentile(v, q)))
    print()
    print("按当前阈值映射：")
    for lo, hi, qy, name in ((0, 120, 46, "很糊"), (120, 400, 56, "一般"),
                             (400, 900, 66, "较清晰"), (900, 1e9, 74, "高清")):
        n = ((v >= lo) & (v < hi)).sum()
        print("  lap %d~%d (%s) → quality %d，占 %.0f%%"
              % (lo, hi if hi < 1e8 else 99999, name, qy, 100.0 * n / len(v)))
    print()
    print("建议按实测分位重设阈值：")
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    print("  很糊   lap < %.0f      → quality 46" % p25)
    print("  一般   %.0f~%.0f  → quality 56" % (p25, p50))
    print("  较清晰 %.0f~%.0f  → quality 66" % (p50, p75))
    print("  高清   > %.0f       → quality 74" % p75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
