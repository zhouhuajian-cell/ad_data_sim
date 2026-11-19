#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_color.py — 量化对比仿真标牌与原牌的色彩/亮度差异

用户反馈：「形状差不多了，但色彩差点，亮度上低点，还是失真」
这轮测三件事：
  亮度   均值 / 中位 / P10 / P90
  饱和度 均值 / P90
  对比度 标准差
分别对「原牌区域」和「仿真标牌区域」统计，看差在哪。
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np


def stats_of(patch, mask=None):
    """返回一块区域的亮度/饱和度/对比度统计。"""
    if patch.size == 0:
        return None
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if mask is not None and mask.size == g.size and mask.sum() > 20:
        gv = g[mask > 127]
        sv = hsv[:, :, 1][mask > 127]
    else:
        gv = g.reshape(-1)
        sv = hsv[:, :, 1].reshape(-1)
    if gv.size < 20:
        return None
    return {
        "L_mean": float(gv.mean()),
        "L_med": float(np.median(gv)),
        "L_p10": float(np.percentile(gv, 10)),
        "L_p90": float(np.percentile(gv, 90)),
        "L_std": float(gv.std()),
        "S_mean": float(sv.mean()),
        "S_p90": float(np.percentile(sv, 90)),
    }


def main(outdir):
    rows = []
    for a in sorted(glob.glob(outdir + "/annotations/*.json")):
        try:
            with open(a, encoding="utf-8") as f:
                d = json.load(f)
            o = d["objects"][0]
            src = d["source_path"]
            comp = os.path.join(outdir, d["image"])
            if not (os.path.isfile(src) and os.path.isfile(comp)):
                continue
            im = cv2.imread(src)
            ci = cv2.imread(comp)
            if im is None or ci is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
            H, W = im.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            # 内缩 10% 避开边缘
            pad = max(2, int((x2 - x1) * 0.10))
            po = im[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
            pc = ci[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
            so = stats_of(po)
            sc = stats_of(pc)
            if not so or not sc:
                continue
            rows.append((so, sc, o["class"], min(x2 - x1, y2 - y1)))
        except Exception:
            continue

    if not rows:
        print("没有样本")
        return 2

    print("样本数: %d（%s）" % (len(rows), outdir))
    keys = ["L_mean", "L_med", "L_p10", "L_p90", "L_std", "S_mean", "S_p90"]
    names = ["亮度均值", "亮度中位", "亮度P10", "亮度P90", "对比度", "饱和度均值", "饱和度P90"]
    print()
    print("  %-12s %10s %10s %10s" % ("指标", "原牌", "仿真", "差值"))
    print("  " + "-" * 46)
    for k, nm in zip(keys, names):
        o_vals = [r[0][k] for r in rows]
        c_vals = [r[1][k] for r in rows]
        mo = float(np.mean(o_vals))
        mc = float(np.mean(c_vals))
        print("  %-12s %10.1f %10.1f %+10.1f" % (nm, mo, mc, mc - mo))

    # 正负统计
    print()
    d_l = np.array([r[1]["L_mean"] - r[0]["L_mean"] for r in rows])
    d_s = np.array([r[1]["S_mean"] - r[0]["S_mean"] for r in rows])
    print("  亮度：偏亮的 %d 个 (%.0f%%)，偏暗的 %d 个，差值范围 %+.0f ~ %+.0f"
          % ((d_l > 5).sum(), 100.0 * (d_l > 5).mean(),
             (d_l < -5).sum(), d_l.min(), d_l.max()))
    print("  饱和：偏高的 %d 个 (%.0f%%)，偏低的 %d 个，差值范围 %+.0f ~ %+.0f"
          % ((d_s > 5).sum(), 100.0 * (d_s > 5).mean(),
             (d_s < -5).sum(), d_s.min(), d_s.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/data/render/tsr_v4_30"))
