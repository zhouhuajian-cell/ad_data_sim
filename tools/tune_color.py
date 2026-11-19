#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tune_color.py — 调参验证：对比度与饱和度损失该修多少

实测（30 张）：
  亮度均值 119.5 → 125.1（+5.6，可接受）
  对比度   54.0 → 47.4（-6.6，掉了 12%）
  饱和度P90 154.6 → 114.0（-40.6，掉了 26%）

结论：不是亮度问题，是对比度和饱和度被压过头，
      导致标牌"平淡、没有红圈的鲜艳感"，看起来假。

这轮离线对比几种配置，找出最能贴近原牌色彩的一组：
  cfg A: P5~P95 + 饱和度压制 100%（当前 v4）
  cfg B: P5~P95 + 饱和度压制 50%
  cfg C: P10~P90 + 饱和度压制 100%
  cfg D: P5~P95 + 饱和度压制 0%（完全不压）
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np


def stats_of(patch):
    if patch.size == 0:
        return None
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return {
        "L_mean": float(g.mean()),
        "L_std": float(g.std()),
        "S_mean": float(hsv[:, :, 1].mean()),
        "S_p90": float(np.percentile(hsv[:, :, 1], 90)),
    }


def degrade_with_cfg(sign_bgr, mask, ref_patch, cfg):
    """离线复现 degrade_v3 的色彩部分，带可调参数。"""
    valid = mask > 127
    if valid.sum() < 20:
        return sign_bgr

    p = sign_bgr.astype(np.float32)
    bg_lab = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    as_lab = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_l = bg_lab[:, :, 0][valid]
    as_l = as_lab[:, :, 0][valid]
    lo, hi = cfg["pct"]
    p_lo_bg, p_hi_bg = np.percentile(bg_l, [lo, hi])
    p_lo_as, p_hi_as = np.percentile(as_l, [lo, hi])

    scale_l = (p_hi_bg - p_lo_bg) / (p_hi_as - p_lo_as + 1e-5)
    scale_l = float(np.clip(scale_l, 0.55, 1.6))
    as_lab[:, :, 0] = (as_lab[:, :, 0] - p_lo_as) * scale_l + p_lo_bg
    as_lab[:, :, 0] = np.clip(as_lab[:, :, 0], 0, 255)

    for ch in (1, 2):
        diff = float(np.mean(bg_lab[:, :, ch][valid]) - np.mean(as_lab[:, :, ch][valid]))
        as_lab[:, :, ch] = np.clip(as_lab[:, :, ch] + diff * 0.35, 0, 255)

    out = cv2.cvtColor(as_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # 饱和度压制：strength 1.0=按 P90 全压  0=不压
    st = cfg["sat_strength"]
    if st > 0.01:
        bg_hsv = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        cur = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_s90 = float(np.percentile(bg_hsv[:, :, 1][valid], 90))
        as_s90 = float(np.percentile(cur[:, :, 1][valid], 90))
        if as_s90 > 1e-3:
            full = min(1.0, (bg_s90 * 1.05) / as_s90)
            # 按 st 在 1.0 和 full 之间插值
            sc = 1.0 - (1.0 - full) * st
            cur[:, :, 1] = np.clip(cur[:, :, 1] * sc, 0, 255)
            out = cv2.cvtColor(cur.astype(np.uint8), cv2.COLOR_HSV2BGR)

    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg"]])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def main():
    outdir = "/data/render/tsr_v4_30"
    cfgs = {
        "A P10-90 sat100 j58": dict(pct=(10, 90), sat_strength=1.0, jpeg=58),
        "B P5-95 sat100 j58":  dict(pct=(5, 95), sat_strength=1.0, jpeg=58),
        "C P5-95 sat50 j58":   dict(pct=(5, 95), sat_strength=0.5, jpeg=58),
        "D P5-95 sat0 j58":    dict(pct=(5, 95), sat_strength=0.0, jpeg=58),
        "E P5-95 sat50 j75":   dict(pct=(5, 95), sat_strength=0.5, jpeg=75),
    }

    # 收集样本：原牌 patch + 按记录重算
    samples = []
    for a in sorted(glob.glob(outdir + "/annotations/*.json"))[:30]:
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
            pad = max(2, int((x2 - x1) * 0.10))
            po = im[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
            pc = ci[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
            if po.size == 0 or pc.size == 0 or po.shape != pc.shape:
                continue
            samples.append((po, pc, o["bbox"]))
        except Exception:
            continue

    if not samples:
        print("没有样本")
        return 2

    so_all = [stats_of(s[0]) for s in samples]
    target = {k: float(np.mean([s[k] for s in so_all])) for k in so_all[0]}
    print("原牌基准：亮度 %.1f  对比度 %.1f  饱和度 %.1f  S_P90 %.1f"
          % (target["L_mean"], target["L_std"], target["S_mean"], target["S_p90"]))
    print()

    # 对每个 cfg，用「原图贴图前后的比例」近似评估：直接对现有仿真图做二次处理不可靠，
    # 改用：把原牌 patch 当作素材，按 cfg 处理后再统计（模拟降质效果）
    print("%-24s %8s %8s %8s %8s" % ("配置", "亮度", "对比度", "饱和度", "S_P90"))
    print("-" * 62)
    for name, cfg in cfgs.items():
        vals = []
        for po, pc, bb in samples:
            # 把原牌 patch 当素材，用它的模糊版本做 ref（近似）
            sig = cv2.GaussianBlur(po, (3, 3), 0.5)
            m = np.full(po.shape[:2], 255, np.uint8)
            dg = degrade_with_cfg(sig, m, po, cfg)
            st = stats_of(dg)
            if st:
                vals.append(st)
        if not vals:
            continue
        g = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        print("%-24s %8.1f %8.1f %8.1f %8.1f"
              % (name, g["L_mean"], g["L_std"], g["S_mean"], g["S_p90"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
