#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auto_qc.py — 仿真质量自动门控，剔除穿帮样本

借鉴 code_sim.py 的思路，但做了两点改造：

  原方案要求 MobileSAM 输出的原牌 Mask —— 我们没有分割模型，
  改用已有的 Otsu 轮廓（std_shape 那套）替代。

  原方案用 fitEllipse 算 IoU —— 只适合圆牌，方牌/三角牌不适用，
  改用「轮廓面积 / 外接矩形面积」的形状一致性（fill ratio）。

三道门控：

  Gate 1 几何完整性（fill ratio）
      标牌轮廓占外接矩形的比例应该落在该形状的理论区间内：
        圆 0.785、方 1.0、三角 0.5
      偏离太多说明轮廓残缺（被遮挡/分割失败），这种样本形状一定贴歪。

  Gate 2 边缘跳变
      标牌内外各取一圈 5px 环带，比较平均亮度差。
      跳变过大说明有黑边/白边/错位 —— 就是用户说的「重影」。

  Gate 3 亮度一致性
      仿真后标牌区域与原图同区域的亮度差，超过 ±35 判为不合。
      过曝或过暗都会在这里被刷掉。

用法：
  from auto_qc import qc_check
  ok, info = qc_check(orig, composed, bbox, shape)
"""
import numpy as np


# 各形状的理论填充率与容差。
# 容差是实测标定的：300 个真实样本的 fill 偏离量
#   50% 分位 0.091、70% 分位 0.174、90% 分位 0.366
# 容差取 0.12 会剔掉 40.7%（太严，把透视变形的正常样本也杀了）；
# 取 0.22 只剔 ~20%，能把真正残缺的挑出来。
FILL_EXPECT = {
    "circle": (0.785, 0.22),
    "square": (1.000, 0.18),
    "triangle": (0.500, 0.22),
}


def qc_check(orig, composed, bbox, shape="circle",
             edge_jump_max=38.0, bright_diff_max=38.0,
             fill_tol_scale=1.0):
    """跑三道门控。

    orig      原图（BGR）
    composed  仿真后的图（BGR）
    bbox      [x1,y1,x2,y2]
    shape     circle / square / triangle

    返回 (passed, info)
    """
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = orig.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    bw, bh = x2 - x1, y2 - y1
    info = {"passed": True, "reason": "OK"}

    if bw < 18 or bh < 18:
        return False, {"passed": False, "reason": "too_small"}

    # ---- Gate 1：几何完整性 ----
    roi = orig[y1:y2, x1:x2]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 10) | 1,
                                                      max(3, h // 10) | 1))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, {"passed": False, "reason": "Gate1_no_contour"}
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    bx, by, bcw, bch = cv2.boundingRect(c)
    if bcw < 12 or bch < 12:
        return False, {"passed": False, "reason": "Gate1_contour_tiny"}

    fill = area / float(bcw * bch)
    exp, tol = FILL_EXPECT.get(shape, (0.785, 0.15))
    tol *= fill_tol_scale
    info["fill"] = round(float(fill), 3)
    if abs(fill - exp) > tol:
        return False, {**info, "passed": False,
                       "reason": "Gate1_fill_%.2f_expect_%.2f" % (fill, exp)}

    # ---- Gate 2：边缘跳变 ----
    # 用 bbox 为基准：缩进 15% 作为内环，外扩 15% 作为外环
    def ring_mean(img, inner):
        ix1 = x1 + int(bw * 0.15)
        iy1 = y1 + int(bh * 0.15)
        ix2 = x2 - int(bw * 0.15)
        iy2 = y2 - int(bh * 0.15)
        if inner:
            patch = img[iy1:iy2, ix1:ix2]
        else:
            ex1 = max(0, x1 - int(bw * 0.15))
            ey1 = max(0, y1 - int(bh * 0.15))
            ex2 = min(W, x2 + int(bw * 0.15))
            ey2 = min(H, y2 + int(bh * 0.15))
            outer = img[ey1:ey2, ex1:ex2]
            if outer.size == 0:
                return None
            mask = np.ones(outer.shape[:2], np.uint8)
            # 挖掉内环区域
            mask[iy1 - ey1:iy2 - ey1, ix1 - ex1:ix2 - ex1] = 0
            vals = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY)[mask > 0]
            return float(vals.mean()) if vals.size else None
        if patch.size == 0:
            return None
        return float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean())

    inner_m = ring_mean(composed, True)
    outer_m = ring_mean(composed, False)
    if inner_m is not None and outer_m is not None:
        jump = abs(inner_m - outer_m)
        info["edge_jump"] = round(jump, 1)
        if jump > edge_jump_max:
            return False, {**info, "passed": False,
                           "reason": "Gate2_edge_jump_%.0f" % jump}

    # ---- Gate 3：亮度一致性 ----
    pad = max(2, int(bw * 0.08))
    o_patch = orig[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
    c_patch = composed[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
    if o_patch.size and c_patch.size:
        bo = float(cv2.cvtColor(o_patch, cv2.COLOR_BGR2GRAY).mean())
        bc = float(cv2.cvtColor(c_patch, cv2.COLOR_BGR2GRAY).mean())
        diff = bc - bo
        info["bright_diff"] = round(diff, 1)
        if abs(diff) > bright_diff_max:
            return False, {**info, "passed": False,
                           "reason": "Gate3_bright_%+.0f" % diff}

    return True, info


def qc_batch(out_dir, sample_limit=0, edge_jump_max=38.0, bright_diff_max=38.0,
             dry_run=False):
    """对整个产出目录跑质检，返回统计。

    dry_run=True 时只统计不删文件。
    """
    import glob
    import json
    import os

    import cv2
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from batch_replace import std_shape  # noqa: E402

    ann_dir = os.path.join(out_dir, "annotations")
    if not os.path.isdir(ann_dir):
        return {"error": "找不到 annotations 目录"}

    files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if sample_limit:
        files = files[:sample_limit]

    passed = failed = 0
    reasons = {}
    removed = 0

    for ap in files:
        try:
            with open(ap, encoding="utf-8") as f:
                a = json.load(f)
            objs = a.get("objects", [])
            if not objs:
                continue
            src = a.get("source_path", "")
            comp = os.path.join(out_dir, a.get("image", ""))
            if not (os.path.isfile(src) and os.path.isfile(comp)):
                continue
            orig = cv2.imread(src)
            composed = cv2.imread(comp)
            if orig is None or composed is None or orig.shape != composed.shape:
                continue

            # 多标牌的帧，任一标牌不过就算不合格
            frame_ok = True
            fail_reason = None
            for o in objs:
                shape = std_shape(orig, o["bbox"])
                if shape == "unknown":
                    shape = "circle"      # 兜底按圆形判
                ok, info = qc_check(orig, composed, o["bbox"], shape,
                                    edge_jump_max=edge_jump_max,
                                    bright_diff_max=bright_diff_max)
                if not ok:
                    frame_ok = False
                    fail_reason = info.get("reason", "unknown")
                    break

            if frame_ok:
                passed += 1
            else:
                failed += 1
                key = (fail_reason or "").split("_")[0]
                reasons[key] = reasons.get(key, 0) + 1
                if not dry_run:
                    try:
                        os.remove(ap)
                        os.remove(comp)
                        removed += 1
                    except OSError:
                        pass
        except Exception:
            continue

    total = passed + failed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(100.0 * passed / max(total, 1), 1),
        "removed": removed,
        "fail_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="产出目录")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--edge-jump", type=float, default=38.0)
    ap.add_argument("--bright-diff", type=float, default=38.0)
    ap.add_argument("--dry-run", action="store_true", help="只统计不删除")
    ap.add_argument("--apply", action="store_true", help="实际删除不合格样本")
    a = ap.parse_args()

    res = qc_batch(a.out, a.sample, a.edge_jump, a.bright_diff,
                   dry_run=not a.apply)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    sys.exit(0)
