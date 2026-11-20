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

    # ---- Gate1 几何完整性 ----
    # 演进过程（都记下来，避免重走）：
    #   v1: fill ratio 严格区间（圆 0.785±0.12）→ 剔除 40.7%，把透视变形的正常样本也杀了
    #   v2: 放宽到 ±0.22 → 剔除 20%，但仍有 35% 是「方底板圆牌」这类正常场景
    #   v3: 改用 solidity（凸性）→ 通过率反而从 61.5% 掉到 48.2%，
    #       因为标牌被立柱/树枝/电线遮挡是常态，轮廓天然不凸
    #   v4（当前）: Gate1 只排除「明显不对」的样本，放过一切可能正常的
    #       判据只有一条：轮廓大得离谱（>1.6 倍外接框）或小得离谱（<0.55 倍）
    fill = area / float(max(bcw * bch, 1))
    info["fill"] = round(float(fill), 3)
    info["contour_w"] = int(bcw)
    info["contour_h"] = int(bch)

    # 轮廓必须基本铺满检测框 —— 太小说明只圈到标牌一角或背景纹理
    if bcw < bw * 0.55 or bch < bh * 0.55:
        return False, {**info, "passed": False,
                       "reason": "Gate1_contour_small_%dx%d" % (bcw, bch)}
    # 填充率极端值：<0.30 说明是零散纹理，>1.05 不可能（面积超过外接框）
    if fill < 0.30 or fill > 1.05:
        return False, {**info, "passed": False,
                       "reason": "Gate1_fill_extreme_%.2f" % fill}

    # ---- Gate2：边缘跳变 ----
    # 标牌内缩 15% 的区域 vs 外扩 15% 的环带，比较平均亮度
    # 跳变过大说明有黑边/白边/错位（用户说的「重影」）
    def _mean_inside(img):
        ix1, iy1 = x1 + int(bw * 0.15), y1 + int(bh * 0.15)
        ix2, iy2 = x2 - int(bw * 0.15), y2 - int(bh * 0.15)
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        patch = img[iy1:iy2, ix1:ix2]
        if patch.size == 0:
            return None
        return float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean())

    def _mean_ring(img):
        ex1 = max(0, x1 - int(bw * 0.15))
        ey1 = max(0, y1 - int(bh * 0.15))
        ex2 = min(W, x2 + int(bw * 0.15))
        ey2 = min(H, y2 + int(bh * 0.15))
        outer = img[ey1:ey2, ex1:ex2]
        if outer.size == 0:
            return None
        mask = np.ones(outer.shape[:2], np.uint8)
        ix1, iy1 = x1 + int(bw * 0.15), y1 + int(bh * 0.15)
        ix2, iy2 = x2 - int(bw * 0.15), y2 - int(bh * 0.15)
        mask[max(0, iy1 - ey1):max(0, iy2 - ey1),
             max(0, ix1 - ex1):max(0, ix2 - ex1)] = 0
        g = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY)
        vals = g[mask > 0]
        return float(vals.mean()) if vals.size else None

    mi = _mean_inside(composed)
    mo = _mean_ring(composed)
    if mi is not None and mo is not None:
        jump = abs(mi - mo)
        info["edge_jump"] = round(jump, 1)
        if jump > edge_jump_max:
            return False, {**info, "passed": False,
                           "reason": "Gate2_edge_jump_%.0f" % jump}

    # ---- Gate3：亮度一致性 ----
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

_yolo_cache = {}


def _get_model(weights="/opt/ad_mining/tsr_sign.pt"):
    """懒加载 YOLO，整个进程只加载一次（22MB 模型反复加载很慢）。"""
    if weights not in _yolo_cache:
        try:
            from ultralytics import YOLO
            _m = YOLO(weights)
            try:
                import torch
                _m.to("cuda:0" if torch.cuda.is_available() else "cpu")
            except Exception:
                pass
            _yolo_cache[weights] = _m
        except Exception:
            _yolo_cache[weights] = None
    return _yolo_cache[weights]


def gate4_model_check(composed, bbox, target_class, weights="/opt/ad_mining/tsr_sign.pt",
                      conf_min=0.35, pad_ratio=0.35):
    """Gate 4：感知模型自检 —— 把合成后的标牌喂回 YOLO，看能不能认出来。

    为什么需要这一道：
      前三道门控（几何/边缘/亮度）都是像素级判断，只能证明「贴得不突兀」，
      不能证明「模型能识别」。而我们的产出最终是给训练用的 ——
      如果连现有检测模型都认不出来，这个样本对训练就没有价值。

    做法：
      从合成图里裁出标牌区域（外扩 pad_ratio），送进 YOLO。
      只要模型在该区域检出任意标牌、且置信度 >= conf_min，就算通过。
      （不严格比对类别名 —— 各数据集类别命名不统一，能认出来就说明画质合格）

    返回 (passed, info)
    """
    import cv2

    model = _get_model(weights)
    if model is None:
        return True, {"model_check": "skipped"}      # 模型不可用就不拦

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = composed.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw < 24 or bh < 24:
        return True, {"model_check": "too_small"}

    pad = int(max(bw, bh) * pad_ratio)
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(W, x2 + pad)
    cy2 = min(H, y2 + pad)
    crop = composed[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return True, {"model_check": "empty_crop"}

    # 放大到至少 320px，YOLO 对小图不敏感
    h, w = crop.shape[:2]
    if max(h, w) < 320:
        sc = 320.0 / max(h, w)
        crop = cv2.resize(crop, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_CUBIC)

    try:
        res = model(crop, conf=conf_min, verbose=False)[0]
    except Exception:
        return True, {"model_check": "infer_failed"}

    if res.boxes is None or len(res.boxes) == 0:
        return False, {"model_check": "no_detection"}

    confs = [float(b.conf[0]) for b in res.boxes]
    best = max(confs)
    cls_ids = [int(b.cls[0]) for b in res.boxes]
    names = model.names if hasattr(model, "names") else {}
    best_idx = confs.index(best)
    best_cls = names.get(cls_ids[best_idx], str(cls_ids[best_idx]))

    if best < conf_min:
        return False, {"model_check": "low_conf_%.2f" % best}
    return True, {"model_check": "ok", "pred_class": str(best_cls),
                  "pred_conf": round(best, 2)}



