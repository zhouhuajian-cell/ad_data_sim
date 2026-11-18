#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_occlusion.py — 保留前景遮挡 + 投射阴影，解决「硬贴」感

问题（实测）：
  直接把新标牌贴到检测框里，会把原本挡在标牌前面的东西一起盖掉 ——
  实测一块挂在立柱旁的标牌，合成后立柱消失，整张图立刻穿帮。
  另外标牌没有影子，像浮在墙面上。

本模块做两件事：

一、保留前景遮挡
  思路：原标牌的「可见区域」里，比标牌本体明显暗的连通块，通常是
  挡在前面的物体（立柱、树枝、电线）。贴合时把这些像素抠掉，
  让原图透出来，新标牌看起来就是「被挡住」的。
  判定：在检测框内做 Otsu，得到标牌本体掩码；本体掩码内的暗色连通块
       （亮度 < 本体均值的某比例、且面积够大）判为前景遮挡。

二、投射阴影
  思路：真实标牌会在背后的墙面/路面上留下影子。
  用估计出的光照方向，从标牌下沿/侧沿向外画一个渐隐的暗区。
  光照方向从原标牌区域的亮度梯度推（亮面朝向光源）。
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 一、前景遮挡
# ---------------------------------------------------------------------------
def detect_occluder(image, bbox, min_area_ratio=0.015):
    """在检测框内找出「挡在标牌前面」的物体，返回前景掩码（255=保留原图）。

    判定依据（按可靠性排序）：
      1) 从框边缘进入 —— 立柱/树枝/电线穿过标牌时，一定是从边缘连进来的
      2) 形态细长 —— 长宽比 >= 2.5，或者横贯/纵贯整个框
      3) 与标牌主体色彩差异大 —— 用 Lab 空间色差，避免把牌面深色图案误判

    为什么不能只看亮度：
      实测一块深红色「禁止停车」牌，主体均值只有 33，
      按「比均值暗 X%」判定会把整块牌吃成遮挡物，新标牌只剩碎片。
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    empty = np.zeros((h, w), np.uint8)

    # ⚠️ 小标牌不做遮挡检测。
    #    实测一块 26×26 的标牌：Otsu 只把 26.6% 判成牌面本体，
    #    剩下的墙/背景与本体色差很大，被整块判成"遮挡物"（93.8%），
    #    结果新标牌几乎贴不上、看起来"没变化"。
    #    小尺寸本身分辨率不足，判不出遮挡关系，直接放行更可靠。
    MIN_OCCL_SIZE = 40
    if h < MIN_OCCL_SIZE or w < MIN_OCCL_SIZE:
        return empty, {"occluder_ratio": 0.0, "skipped": "too_small"}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 标牌主体：Otsu 取亮的一半（圆形/矩形牌面通常是框内最亮的成片区域）
    _, body = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if body[h // 2, w // 2] == 0:
        body = cv2.bitwise_not(body)
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    body_px = body > 0
    body_ratio = float(body_px.sum()) / float(h * w)
    if body_px.sum() < 20:
        return empty, {"occluder_ratio": 0.0, "skipped": "no_body"}
    # 本体占比过低（<35%）说明 Otsu 没抓到标牌，此时算出的色差不可信
    if body_ratio < 0.35:
        return empty, {"occluder_ratio": 0.0, "skipped": "body_too_small",
                       "body_ratio": round(body_ratio, 3)}
    body_lab = lab[body_px].mean(axis=0)

    # 与主体色差大的像素 = 可能不是牌面
    dist = np.linalg.norm(lab - body_lab[None, None, :], axis=2)
    cand = (dist > 45).astype(np.uint8) * 255

    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros((h, w), np.uint8)
    min_area = max(10, int(h * w * min_area_ratio))
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        if area < min_area:
            continue
        # 条件1：从框边缘进入
        touches = (left <= 1 or top <= 1 or left + cw >= w - 1 or top + ch >= h - 1)
        # 条件2：细长或贯穿
        aspect = max(cw, ch) / float(max(1, min(cw, ch)))
        spanning = (max(cw, ch) >= 0.55 * max(w, h))
        if touches and (aspect >= 2.0 or spanning):
            keep[lab_img == i] = 255

    if keep.any():
        keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)

    ratio = float(keep.mean() / 255.0)
    # 遮挡占比超过 60% 基本是误判（真实场景里很少有物体挡住标牌大半还看得见）
    if ratio > 0.60:
        return empty, {"occluder_ratio": 0.0, "skipped": "ratio_too_high",
                       "raw_ratio": round(ratio, 3)}

    return keep, {"occluder_ratio": round(ratio, 3)}


# ---------------------------------------------------------------------------
# 二、阴影
# ---------------------------------------------------------------------------
def estimate_light(image, bbox):
    """从标牌周围区域估计光照方向与强度。

    实测坑：只取标牌外一小圈算梯度，遇到「白墙上的红牌」这种场景，
    墙面太平、梯度几乎为 0，strength 会被算成 0，阴影就永远不出现。
    改为：
      - 取标牌外 3 倍尺寸的大范围（能吃到天空/地面/建筑等明暗结构）
      - 强度用梯度的分位数而不是均值，避免被大片平墙稀释
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    pad = int(max(x2 - x1, y2 - y1) * 3.0)
    # ⚠️ 注意切片顺序是 [y1:y2, x1:x2]；之前写成 [x起:x止, y起:y止] 导致 ROI 为空
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(W, x2 + pad)
    ry2 = min(H, y2 + pad)
    roi = image[ry1:ry2, rx1:rx2]
    if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
        return 0.0, 1.0, 0.0

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (5, 5), 0)          # 先去噪，避免把噪点当梯度
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.size == 0:
        return 0.0, 1.0, 0.0

    # 用第 90 百分位当强度基准（平墙占比大时不会被拉低）
    p90 = float(np.percentile(mag, 90))
    strength = float(np.clip(p90 / 60.0, 0.0, 1.0))
    if p90 < 1e-3:
        return 0.0, 1.0, 0.0

    # 方向：只统计有明显梯度的像素，避免平墙像素互相抵消
    strong = mag > p90 * 0.5
    if strong.sum() < 20:
        return 0.0, 1.0, strength * 0.5
    wgt = mag[strong] / (mag[strong].sum() + 1e-6)
    dx = float((gx[strong] * wgt).sum())
    dy = float((gy[strong] * wgt).sum())
    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        return 0.0, 1.0, strength * 0.5
    return dx / norm, dy / norm, strength


def paint_shadow(canvas, bbox, dx, dy, strength=0.5, length_ratio=0.5, opacity=0.45):
    """在标牌下方/背光侧画一块渐隐阴影（直接画在 canvas 上，原地修改）。

    做法：沿「光向的反方向」偏移出一个形状，模糊后按透明度乘到画面上。
    这样标牌看起来是「贴在墙上、有影子」的，而不是浮着的。
    """
    if strength < 0.08:
        return canvas
    H, W = canvas.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return canvas

    # 阴影偏移方向 = -光向（光从右上打来，影子落左下）
    off = int(max(4, min(bw, bh) * length_ratio))
    sx = int(-dx * off)
    sy = int(-dy * off)

    mask = np.zeros((H, W), np.uint8)
    # 阴影形状比标牌稍大一点，并按方向偏移
    cv2.rectangle(mask, (x1 + sx, y1 + sy), (x2 + sx, y2 + sy), 255, -1)
    k = max(3, int(min(bw, bh) * 0.5) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    # 渐隐：靠近标牌一侧更深
    a = (mask.astype(np.float32) / 255.0) * opacity * strength
    out = canvas.astype(np.float32)
    out *= (1.0 - a[:, :, None] * 0.55)     # 压暗而不是涂黑，保留背景纹理
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 三、合成（把上面两步串起来）
# ---------------------------------------------------------------------------
def paste_with_occlusion(image, sign_bgr, sign_alpha, bbox,
                         degrade_fn=None, shadow=True, strength=1.0):
    """带遮挡保留与阴影的贴合。

    image       原图
    sign_bgr    已缩放到目标尺寸的标牌图案
    sign_alpha  标牌掩码（0~255），与 sign_bgr 同尺寸
    bbox        目标区域 [x1,y1,x2,y2]
    degrade_fn  劣化函数，签名 (sign, mask, ref) -> (sign, mask)
    shadow      是否投阴影

    返回 (合成图, 诊断信息)
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    tw, th = x2 - x1, y2 - y1
    if tw < 6 or th < 6:
        return image, {"ok": False}

    ref = image[y1:y2, x1:x2]

    # 1) 找遮挡物（要用原图，不能用合成后的）
    occl, oinfo = detect_occluder(image, [x1, y1, x2, y2])

    # 2) 劣化图案
    # ⚠️ 用 NEAREST：INTER_AREA 会把透明边缘和图案混合，贴出来一圈白边
    sig = cv2.resize(sign_bgr, (tw, th), interpolation=cv2.INTER_NEAREST)
    alp = cv2.resize(sign_alpha, (tw, th), interpolation=cv2.INTER_NEAREST)
    if degrade_fn is not None:
        sig, alp = degrade_fn(sig, alp, ref)

    # 3) 软边
    soft = cv2.GaussianBlur(alp.astype(np.float32) / 255.0, (3, 3), 0.6)
    soft = np.clip(soft, 0, 1)

    # 4) 遮挡物像素不贴（保留原图）
    if occl.any():
        keep = (occl.astype(np.float32) / 255.0)          # 1 = 遮挡
        soft = soft * (1.0 - keep)

    # 5) 合成
    out = image.astype(np.float32).copy()
    roi = out[y1:y2, x1:x2]
    a = soft[:, :, None]
    roi[:] = sig.astype(np.float32) * a + roi * (1.0 - a)
    out[y1:y2, x1:x2] = roi
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 6) 阴影
    info = dict(oinfo)
    if shadow:
        dx, dy, st = estimate_light(image, [x1, y1, x2, y2])
        if st > 0.08:
            out = paint_shadow(out, [x1, y1, x2, y2], dx, dy, st * strength)
            info["shadow"] = {"dx": round(dx, 2), "dy": round(dy, 2),
                              "strength": round(st, 2)}
    info["ok"] = True
    return out, info


if __name__ == "__main__":
    print("模块自检：")
    print("  detect_occluder  保留前景遮挡（立柱/树枝不会被子新标牌盖掉）")
    print("  estimate_light   从周围亮度梯度估计光向")
    print("  paint_shadow     按光向投渐隐阴影")
    print("  paste_with_occlusion  合成入口")
