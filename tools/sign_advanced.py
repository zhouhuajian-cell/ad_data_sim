#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_advanced.py — 引入业界高 star 项目的几个关键技术，提升仿真精度

参考来源（都是 star 高、被广泛使用的做法）：
  - albumentations (13k+★) 的 FDA / HistogramMatching / Illumination / PlanckianJitter
    https://github.com/albumentations-team/albumentations
  - COCO-traffic-sign-dataset-generator：前景与背景各自增强后再合成
    https://github.com/ph-phuc/COCO-traffic-sign-dataset-generator
  - copy-paste-augmentation：粘贴前做尺度/位置抖动
    https://github.com/itstechaj/copy-paste-augmentation

实现的四个技术（按对仿真精度的影响排序）：
  1) FDA  —— 傅里叶域自适应。把图案的低频幅度谱替换成参照区域的低频幅度谱，
             颜色/光照整体自动贴合场景，比手调均值方差稳健得多。
  2) 直方图匹配 —— 把图案的亮度分布对齐到参照，比只对齐均值方差更细。
  3) 光照方向 —— 用参照区域的梯度主方向估计光从哪来，给图案加对应的高光/阴影渐变。
  4) 尺度+位置抖动 —— 同一张原图标牌生成多个变体（尺度、轻微位移），提升数据多样性。
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1) FDA（Fourier Domain Adaptation）
# ---------------------------------------------------------------------------
def fda(sign_bgr, ref_patch, beta=0.01):
    """傅里叶域自适应：用参照区域的低频幅度谱替换图案的幅度谱。

    beta 控制替换范围（越小只换最低频，即整体色调；越大换得越多）。
    论文里 beta=0.01 是常用起点。

    为什么有效：图像的低频承载颜色和光照分布，高频承载纹理和边缘。
    只换低频幅度、保留相位（=保留图案的形状与内容），就能让图案
    「长在」场景的光照里，同时内容不变。
    单通道分别处理，避免通道间串色。
    """
    sh, sw = sign_bgr.shape[:2]
    ref = cv2.resize(ref_patch, (sw, sh)) if ref_patch.shape[:2] != (sh, sw) else ref_patch

    out = np.zeros_like(sign_bgr, dtype=np.float32)
    for c in range(sign_bgr.shape[2]):
        src = sign_bgr[:, :, c].astype(np.float32)
        tgt = ref[:, :, c].astype(np.float32)

        f_src = np.fft.fftshift(np.fft.fft2(src))
        f_tgt = np.fft.fftshift(np.fft.fft2(tgt))

        amp_src, pha_src = np.abs(f_src), np.angle(f_src)
        amp_tgt = np.abs(f_tgt)

        # 低频掩码：以中心为圆心的圆内
        h, w = sh, sw
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        radius = min(h, w) * beta
        low = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

        amp_new = amp_src.copy()
        amp_new[low] = amp_tgt[low]

        f_new = amp_new * np.exp(1j * pha_src)
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(f_new)))
    return np.clip(out, 0, 255)


# ---------------------------------------------------------------------------
# 2) 直方图匹配（仅亮度，保留色彩）
# ---------------------------------------------------------------------------
def match_histogram_luma(sign_bgr, ref_patch, strength=0.7):
    """把图案的亮度直方图向参照靠拢，色彩保持不变。

    为什么不直接对 RGB 各通道做匹配：会串色，红环可能变紫。
    只动亮度（Lab 的 L 通道），a/b（色度）保留图案自己的。
    """
    ref = cv2.resize(ref_patch, (sign_bgr.shape[1], sign_bgr.shape[0])) \
        if ref_patch.shape[:2] != sign_bgr.shape[:2] else ref_patch

    lab_s = cv2.cvtColor(np.clip(sign_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab_r = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)

    Ls = lab_s[:, :, 0].astype(np.float32)
    Lr = lab_r[:, :, 0].astype(np.float32)

    # 累计分布匹配
    s_vals, s_idx, s_cnt = np.unique(Ls.ravel(), return_inverse=True, return_counts=True)
    r_vals, r_cnt = np.unique(Lr.ravel(), return_counts=True)
    s_cdf = np.cumsum(s_cnt).astype(np.float64) / Ls.size
    r_cdf = np.cumsum(r_cnt).astype(np.float64) / Lr.size
    # 为每个源灰度值找目标灰度值
    mapped = np.interp(s_cdf, r_cdf, r_vals)
    Lm = mapped[s_idx].reshape(Ls.shape)

    # 按 strength 与原始亮度混合，避免过强
    L_new = Ls * (1 - strength) + Lm * strength
    lab_s[:, :, 0] = np.clip(L_new, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_s, cv2.COLOR_LAB2BGR).astype(np.float32)


# ---------------------------------------------------------------------------
# 3) 光照方向
# ---------------------------------------------------------------------------
def estimate_light_direction(ref_patch):
    """用亮度梯度的主方向估计光从哪来，返回 (dx, dy, 强度 0~1)。

    做法：对参照区域做 Sobel，取梯度向量的主成分方向。
    亮面朝向光源，所以梯度（暗→亮）的方向近似「指向光」。
    """
    g = cv2.cvtColor(np.clip(ref_patch, 0, 255).astype(np.uint8),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.mean() < 1e-3:
        return 0.0, 0.0, 0.0
    # 按梯度强度加权求平均方向
    w = mag / (mag.sum() + 1e-6)
    dx = float((gx * w).sum())
    dy = float((gy * w).sum())
    norm = np.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0, 0.0, 0.0
    strength = float(np.clip(mag.mean() / 40.0, 0.0, 1.0))
    return dx / norm, dy / norm, strength


def apply_directional_light(sign_bgr, ref_patch, amount=0.18):
    """按估计的光向给图案加方向性明暗渐变（模拟侧光）。"""
    dx, dy, s = estimate_light_direction(ref_patch)
    if s < 0.05:
        return sign_bgr
    h, w = sign_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 归一化坐标 [-1, 1]
    nx = (xx / max(1, w - 1)) * 2 - 1
    ny = (yy / max(1, h - 1)) * 2 - 1
    # 沿光向的投影：朝光一侧更亮
    field = (nx * dx + ny * dy)
    field = field - field.mean()
    mx = np.abs(field).max() + 1e-6
    field = field / mx
    gain = 1.0 + field * amount * s
    return np.clip(sign_bgr.astype(np.float32) * gain[:, :, None], 0, 255)


# ---------------------------------------------------------------------------
# 4) 尺度/位置抖动
# ---------------------------------------------------------------------------
def jitter_bbox(bbox, rng, scale_pct=0.10, shift_pct=0.05, img_shape=None):
    """对 bbox 做尺度与位置抖动，返回新 bbox。

    为什么需要：同一张原图标牌可以生成多个变体，增加数据多样性而不增加采集成本。
    参考 copy-paste-augmentation 的做法，抖动幅度保持小（尺 10%、位移 5%），
    否则会脱离真实分布。
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

    s = 1.0 + rng.uniform(-scale_pct, scale_pct)
    nw, nh = w * s, h * s
    cx += rng.uniform(-shift_pct, shift_pct) * w
    cy += rng.uniform(-shift_pct, shift_pct) * h

    nx1, ny1, nx2, ny2 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if img_shape is not None:
        H, W = img_shape[:2]
        # 保持尺寸不变，只平移回画面内
        if nx1 < 0:
            nx2 -= nx1
            nx1 = 0
        if ny1 < 0:
            ny2 -= ny1
            ny1 = 0
        if nx2 > W:
            nx1 -= (nx2 - W)
            nx2 = W
        if ny2 > H:
            ny1 -= (ny2 - H)
            ny2 = H
        nx1, ny1 = max(0, nx1), max(0, ny1)
    return [nx1, ny1, nx2, ny2]


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        s = cv2.imread(sys.argv[1])
        r = cv2.imread(sys.argv[2])
        if s is None or r is None:
            print("读图失败")
            sys.exit(1)
        a = fda(s, r)
        cv2.imwrite("/tmp/adv_fda.png", a)
        b = match_histogram_luma(a, r)
        cv2.imwrite("/tmp/adv_hist.png", b)
        c = apply_directional_light(b, r)
        cv2.imwrite("/tmp/adv_light.png", c)
        dx, dy, st = estimate_light_direction(r)
        print("光向 dx=%.2f dy=%.2f 强度=%.2f" % (dx, dy, st))
        print("输出 /tmp/adv_fda.png /tmp/adv_hist.png /tmp/adv_light.png")
    else:
        print("用法: python3 sign_advanced.py <图案> <参照区域>")
