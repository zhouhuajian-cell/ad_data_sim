#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_degrade.py — 把「矢量图案」劣化到「实拍照片」的质感，让替换不露馅

背景（为什么需要这一步）：
  上传的 TSR 图案通常是纯矢量（纯白底、纯红圈、锐利边缘），
  而实拍照片里的标牌有：模糊、噪声、JPEG 压缩伪影、不均匀反光、色偏、磨损。
  直接把矢量图贴上去，边缘锐得像贴纸，一眼就能看出是假的（实测效果很差）。

本模块按顺序做 5 件事，参数都可以调：
  1) 颜色对齐   把图案的均值/方差匹配到原标牌区域（保留背光/顺光感）
  2) 模糊       按原标牌的边缘锐度做高斯/运动模糊
  3) 纹理迁移   从原标牌区域提取高频纹理，加权叠回新图案（模拟磨损/污渍）
  4) 噪声       加与原图同量级的高斯噪声
  5) 边缘软化   腐蚀+羽化，避免硬边
"""
import cv2
import numpy as np


def estimate_blur(gray, max_sigma=3.0):
    """用拉普拉斯方差估计模糊程度：方差越小越模糊。返回建议的 sigma。"""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    # 经验映射：清晰图 var 常在 500+，模糊图 <100
    sigma = max_sigma * (1.0 - min(var, 500.0) / 500.0)
    return float(np.clip(sigma, 0.0, max_sigma)), var


def color_match(patch, ref):
    """把 patch 的颜色分布对齐到 ref（逐通道均值/方差）。"""
    p = patch.astype(np.float32)
    r = ref.reshape(-1, 3).astype(np.float32)
    p_mean = p.reshape(-1, 3).mean(axis=0)
    p_std = p.reshape(-1, 3).std(axis=0) + 1e-6
    r_mean = r.mean(axis=0)
    r_std = r.std(axis=0) + 1e-6
    out = (p - p_mean) / p_std * r_std * 0.85 + r_mean   # 0.85 保留一点对比度
    return np.clip(out, 0, 255)


def transfer_texture(patch, ref):
    """把 ref 的高频纹理（细节层）按强度叠加到 patch，模拟真实表面污渍/磨损。"""
    def high_freq(img):
        g = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        low = cv2.GaussianBlur(g, (0, 0), 3.0)
        return g - low

    h_p = high_freq(patch)
    h_r = high_freq(ref)
    if h_p.shape != h_r.shape:
        h_r = cv2.resize(h_r, (patch.shape[1], patch.shape[0]))
    # 只叠加参照的纹理，不改变整体色调
    detail = h_r[:, :, None]
    return np.clip(patch.astype(np.float32) + detail * 0.35, 0, 255)


def add_noise(patch, ref):
    """按参照图的噪声水平给 patch 加噪。"""
    def noise_std(img):
        g = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        med = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
        return float((g - med).std())

    n_std = min(noise_std(ref), 12.0)
    noise = np.random.normal(0, n_std, patch.shape).astype(np.float32)
    return np.clip(patch.astype(np.float32) + noise, 0, 255)


def degrade(sign_bgr, mask, ref_patch, strength=1.0):
    """主入口：把图案劣化到参照区域的质感。

    sign_bgr  图案（BGR，已缩放到目标尺寸）
    mask      图案掩码（0/255）
    ref_patch 原标牌区域的图像（同尺寸）
    strength  总强度 0~2（1=默认）
    """
    if ref_patch.shape[:2] != sign_bgr.shape[:2]:
        ref_patch = cv2.resize(ref_patch, (sign_bgr.shape[1], sign_bgr.shape[0]))

    out = sign_bgr.astype(np.float32)

    # 1) 颜色对齐
    if strength > 0:
        out = color_match(out, ref_patch)

    # 2) 模糊：按参照图的锐度
    gray_ref = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    sigma, _ = estimate_blur(gray_ref)
    sigma *= strength
    if sigma > 0.3:
        k = int(max(3, 2 * round(sigma * 2) + 1))
        out = cv2.GaussianBlur(out, (k, k), sigma)

    # 3) 纹理迁移
    if strength > 0:
        out = transfer_texture(out, ref_patch)

    # 4) 噪声
    if strength > 0:
        out = add_noise(out, ref_patch)

    # 5) 边缘软化：把掩码腐蚀一点再羽化，避免硬边
    soft_mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    soft_mask = cv2.GaussianBlur(soft_mask.astype(np.float32) / 255.0, (5, 5), 0)
    soft_mask = np.clip(soft_mask, 0, 1)[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8), (soft_mask[:, :, 0] * 255).astype(np.uint8)


def paste_degraded(image, sign_bgr, mask, bbox, strength=1.0):
    """劣化 + 粘贴，一步到位。bbox 是目标区域 [x1,y1,x2,y2]。"""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    tw, th = x2 - x1, y2 - y1
    if tw < 6 or th < 6:
        return image, None

    # 图案缩放到目标尺寸（保持长宽比的缩放由调用方做，这里直接铺满）
    sign_r = cv2.resize(sign_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    ref = image[y1:y2, x1:x2]

    degraded, soft = degrade(sign_r, mask_r, ref, strength)

    out = image.astype(np.float32).copy()
    roi = out[y1:y2, x1:x2]
    a = (soft.astype(np.float32) / 255.0)[:, :, None]
    roi[:] = degraded.astype(np.float32) * a + roi * (1 - a)
    out[y1:y2, x1:x2] = roi
    return np.clip(out, 0, 255).astype(np.uint8), soft


if __name__ == "__main__":
    # 自检：用一张实拍标牌当参照，检查劣化前后的差异
    import sys
    if len(sys.argv) >= 3:
        sign = cv2.imread(sys.argv[1])
        ref = cv2.imread(sys.argv[2])
        m = np.full(sign.shape[:2], 255, np.uint8)
        d, soft = degrade(sign, m, ref)
        cv2.imwrite("/tmp/degraded.png", d)
        print("劣化完成 -> /tmp/degraded.png")
    else:
        print("用法: python3 sign_degrade.py <图案> <参照区域>")
