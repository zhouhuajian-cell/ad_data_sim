#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_realism.py — 让替换后的 TSR 标牌「像真实拍摄的」，可直接当训练数据

上一版的问题（实测）：
  sign_degrade.py 把颜色整体对齐到参照区域，导致：
    - 红环被冲淡（交通红应该饱和）
    - 白底变灰（标牌白底在照片里是亮的）
    - 整体对比度下降，训练时模型分不清

本版的思路（在保留「真实感」的同时保住「可辨识度」）：
  1) 亮度对齐但不压对比度 —— 用参照区域的**亮度均值**对齐整体明暗，
     但保持图案自己的对比度（不像上一版那样把方差也压过去）
  2) 保持饱和度 —— 交通标志的红/黄/蓝是高饱和色，照片里也是；
     只对饱和度过高的合成色做轻微收敛，不做整体降饱和
  3) 局部光照模拟 —— 按原标牌的亮度梯度叠加渐变，模拟侧光/逆光
  4) 真实劣化链 —— 模糊（按原图锐度）→ 轻微噪声 → JPEG 压缩 → 轻微色偏
  5) 边缘处理 —— 不做硬腐蚀，用 1px 羽化保留标牌轮廓（避免变成「软塌塌的贴纸」）

训练数据友好性：
  - 保留标牌的类别特征（颜色、形状、对比度）
  - 只增加「采集侧」的劣化（噪声/模糊/压缩），不加「合成侧」的假痕迹
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _lum(bgr):
    return cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def target_brightness(ref_patch, mask=None):
    """参照区域的目标平均亮度（0~255）。有掩码就只统计掩码内。"""
    g = _lum(ref_patch)
    if mask is not None and mask.sum() > 0:
        vals = g[mask > 0]
    else:
        vals = g.reshape(-1)
    if vals.size == 0:
        return 128.0
    return float(np.mean(vals))


def match_brightness_keep_contrast(sign_bgr, target_mean, gain=1.0):
    """把图案的**平均亮度**移到 target_mean，但保留图案自己的对比度。

    做法：先按图案自身均值做中心化，再整体平移到目标亮度。
    这样红环依然红、白底依然白，只是整体明暗贴合场景。
    """
    p = sign_bgr.astype(np.float32)
    cur = float(p.mean())
    if cur < 1e-3:
        return p
    # 轻微放大对比度（交通标志本身对比强），再平移到目标亮度
    out = (p - cur) * gain + target_mean
    return np.clip(out, 0, 255)


def boost_saturation(bgr, factor=1.15):
    """提升饱和度：交通标志的红/黄/蓝是高对比色，照片里也饱和。"""
    hsv = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def apply_lighting_gradient(sign_bgr, ref_patch):
    """按参照区域的亮度梯度给图案加渐变，模拟侧光/逆光。

    做法：把参照图降采样成很粗的亮度场（只保留大范围明暗变化），
    归一化后乘到图案上——这样标牌会「跟着场景一起亮/暗」。
    """
    g = _lum(ref_patch)
    h, w = g.shape
    cells = max(2, min(h, w) // 8)
    small = cv2.resize(g, (cells, cells), interpolation=cv2.INTER_AREA)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    m = float(field.mean())
    if m < 1e-3:
        return sign_bgr
    # 归一化到 0.75~1.25 之间，避免过强
    norm = np.clip(field / m, 0.75, 1.25)
    return np.clip(sign_bgr.astype(np.float32) * norm[:, :, None], 0, 255)


def estimate_blur_sigma(ref_patch, max_sigma=2.5):
    """用拉普拉斯方差估参照区域锐度，返回建议的高斯 sigma。"""
    # OpenCV 5.0 对 Laplacian(CV_64F) 会报 "Unsupported combination"，
    # 改成先把灰度转 float32 再用 ddepth=-1（与输入同深度）计算方差
    g = _lum(ref_patch).astype(np.float32)
    var = float(cv2.Laplacian(g, -1).var())
    sigma = max_sigma * (1.0 - min(var, 400.0) / 400.0)
    return float(np.clip(sigma, 0.0, max_sigma))


def add_real_noise(patch, ref_patch, strength=0.6):
    """加与参照区域同量级的高斯噪声（模拟传感器噪声）。"""
    g = _lum(ref_patch)
    med = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    n_std = float(np.std(g - med)) * strength
    n_std = min(n_std, 8.0)
    if n_std < 0.5:
        return patch
    noise = np.random.normal(0, n_std, patch.shape).astype(np.float32)
    return np.clip(patch.astype(np.float32) + noise, 0, 255)


def jpeg_roundtrip(patch, quality=88):
    """走一次 JPEG 编解码，模拟照片的压缩伪影（训练数据里所有图都有）。"""
    ok, buf = cv2.imencode(".jpg", np.clip(patch, 0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return patch
    return cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def make_realistic(sign_bgr, mask, ref_patch, strength=1.0, sat_boost=1.18):
    """把图案处理成「像真实拍摄的标牌」。

    sign_bgr  图案（BGR，已缩放到目标尺寸）
    mask      图案掩码（0/255）
    ref_patch 原标牌区域（同尺寸），用来取亮度/锐度/噪声
    strength  劣化强度（0=不处理，1=默认，1.5=更旧更脏）
    sat_boost 饱和度提升系数（交通标志建议 1.1~1.25）

    返回 (处理后的图案, 软掩码)
    """
    if ref_patch.shape[:2] != sign_bgr.shape[:2]:
        ref_patch = cv2.resize(ref_patch, (sign_bgr.shape[1], sign_bgr.shape[0]))

    out = sign_bgr.astype(np.float32)

    # 1) 亮度对齐（保对比）
    tgt = target_brightness(ref_patch, mask)
    out = match_brightness_keep_contrast(out, tgt, gain=1.0 + 0.1 * strength)

    # 2) 饱和度：交通标志本身饱和度高，这里略微加强让类别特征更清晰
    out = boost_saturation(out, sat_boost)

    # 3) 局部光照渐变（跟着场景一起明暗）
    if strength > 0:
        out = apply_lighting_gradient(out, ref_patch)

    # 4) 模糊：匹配原图锐度
    sigma = estimate_blur_sigma(ref_patch) * strength
    if sigma > 0.3:
        k = int(max(3, 2 * round(sigma * 1.5) + 1))
        out = cv2.GaussianBlur(out, (k, k), sigma)

    # 5) 噪声
    if strength > 0:
        out = add_real_noise(out, ref_patch, 0.6 * strength)

    # 6) JPEG 伪影（只做一次，强度随 strength 变）
    if strength > 0:
        out = jpeg_roundtrip(out, quality=int(92 - 8 * strength))

    # 7) 软掩码：1~2px 羽化，保留清晰轮廓（不做腐蚀）
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (3, 3), 0.6)
    soft = np.clip(soft, 0, 1)

    return np.clip(out, 0, 255).astype(np.uint8), (soft * 255).astype(np.uint8)


def paste_realistic(image, sign_bgr, mask, bbox, strength=1.0, sat_boost=1.18):
    """一步完成：取参照 → 处理 → 粘贴。bbox=[x1,y1,x2,y2]。"""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    tw, th = x2 - x1, y2 - y1
    if tw < 6 or th < 6:
        return image, None

    sign_r = cv2.resize(sign_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    ref = image[y1:y2, x1:x2]

    out_patch, soft = make_realistic(sign_r, mask_r, ref, strength, sat_boost)

    out = image.astype(np.float32).copy()
    roi = out[y1:y2, x1:x2]
    a = (soft.astype(np.float32) / 255.0)[:, :, None]
    roi[:] = out_patch.astype(np.float32) * a + roi * (1 - a)
    out[y1:y2, x1:x2] = roi
    return np.clip(out, 0, 255).astype(np.uint8), soft


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        s = cv2.imread(sys.argv[1])
        r = cv2.imread(sys.argv[2])
        m = np.full(s.shape[:2], 255, np.uint8)
        d, _ = make_realistic(s, m, r)
        cv2.imwrite("/tmp/realistic.png", d)
        print("输出 /tmp/realistic.png")
    else:
        print("用法: python3 sign_realism.py <图案> <参照区域>")
