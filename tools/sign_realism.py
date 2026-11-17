#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_realism.py — 让替换后的 TSR 标牌「像真实拍摄的」，可直接当训练数据

当前版本的核心改进：**全部参数从原标牌实测自适应**，不再用固定值。

为什么必须自适应（实测数据）：
  同一批实拍图里，原标牌的亮度在 79~137、对比度 36~73、饱和度 46~93 之间波动。
  用固定参数劣化，在暗场景里过亮、在亮场景里偏灰，一眼就能看出是贴的。

处理链（每一步的参数都来自参照区域）：
  1) 亮度对齐   把图案的**平均亮度**移到参照区域的水平（保留图案自身对比度）
  2) 对比度对齐 图案的对比度按参照区域缩放，但不低于原图案的 92%（保住可辨识度）
  3) 饱和度    以参照区饱和度为基准，但不低于图案自身的 85%（保住交通标志的鲜艳）
  4) 局部光照  用参照区域的低频亮度场给图案加渐变（侧光/逆光跟随场景）
  5) 模糊      按参照区域的拉普拉斯方差决定高斯 sigma
  6) 噪声      按参照区域的噪声水平加高斯噪声
  7) JPEG 伪影 走一次真实编码，模拟采集链路的压缩
  8) 边缘羽化  1~2px 软边，保留轮廓（不做腐蚀，避免变「软塌塌的贴纸」）

训练数据友好性：
  - 保住类别特征（红/黄/蓝的高饱和、形状轮廓、数字/符号对比度）
  - 只增加「采集侧」劣化，不加「合成侧」假痕迹
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 参照区域的统计量
# ---------------------------------------------------------------------------
def plate_mask(ref_patch):
    """估计「标牌本体」的掩码：把亮的/高对比的那一块抠出来，排除背景。

    为什么需要：实测直接用整框均值会让替换后偏亮 20~30——
    框里混了墙、天空、电线这些背景，它们比标牌暗，把均值拉低了，
    而我按这个偏低的均值去对齐，结果新标牌比原来的亮。

    做法：Otsu 分割取「与中心同色」的那一半，再做一次开运算去掉细碎噪点。
    """
    g = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = th.shape
    if h < 4 or w < 4:
        return np.full((h, w), 255, np.uint8)
    # 中心像素落在哪一半，就以哪一半为前景
    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    if th.sum() < 10:
        return np.full((h, w), 255, np.uint8)
    return th


def ref_stats(ref_patch, mask=None):
    """提取参照区域的亮度/对比度/饱和度/锐度/噪声，作为劣化参数的依据。

    mask 为空时自动抠出标牌本体（见 plate_mask），避免背景拉低均值。
    """
    if mask is None or mask.sum() < 10:
        mask = plate_mask(ref_patch)

    g = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)

    gm = g[mask > 0]
    sm = s[mask > 0]

    # 锐度：用 float32 输入（OpenCV 5.0 不接受 Laplacian(CV_64F)）
    lap_var = float(cv2.Laplacian(g, -1).var())
    # 噪声：与 3x3 中值的残差
    noise_std = float(np.std(g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)))

    return {
        "mean": float(np.mean(gm)) if gm.size else 128.0,
        "std": float(np.std(gm)) if gm.size else 48.0,
        "sat": float(np.mean(sm)) if sm.size else 60.0,
        "lap_var": lap_var,
        "noise_std": noise_std,
    }


# ---------------------------------------------------------------------------
# 各步骤（参数都由 stats 驱动）
# ---------------------------------------------------------------------------
def match_tone(sign_bgr, stats, keep_contrast=0.92, min_brightness=88.0,
               brightness_gain=1.25):
    """亮度与对比度对齐。

    两个目标在打架，这里取折中：
      - 视觉融合：要和周围亮度接近，不然像贴纸
      - 训练可辨识：标牌本身反光强，不该被压到和周围一样暗
    实测：原标牌本体亮度只有 46~70（背光/阴影/镜头衰减），
    直接按它对齐会让新标牌也变暗，模型更难学。
    所以目标亮度 = max(参照亮度 × gain, min_brightness)，
    既跟随环境，又给可辨识度兜底。

    对比度同理：取参照对比度与「原图案对比度 × keep_contrast」中较大者。
    """
    p = sign_bgr.astype(np.float32)
    cur_mean = float(p.mean())
    cur_std = float(p.std())
    if cur_std < 1e-3:
        return p

    target_bright = max(stats["mean"] * brightness_gain, min_brightness)
    target_std = max(stats["std"] * 0.92, cur_std * keep_contrast)
    scale = target_std / cur_std

    out = (p - cur_mean) * scale + target_bright
    return np.clip(out, 0, 255)


def match_saturation(sign_bgr, stats, floor=0.85):
    """饱和度向参照靠拢，但不低于图案自身 floor 倍（保住交通标志的鲜艳度）。"""
    hsv = cv2.cvtColor(np.clip(sign_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    cur_s = float(hsv[:, :, 1].mean())
    if cur_s < 1e-3:
        return sign_bgr.astype(np.float32)
    target_s = max(stats["sat"], cur_s * floor)
    factor = float(np.clip(target_s / cur_s, 0.8, 1.35))
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def apply_lighting(sign_bgr, ref_patch, strength=1.0):
    """用参照区域的低频亮度场给图案加渐变，让标牌跟着场景一起明暗。"""
    g = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    cells = max(2, min(h, w) // 8)
    small = cv2.resize(g, (cells, cells), interpolation=cv2.INTER_AREA)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    m = float(field.mean())
    if m < 1e-3:
        return sign_bgr
    norm = np.clip(field / m, 1.0 - 0.25 * strength, 1.0 + 0.25 * strength)
    return np.clip(sign_bgr.astype(np.float32) * norm[:, :, None], 0, 255)


def blur_sigma_from(stats, max_sigma=2.2):
    """按参照区域的拉普拉斯方差决定模糊量：越模糊的图，贴上去也越模糊。"""
    v = max(0.0, stats["lap_var"])
    sigma = max_sigma * (1.0 - min(v, 400.0) / 400.0)
    return float(np.clip(sigma, 0.0, max_sigma))


def add_noise(sign_bgr, stats, factor=0.8):
    """按参照区域的噪声水平加噪（上限 8，避免糊掉）。"""
    n_std = min(stats["noise_std"] * factor, 8.0)
    if n_std < 0.4:
        return sign_bgr
    noise = np.random.normal(0, n_std, sign_bgr.shape).astype(np.float32)
    return np.clip(sign_bgr.astype(np.float32) + noise, 0, 255)


def jpeg_roundtrip(patch, quality=90):
    ok, buf = cv2.imencode(".jpg", np.clip(patch, 0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return patch
    return cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def make_realistic(sign_bgr, mask, ref_patch, strength=1.0, jpeg_quality=90):
    """把图案处理成「像真实拍摄的标牌」。所有参数从 ref_patch 实测。

    返回 (处理后的图案, 软掩码)
    """
    if ref_patch.shape[:2] != sign_bgr.shape[:2]:
        ref_patch = cv2.resize(ref_patch, (sign_bgr.shape[1], sign_bgr.shape[0]))

    stats = ref_stats(ref_patch)
    out = sign_bgr.astype(np.float32)

    out = match_tone(out, stats)          # 1) 亮度 + 对比度
    out = match_saturation(out, stats)    # 2) 饱和度

    if strength > 0:
        out = apply_lighting(out, ref_patch, strength)    # 3) 光照渐变

    sigma = blur_sigma_from(stats) * strength             # 4) 模糊
    if sigma > 0.3:
        k = int(max(3, 2 * round(sigma * 1.5) + 1))
        out = cv2.GaussianBlur(out, (k, k), sigma)

    if strength > 0:
        out = add_noise(out, stats, 0.8 * strength)       # 5) 噪声
        out = jpeg_roundtrip(out, quality=jpeg_quality)   # 6) JPEG 伪影

    # 7) 软掩码：1~2px 羽化（不做腐蚀）
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (3, 3), 0.6)
    soft = np.clip(soft, 0, 1)

    return np.clip(out, 0, 255).astype(np.uint8), (soft * 255).astype(np.uint8)


def paste_realistic(image, sign_bgr, mask, bbox, strength=1.0, jpeg_quality=90):
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

    out_patch, soft = make_realistic(sign_r, mask_r, ref, strength, jpeg_quality)

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
