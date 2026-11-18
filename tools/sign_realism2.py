#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_realism2.py — 提升真实感：环境氛围匹配 + 边缘过渡

上一版的问题（用户反馈「像贴纸」）：
  1) 亮度按「标牌本体均值」对齐 —— 实测阴雨天场景全图亮度 113，
     但标牌区域只有 42，按它对齐后贴上去的牌比环境暗一大截，很假。
     真实标牌是反光材料，在灰白天空下应当偏亮，不该跟着周围暗部走。
  2) 边缘硬切 —— 直接按掩码软边贴，和背景之间一条利落的界线。
     真实拍摄的物体边缘有镜头光晕、有 2~4px 的过渡。
  3) 缺环境氛围 —— 雨雾天拍出来的东西低对比、泛灰。贴上去的牌对比度太高、"太实"。

本版做法：
  A. 亮度基准改为「场景参考亮度」= 取标牌外围一圈的较亮分位数（天空/远景），
     而不是标牌自身的暗部。
  B. 对比度向环境靠拢：场景对比低（雾天），贴的牌也降对比。
  C. 加雾/霾：按场景的「暗部抬高程度」估计雾度，对牌面叠加同样的抬升。
  D. 边缘：按原标牌边缘的实际过渡宽度做羽化，并保留一圈微弱的边缘光。
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 环境分析
# ---------------------------------------------------------------------------
def scene_stats(image, bbox, ring_ratio=1.6):
    """分析标牌周围的环境：亮度、对比、雾度。

    返回 dict:
      bright    环境参考亮度（取外围较亮分位，代表天空/远景）
      contrast  环境对比度
      haze      雾度 0~1（暗部被抬高的程度，雾天值大）
      bg_mean   外围均值
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * ring_ratio)
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(W, x2 + pad), min(H, y2 + pad)
    roi = image[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return {"bright": 128.0, "contrast": 48.0, "haze": 0.3, "bg_mean": 128.0}

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 排除标牌本体所在的小块
    mask = np.ones(g.shape, np.uint8) * 255
    ix1, iy1 = x1 - rx1, y1 - ry1
    ix2, iy2 = x2 - rx1, y2 - ry1
    mask[max(0, iy1):iy2, max(0, ix1):ix2] = 0
    vals = g[mask > 0]
    if vals.size < 50:
        vals = g.reshape(-1)

    bright = float(np.percentile(vals, 75))       # 较亮分位 ≈ 天空/远景
    contrast = float(vals.std())
    bg_mean = float(vals.mean())

    # 雾度：暗部被抬高的程度。晴天暗部接近 0，雾天暗部会明显抬高
    dark = float(np.percentile(vals, 10))
    haze = float(np.clip(dark / 120.0, 0.0, 0.75))

    return {"bright": bright, "contrast": contrast, "haze": haze, "bg_mean": bg_mean}


# ---------------------------------------------------------------------------
# 劣化
# ---------------------------------------------------------------------------
# 可调参数（面板滑块控制）
# 默认值来自用户实测调优（阴雨天南美实拍场景），偏暗一档更贴近真实拍摄
PARAMS = {
    "bright_gain": 0.50,     # 目标亮度 = 环境亮分位 × gain（调小变暗）
    "haze_gain": 0.75,       # 雾度强度（调小更实、对比更强）
    "contrast_gain": 1.15,   # 对比度相对环境的倍数（调小更灰）
    "satur_keep": 0.70,      # 雾天饱和度保留比例（调小更灰）
    "feather_ratio": 0.08,   # 边缘羽化 = 标牌短边 × 该比例
    "min_bright": 55.0,      # 亮度下限（保证可辨识）
}


def set_params(**kw):
    """更新参数。只接受 PARAMS 里有的键。"""
    for k, v in kw.items():
        if k in PARAMS and v is not None:
            try:
                PARAMS[k] = float(v)
            except (TypeError, ValueError):
                pass
    return dict(PARAMS)


def degrade_v2(sign_bgr, mask, ref_patch, scene, strength=1.0):
    """按环境氛围劣化图案。返回 (图案, 软掩码)。

    sign_bgr   已缩放到目标尺寸的图案
    mask       图案掩码
    ref_patch  原标牌区域（同尺寸），用于取锐度/噪声
    scene      scene_stats 的结果
    """
    if ref_patch.shape[:2] != sign_bgr.shape[:2]:
        ref_patch = cv2.resize(ref_patch, (sign_bgr.shape[1], sign_bgr.shape[0]))

    p = sign_bgr.astype(np.float32)

    # ---- A. 亮度：向环境参考亮度靠拢，而不是向标牌暗部靠拢 ----
    # 标牌是反光材料，亮度取「环境亮分位 × 0.95」附近；雾天略低
    target = scene["bright"] * (PARAMS["bright_gain"] - 0.15 * scene["haze"] * PARAMS["haze_gain"])
    # 但不低于下限（保证可辨识）
    target = max(target, PARAMS["min_bright"])
    cur_mean = float(p.mean())
    cur_std = float(p.std()) + 1e-6
    # 对比度：向环境对比靠拢，但保留至少 60%（交通标志本身对比强）
    target_std = max(scene["contrast"] * PARAMS["contrast_gain"], cur_std * 0.60)
    p = (p - cur_mean) / cur_std * target_std + target

    # ---- C. 雾/霾：抬起暗部，压低整体反差（模拟空气散射） ----
    hz = scene["haze"] * strength * PARAMS["haze_gain"]
    if hz > 0.02:
        # 暗部抬升：p = p*(1-k) + 255*k*0.85
        k = hz * 0.35
        p = p * (1 - k) + 255 * 0.82 * k
        # 轻微降饱和（雾天色彩发灰）
        hsv = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= (1 - hz * (1.0 - PARAMS["satur_keep"]))
        p = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # ---- 模糊：匹配参照区域的锐度 ----
    g_ref = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap_var = float(cv2.Laplacian(g_ref, -1).var())
    sigma = 2.2 * (1.0 - min(lap_var, 400.0) / 400.0) * strength
    if sigma > 0.3:
        k = int(max(3, 2 * round(sigma * 1.5) + 1))
        p = cv2.GaussianBlur(p, (k, k), sigma)

    # ---- 噪声 + JPEG 伪影 ----
    if strength > 0:
        n_std = min(float(np.std(g_ref - cv2.medianBlur(g_ref.astype(np.uint8), 3).astype(np.float32))) * 0.8, 8.0)
        if n_std > 0.4:
            p = p + np.random.normal(0, n_std, p.shape).astype(np.float32)
        ok, buf = cv2.imencode(".jpg", np.clip(p, 0, 255).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            p = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)

    # ---- D. 软掩码：按原标牌的实际边缘过渡宽度羽化 ----
    soft = mask.astype(np.float32) / 255.0
    # 用小标牌的尺寸决定羽化半径：越小的牌，边缘相对越糊
    h, w = soft.shape
    feather_px = max(1.2, min(h, w) * PARAMS["feather_ratio"])
    kf = int(max(3, 2 * round(feather_px * 2) + 1))
    soft = cv2.GaussianBlur(soft, (kf, kf), feather_px)

    return np.clip(p, 0, 255).astype(np.uint8), (np.clip(soft, 0, 1) * 255).astype(np.uint8)


def paste_v2(image, sign_bgr, sign_alpha, bbox, strength=1.0, edge_glow=True):
    """带环境氛围匹配与边缘过渡的贴合。

    返回 (合成图, 信息)
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    tw, th = x2 - x1, y2 - y1
    if tw < 6 or th < 6:
        return image, {"ok": False}

    ref = image[y1:y2, x1:x2]
    scene = scene_stats(image, [x1, y1, x2, y2])

    sig = cv2.resize(sign_bgr, (tw, th), interpolation=cv2.INTER_NEAREST)
    alp = cv2.resize(sign_alpha, (tw, th), interpolation=cv2.INTER_NEAREST)
    sig, soft = degrade_v2(sig, alp, ref, scene, strength)

    a = (soft.astype(np.float32) / 255.0)[:, :, None]
    out = image.astype(np.float32).copy()
    roi = out[y1:y2, x1:x2]
    roi[:] = sig.astype(np.float32) * a + roi * (1 - a)

    # 边缘微光：真实拍摄的物体边缘常有一圈更亮/更暗的过渡
    if edge_glow:
        g = cv2.GaussianBlur(soft, (3, 3), 0).astype(np.float32) / 255.0
        ring = np.clip(g - a[:, :, 0], 0, 1) * 0.18     # 只在边缘环上生效
        # 边缘朝环境亮度偏一点
        edge_val = scene["bright"] * 0.9
        roi[:] = roi * (1 - ring[:, :, None] * 0.5) + edge_val * (ring[:, :, None] * 0.5)

    out[y1:y2, x1:x2] = np.clip(roi, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8), {
        "ok": True, "scene": {k: round(v, 1) for k, v in scene.items()},
        "target_brightness": round(scene["bright"] * (0.95 - 0.15 * scene["haze"]), 1),
    }


if __name__ == "__main__":
    print("sign_realism2：环境氛围匹配 + 边缘过渡")
