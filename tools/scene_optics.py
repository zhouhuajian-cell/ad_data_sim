#!/usr/bin/env python3
"""Scene Optics Probe —— 统一场景光学探针。

纯 OpenCV/NumPy，<3ms。同时供 TSR / OBJ / GND 使用。

核心思路：
  不依赖天气标签，直接从原图和局部 ROI 连续估计物理光学量：
    ① 雾度（暗通道先验）         → haze_blend / sat_scale
    ② 清晰度（拉普拉斯方差）     → blur_sigma
    ③ 昼夜（全图亮度）           → is_night（夜间逆反射）
    ④ 色温（LAB A/B 均值）       → color pull
"""
import cv2
import numpy as np


def estimate_local_optics(orig_img_bgr, orig_roi_bgr=None):
    """从原图和局部 ROI 连续估计物理光学参数。

    orig_img_bgr: 整帧原图（判断昼夜用）
    orig_roi_bgr: 局部区域（算雾度/清晰度/色温）；None 时用整图
    """
    if orig_roi_bgr is None:
        orig_roi_bgr = orig_img_bgr

    # ① 雾度 / 空气散射（暗通道先验）
    #    晴天清晰图暗通道 P10 通常 < 25；重度雨雾天 > 70
    min_channel = np.min(orig_roi_bgr, axis=2)
    dark_floor = float(np.percentile(min_channel, 10))
    haze_score = float(np.clip((dark_floor - 25.0) / 75.0, 0.0, 1.0))

    # ② 清晰度（拉普拉斯方差）
    gray_roi = cv2.cvtColor(orig_roi_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
    blur_sigma = float(np.clip(1.3 - (np.log1p(lap_var) / 7.5), 0.25, 1.3))

    # ③ 昼夜（全图平均亮度）
    global_luma = float(np.mean(
        cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2GRAY)))
    is_night = global_luma < 55.0

    # ④ 环境色温（LAB A/B 均值）
    lab = cv2.cvtColor(orig_roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_mean = float(lab[:, :, 1].mean())
    b_mean = float(lab[:, :, 2].mean())

    # 连续映射出自适应融合参数
    return {
        "is_night": bool(is_night),
        "global_luma": round(global_luma, 1),
        "haze_score": round(haze_score, 3),
        "dark_floor": round(dark_floor, 1),
        "haze_blend": round(haze_score * 0.24, 3),      # 晴天≈0，重雾≈0.24
        "sat_scale": round(0.95 - haze_score * 0.35, 3),  # 晴天≈0.95，重雾≈0.60
        "blur_sigma": round(blur_sigma, 2),
        "lap_var": round(lap_var, 1),
        "a_mean": round(a_mean, 1),
        "b_mean": round(b_mean, 1),
    }


def adaptive_tsr_harmonize(orig_img_bgr, orig_roi_bgr, syn_sign_bgr,
                           syn_alpha):
    """全场景自适应 TSR 融合。

    晴天：鲜艳锐利（haze≈0、卫星≈0、blur≈0.3）
    雨雾：自动压暗起雾（haze→0.24、sat→0.60、blur→1.2）
    夜间：保留逆反射高亮，不压成死黑

    返回 (out_roi_bgr, opt)
    """
    h, w = orig_roi_bgr.shape[:2]
    if syn_sign_bgr.shape[:2] != (h, w):
        syn_sign_bgr = cv2.resize(syn_sign_bgr, (w, h),
                                  interpolation=cv2.INTER_LINEAR)
    if syn_alpha.shape[:2] != (h, w):
        syn_alpha = cv2.resize(syn_alpha.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_LINEAR)
    if syn_alpha.dtype != np.float32:
        syn_alpha = syn_alpha.astype(np.float32)
    if syn_alpha.max() > 1.5:
        syn_alpha = syn_alpha / 255.0

    mask = syn_alpha > 0.5
    if not np.any(mask):
        return orig_roi_bgr, estimate_local_optics(orig_img_bgr, orig_roi_bgr)

    opt = estimate_local_optics(orig_img_bgr, orig_roi_bgr)

    orig_lab = cv2.cvtColor(orig_roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    syn_lab = cv2.cvtColor(syn_sign_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 1. 亮度区间自适应对齐
    orig_L = orig_lab[:, :, 0][mask]
    syn_L = syn_lab[:, :, 0][mask]
    orig_p5, orig_p95 = np.percentile(orig_L, [5, 95])
    syn_p5, syn_p95 = np.percentile(syn_L, [5, 95])

    if opt["is_night"]:
        # 夜间门控：保留标牌逆反射高亮特性
        target_p95 = max(orig_p95, 165.0)
        target_p5 = min(orig_p5, 45.0)
    else:
        target_p5, target_p95 = orig_p5, orig_p95

    scale = float(np.clip(
        (target_p95 - target_p5) / (syn_p95 - syn_p5 + 1e-5), 0.25, 1.15))
    mapped_L = (syn_lab[:, :, 0] - syn_p5) * scale + target_p5
    syn_lab[:, :, 0] = np.clip(mapped_L, target_p5, target_p95)

    # 2. 饱和度与色温自适应
    for c in [1, 2]:
        orig_mean = float(np.mean(orig_lab[:, :, c][mask]))
        syn_lab[:, :, c] = (syn_lab[:, :, c] - 128.0) * \
            opt["sat_scale"] + 128.0
        color_pull = 0.15 + 0.25 * opt["haze_score"]
        syn_lab[:, :, c] = syn_lab[:, :, c] * \
            (1.0 - color_pull) + orig_mean * color_pull

    matched = cv2.cvtColor(np.clip(syn_lab, 0, 255).astype(np.uint8),
                           cv2.COLOR_LAB2BGR).astype(np.float32)

    # 3. 大气散射（晴天 haze≈0 自动跳过）
    if opt["haze_blend"] > 0.02:
        veil = np.percentile(orig_roi_bgr.reshape(-1, 3), 85,
                             axis=0).astype(np.float32)
        matched = matched * (1.0 - opt["haze_blend"]) + \
            veil * opt["haze_blend"]

    # 4. 清晰度匹配
    if opt["blur_sigma"] > 0.3:
        matched = cv2.GaussianBlur(matched, (0, 0),
                                   sigmaX=opt["blur_sigma"])
        syn_alpha = cv2.GaussianBlur(syn_alpha, (0, 0),
                                     sigmaX=opt["blur_sigma"] * 0.7)

    # 5. 融合
    a3 = np.clip(syn_alpha, 0.0, 1.0)[..., None]
    out = matched * a3 + orig_roi_bgr.astype(np.float32) * (1.0 - a3)

    opt["scale"] = round(scale, 3)
    opt["target_L"] = [round(float(target_p5), 1),
                       round(float(target_p95), 1)]
    return np.clip(out, 0, 255).astype(np.uint8), opt
