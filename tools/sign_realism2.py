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
    "bright_gain": 0.62,     # 目标亮度 = 环境亮分位 × gain（调小变暗）
    "haze_gain": 0.75,       # 雾度强度（调小更实、对比更强）
    "contrast_gain": 1.15,   # 对比度相对环境的倍数（调小更灰）
    "satur_keep": 0.70,      # 雾天饱和度保留比例（调小更灰）
    "feather_ratio": 0.025,  # 边缘羽化 = 标牌短边 × 该比例
                             # 实测 0.08 时 66px 的牌羽化近 5px，糊出一圈灰晕
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
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            p = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)

    # ---- D. 软掩码：按原标牌的实际边缘过渡宽度羽化 ----
    soft = mask.astype(np.float32) / 255.0
    # 用小标牌的尺寸决定羽化半径：越小的牌，边缘相对越糊
    h, w = soft.shape
    # 羽化必须克制：标牌只有 40~70px，羽化超过 2px 就糊成一圈灰晕。
    # 实测 0.08（66px 牌 → 4.7px）明显发虚，收到 0.025 且硬上限 2px。
    feather_px = min(2.0, max(0.8, min(h, w) * PARAMS["feather_ratio"]))
    kf = int(max(3, 2 * round(feather_px * 1.5) + 1))
    soft = cv2.GaussianBlur(soft, (kf, kf), feather_px)

    return np.clip(p, 0, 255).astype(np.uint8), (np.clip(soft, 0, 1) * 255).astype(np.uint8)


def _jpeg_quality_for(ref_patch):
    """按原图局部清晰度决定 JPEG 质量。

    用户建议：原图很糊 → 45~50；偏高清 → 65~75。
    用拉普拉斯方差衡量，经验映射：
      < 120   很糊   → 46
      120~400 一般   → 56
      400~900 较清晰 → 66
      > 900   高清   → 74
    """
    import cv2
    if ref_patch is None or ref_patch.size == 0:
        return 56
    g = cv2.cvtColor(ref_patch.astype("uint8"), cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    # 阈值实测校准（180 个真实标牌样本）：
    #   lap 中位 6151、5% 分位 1772、95% 分位 17417
    # 标牌文字边缘锐利，拉普拉斯方差天然很大，
    # 之前按 120/400/900 设阈值导致 99% 都被判成"高清"。
    # 现按实测 25/50/75 分位重设：4140 / 6151 / 9751
    if lap < 2500:
        return 46
    if lap < 4140:
        return 52
    if lap < 6151:
        return 58
    if lap < 9751:
        return 66
    return 74


def apply_jpeg_artifact(blended, mask, quality):
    """在融合之后的成品上打 JPEG 压缩伪影（只作用于标牌及外扩 2px）。

    为什么放在融合之后：
      放在前面的话，压缩块会被边缘 alpha 再削一遍，
      而且伪影不会外溢到牌的边缘外侧 —— 缺少车载视频那种
      「标牌周围一圈也糊」的观感。放在后面才对得上真实编码顺序。
    """
    import cv2
    import numpy as np

    if blended is None or blended.size == 0:
        return blended
    ok, buf = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return blended
    compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # 只在标牌及外扩 2px 范围内应用，避免污染周围真实背景
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(mask, k, iterations=1)
    soft = (cv2.GaussianBlur(dil, (3, 3), 0).astype(np.float32) / 255.0)[:, :, None]
    out = compressed.astype(np.float32) * soft + blended.astype(np.float32) * (1.0 - soft)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_v3(sign_bgr, mask, ref_patch, scene, strength=1.0):
    """劣化 v3 —— 借鉴 Gemini 方案的三个改进点。

    相比 v2 的改动：

    1) L 通道用 P5~P95 黑白点线性映射，而不是均值+标准差。
       均值会被牌面上的极端值（黑字、白底）带偏；
       分位数映射的是「实际动态范围」，更接近真实。

    2) 模糊按拉普拉斯方差迭代搜索匹配，而不是按固定公式估 sigma。
       原图多糊，贴的牌就多糊 —— 用二分迭代逼近，误差小得多。

    3) 噪点从原图的高频残差里估，而不是拍一个常数。
       残差 = 原图 - 中值滤波，它的标准差就是真实噪点水平。

    4) 色温（LAB 的 A/B 通道）做软迁移，权重 0.35，
       避免红圈被环境色洗白。
    """
    import cv2
    import numpy as np

    if ref_patch.shape[:2] != sign_bgr.shape[:2]:
        ref_patch = cv2.resize(ref_patch, (sign_bgr.shape[1], sign_bgr.shape[0]))

    valid = mask > 127
    if valid.sum() < 20:
        return sign_bgr, mask

    p = sign_bgr.astype(np.float32)

    # ---- 1) LAB 空间的 L 通道黑白点映射 ----
    bg_lab = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    as_lab = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_l = bg_lab[:, :, 0][valid]
    as_l = as_lab[:, :, 0][valid]
    p5_bg, p95_bg = np.percentile(bg_l, [5, 95])
    p5_as, p95_as = np.percentile(as_l, [5, 95])

    # 目标亮度：跟随环境，但受 brightness_gain 控制
    gain = PARAMS["bright_gain"]
    scene_bright = scene.get("bright", 150.0)
    target_mid = scene_bright * gain
    # 把原牌的中位亮度拉到 target_mid 附近
    bg_mid = (p5_bg + p95_bg) * 0.5
    shift = target_mid - bg_mid

    scale_l = (p95_bg - p5_bg) / (p95_as - p5_as + 1e-5)
    # 限制对比度缩放，避免过度拉伸
    scale_l = float(np.clip(scale_l, 0.60, 1.8))
    as_lab[:, :, 0] = (as_lab[:, :, 0] - p5_as) * scale_l + p5_bg + shift
    as_lab[:, :, 0] = np.clip(as_lab[:, :, 0], 0, 255)

    # ---- 4) 色温软迁移 ----
    for ch in (1, 2):
        diff = float(np.mean(bg_lab[:, :, ch][valid]) - np.mean(as_lab[:, :, ch][valid]))
        as_lab[:, :, ch] = np.clip(as_lab[:, :, ch] + diff * 0.35, 0, 255)

    p = cv2.cvtColor(as_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    # ---- 雾度：暗部抬升（沿用 v2 的环境雾估计） ----
    hz = scene.get("haze", 0.3) * strength * PARAMS["haze_gain"]
    if hz > 0.02:
        k = hz * 0.30
        p = p * (1 - k) + 255 * 0.82 * k

    # ---- 饱和度：按背景 P90 自适应压制（P10_P90 补丁） ----
    # 原做法用全局 satur_keep 系数，遇到"阴天灰场景 + 鲜红圈"时压不够；
    # 改成用背景高饱和区的 P90 作上限，只压不提，防止把素材拉到过饱和。
    bg_hsv = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    cur_hsv = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    bg_s90 = float(np.percentile(bg_hsv[:, :, 1][valid], 90))
    as_s90 = float(np.percentile(cur_hsv[:, :, 1][valid], 90))
    if as_s90 > 1e-3:
        _full = min(1.0, (bg_s90 * 1.05) / as_s90)
        # 压制减半：实测全压让 S_P90 从 154.6 掉到 114（-26%），
        # 原牌红圈的鲜艳感被磨没了，看着假。只压 40%。
        sat_scale = 1.0 - (1.0 - _full) * 0.4
        # 与全局 satur_keep 取较小值，保证不会比原来更饱和
        sat_scale = min(sat_scale, 1.0)
        cur_hsv[:, :, 1] = np.clip(cur_hsv[:, :, 1] * sat_scale, 0, 255)
        p = cv2.cvtColor(cur_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # ---- 2) 模糊迭代匹配 ----
    bg_gray = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    bg_lap = float(np.var(cv2.Laplacian(bg_gray, cv2.CV_64F)[valid]))
    target_lap = bg_lap * 1.10          # 允许比原图略清晰 10%
    best = p.copy()
    for sigma in np.linspace(0.0, 2.5, 12):
        if sigma < 0.15:
            cur = p
        else:
            ksz = int(2 * round(2 * sigma) + 1)
            cur = cv2.GaussianBlur(p, (ksz, ksz), sigmaX=sigma)
        g = cv2.cvtColor(np.clip(cur, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        lap = float(np.var(cv2.Laplacian(g, cv2.CV_64F)[valid]))
        best = cur
        if lap <= target_lap:
            break
    p = best

    # ---- 3) 噪点从原图高频残差估 ----
    bg_smooth = cv2.medianBlur(ref_patch.astype(np.uint8), 3)
    residual = (ref_patch.astype(np.float32) - bg_smooth.astype(np.float32))[valid]
    noise_std = float(np.std(residual)) * 0.85
    if noise_std > 1.0:
        noise_std = min(noise_std, 8.0)
        p = p + np.random.normal(0, noise_std, p.shape).astype(np.float32)

    # ---- 边缘 alpha：先腐蚀 1px 去掉原牌边缘残留，再羽化 ----
    soft = mask.astype(np.float32) / 255.0
    er = cv2.erode((soft * 255).astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    h, w = soft.shape
    feather_px = min(2.0, max(0.8, min(h, w) * PARAMS["feather_ratio"]))
    kf = int(max(3, 2 * round(feather_px * 1.5) + 1))
    soft = cv2.GaussianBlur(er.astype(np.float32) / 255.0, (kf, kf), feather_px)

    return np.clip(p, 0, 255).astype(np.uint8), (np.clip(soft, 0, 1) * 255).astype(np.uint8)


def paste_v2(image, sign_bgr, sign_alpha, bbox, strength=1.0, edge_glow=True,
             use_v3=False):
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
    if use_v3:
        sig, soft = degrade_v3(sig, alp, ref, scene, strength)
    else:
        sig, soft = degrade_v2(sig, alp, ref, scene, strength)

    a = (soft.astype(np.float32) / 255.0)[:, :, None]
    out = image.astype(np.float32).copy()
    roi = out[y1:y2, x1:x2]
    roi[:] = sig.astype(np.float32) * a + roi * (1 - a)

    # 边缘微光：真实拍摄的物体边缘常有一圈更亮/更暗的过渡
    if edge_glow:
        g = cv2.GaussianBlur(soft, (3, 3), 0).astype(np.float32) / 255.0
        ring = np.clip(g - a[:, :, 0], 0, 1) * 0.06     # 只在边缘环上生效（0.18 会糊出灰边）
        # 边缘朝环境亮度偏一点
        edge_val = scene["bright"] * 0.9
        roi[:] = roi * (1 - ring[:, :, None] * 0.35) + edge_val * (ring[:, :, None] * 0.35)

    out[y1:y2, x1:x2] = np.clip(roi, 0, 255)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # ③ 融合之后再打 JPEG 压缩伪影（用户指定的顺序）
    jq = _jpeg_quality_for(ref)
    full_mask = np.zeros(out.shape[:2], np.uint8)
    full_mask[y1:y2, x1:x2] = soft.astype(np.uint8)
    out = apply_jpeg_artifact(out, full_mask, jq)

    return out, {
        "ok": True, "scene": {k: round(v, 1) for k, v in scene.items()},
        "jpeg_quality": jq,
        "target_brightness": round(scene["bright"] * (0.95 - 0.15 * scene["haze"]), 1),
    }


if __name__ == "__main__":
    print("sign_realism2：环境氛围匹配 + 边缘过渡")
