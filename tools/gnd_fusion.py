#!/usr/bin/env python3
"""GND 光影继承合成（方案落地）。

核心：
  1) LaMa 无痕擦除旧标线（膨胀 10px 包住模糊边缘）
  2) LAB 空间剥离宏观树影 + 微观沥青颗粒
  3) 新标线继承光影，消除"贴纸感"
"""
import cv2
import numpy as np


_LAMA = None


def get_lama():
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()
    return _LAMA


def erase_old_gnd_with_lama(img_bgr, old_gnd_mask, dilate_px=10):
    """LaMa 无痕擦除旧标线。

    img_bgr: 原图
    old_gnd_mask: 二值图，255 = 需擦除
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilate_px, dilate_px))
    dilated_mask = cv2.dilate(old_gnd_mask, kernel, iterations=1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image as _Img
    pil_img = _Img.fromarray(img_rgb)
    pil_mask = _Img.fromarray(dilated_mask)

    inpainted_pil = get_lama()(pil_img, pil_mask)
    inpainted_bgr = cv2.cvtColor(np.array(inpainted_pil), cv2.COLOR_RGB2BGR)

    # 只替换 mask 内部，其他像素 100% 保留
    mask_3ch = (dilated_mask > 127)[..., None]
    clean_bg = np.where(mask_3ch, inpainted_bgr, img_bgr)
    return clean_bg.astype(np.uint8)


def blend_new_gnd_with_shadow(clean_bg_bgr, new_gnd_bgr, new_gnd_alpha,
                              wear_level=0.15):
    """新标线继承树影 + 沥青纹理。

    clean_bg_bgr: 擦除后的干净路面 (H,W,3)
    new_gnd_bgr:  新标线 RGB (H,W,3)
    new_gnd_alpha: 新标线 alpha (H,W) float 0~1
    """
    h, w = clean_bg_bgr.shape[:2]
    if new_gnd_alpha.shape[:2] != (h, w):
        new_gnd_alpha = cv2.resize(new_gnd_alpha, (w, h))
    if new_gnd_bgr.shape[:2] != (h, w):
        new_gnd_bgr = cv2.resize(new_gnd_bgr, (w, h))

    bg_lab = cv2.cvtColor(clean_bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_bg = bg_lab[:, :, 0]

    # 宏观光照/树影
    L_shadow = cv2.GaussianBlur(L_bg, (51, 51), 0)

    mask_bool = new_gnd_alpha > 0.1
    if not np.any(mask_bool):
        return clean_bg_bgr
    road_ref_L = np.percentile(L_shadow[mask_bool], 75) + 1e-5
    shadow_ratio = np.clip(L_shadow / road_ref_L, 0.35, 1.15)

    # 微观沥青颗粒
    L_grain = L_bg - cv2.GaussianBlur(L_bg, (7, 7), 0)

    gnd_lab = cv2.cvtColor(new_gnd_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gnd_lab[:, :, 0] = np.clip(
        gnd_lab[:, :, 0] * shadow_ratio + L_grain * 0.45, 0, 255)
    # 色温向路面靠拢，去掉荧光感
    gnd_lab[:, :, 1] = gnd_lab[:, :, 1] * 0.85 + bg_lab[:, :, 1] * 0.15
    gnd_lab[:, :, 2] = gnd_lab[:, :, 2] * 0.85 + bg_lab[:, :, 2] * 0.15

    realistic = cv2.cvtColor(gnd_lab.astype(np.uint8),
                             cv2.COLOR_LAB2BGR).astype(np.float32)

    # 油漆磨损
    rng = np.random.default_rng(7)
    noise = cv2.resize(rng.random((max(h // 4, 2), max(w // 4, 2)))
                       .astype(np.float32), (w, h))
    eff = new_gnd_alpha * (1.0 - wear_level * noise)
    eff = np.clip(eff, 0.0, 0.92)[..., None]

    out = realistic * eff + clean_bg_bgr.astype(np.float32) * (1.0 - eff)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_lane_alpha(calib, cfg, shape):
    """生成新标线的 alpha（0~1）与颜色，供 blend 使用。

    用真值标定投影到地面，画到全尺寸 mask。
    """
    import sys
    sys.path.insert(0, "/opt/ue5_synth/tools")
    from gnd_calib import ground_to_pixel

    h, w = shape[:2]
    lanes = int(cfg.get("lanes", 3))
    dash = cfg.get("dash", True)
    edge_w = float(cfg.get("edge", 0.20))
    line_w = float(cfg.get("width", 0.15))
    paint = np.zeros((h, w), np.uint8)

    lane_wid = 3.5
    total = lanes * lane_wid
    ys = [-total / 2 + i * lane_wid for i in range(lanes + 1)]
    DASH_ON, DASH_GAP = 6.0, 9.0

    for i, y_m in enumerate(ys):
        is_edge = (i == 0 or i == len(ys) - 1)
        wdt = edge_w if is_edge else line_w
        dashed = (not is_edge) and dash
        prev = None
        for x_m in np.arange(4.0, 55.0, 0.5):
            if dashed and (x_m % (DASH_ON + DASH_GAP)) > DASH_ON:
                prev = None
                continue
            uvl = ground_to_pixel(calib, x_m, y_m + wdt / 2)
            uvr = ground_to_pixel(calib, x_m, y_m - wdt / 2)
            if uvl is None or uvr is None:
                prev = None
                continue
            cur = (uvl, uvr, x_m)
            if prev is not None and abs(cur[2] - prev[2]) <= 1.0:
                q = np.array([prev[0], cur[0], cur[1], prev[1]], np.float32)
                if np.all(np.isfinite(q)):
                    cv2.fillConvexPoly(paint, np.rint(q).astype(np.int32), 255)
            prev = cur

    alpha = (paint.astype(np.float32) / 255.0)
    return alpha, paint


def build_old_mask(image, seg, dilate_px=10):
    """旧标线擦除掩码（语义 + 严格颜色）。"""
    h, w = image.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for k in (1, 2, 3):
        if seg is not None:
            m[seg == k] = 255

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ground = np.zeros((h, w), bool)
    ground[int(h * 0.55):, :] = True
    white = (V > 195) & (S < 45) & ground
    yellow = (H > 20) & (H < 36) & (S > 120) & (V > 150) & ground
    color = (white | yellow).astype(np.uint8) * 255

    if (m > 0).any():
        near = cv2.dilate(m, np.ones((51, 51), np.uint8))
        color = cv2.bitwise_and(color, near)
    else:
        color[:] = 0
    m = cv2.bitwise_or(m, color)

    # 只保留细长结构
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros((h, w), np.uint8)
    max_px = int(h * w * 0.04)
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        if area < 12:
            continue
        ratio = max(ww, hh) / max(min(ww, hh), 1)
        if ratio > 1.8 and area < max_px:
            keep[lab == i] = 255
    return keep
