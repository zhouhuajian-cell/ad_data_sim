#!/usr/bin/env python3
"""合成核心：线性空间 + 保护区 + 边界环校验（融合方案）。

设计：
  - 所有 alpha 合成都走线性空间（消除 sRGB 色偏）
  - protected / allowed 双掩码保护不可动区域
  - donor 背景替换带边界一致性校验，不达标拒绝
  - GND 标线继承树影与沥青颗粒
"""
import cv2
import numpy as np


# ---------- 色彩空间 ----------

def to_linear(x):
    """sRGB → 线性。"""
    x = x.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    """线性 → sRGB。"""
    x = np.clip(x, 0.0, 1.0)
    s = np.where(x <= 0.0031308, x * 12.92,
                 1.055 * x ** (1 / 2.4) - 0.055)
    return np.rint(np.clip(s, 0, 1) * 255).astype(np.uint8)


# ---------- 通用合成 ----------

def composite_linear(base_bgr, layer_rgb, alpha, protected=None,
                     allowed=None):
    """线性空间合成。alpha: (H,W) 0~1。"""
    h, w = base_bgr.shape[:2]
    a = alpha.astype(np.float32)
    if a.shape[:2] != (h, w):
        a = cv2.resize(a, (w, h))

    if allowed is not None:
        a = a * (allowed.astype(np.float32))
    if protected is not None:
        a = a * (1.0 - protected.astype(np.float32))

    a3 = a[..., None]
    out = to_linear(base_bgr) * (1 - a3) + to_linear(layer_rgb) * a3
    res = to_srgb(out)
    # 完全透明处像素 100% 保留（避免舍入误差）
    keep = (a <= 1e-4)
    res[keep] = base_bgr[keep]
    return res, a


# ---------- 邻帧背景替换 ----------

def restore_from_donors(target, erase, protected, donors,
                        max_ring_error=20, ring_px=11):
    """用邻帧恢复 erase 区域的背景。

    donors: [(bgr_image, valid_mask, H_donor_to_target)]
    返回 (restored, covered_mask, diagnostics)
    """
    h, w = target.shape[:2]
    if erase.shape != (h, w) or protected.shape != (h, w):
        raise ValueError("mask 尺寸不匹配")
    if np.any(erase & protected):
        raise ValueError("erase 与 protected 重叠，请修正掩码")

    # 边界环：擦除区外 5px
    ring = cv2.dilate(erase.astype(np.uint8),
                      np.ones((ring_px, ring_px), np.uint8)) > 0
    ring &= ~erase & ~protected

    candidates = []
    validities = []
    diags = []
    for image, valid, H in donors:
        H = np.asarray(H, dtype=float)
        if H.shape != (3, 3) or not np.isfinite(H).all():
            diags.append({"accepted": False, "reason": "bad_homography"})
            continue
        if valid.shape != image.shape[:2]:
            diags.append({"accepted": False, "reason": "mask_mismatch"})
            continue

        warped = cv2.warpPerspective(image, H, (w, h),
                                     flags=cv2.INTER_LINEAR)
        v = cv2.warpPerspective(valid.astype(np.float32), H, (w, h),
                                flags=cv2.INTER_LINEAR) > 0.999
        sample = ring & v
        if sample.sum() < 100:
            diags.append({"accepted": False,
                          "reason": "insufficient_ring"})
            continue

        # 边界环颜色偏移校正
        offset = np.median(target[sample].astype(float)
                           - warped[sample], axis=0)
        offset = np.clip(offset, -15, 15)
        corrected = np.clip(warped.astype(float) + offset, 0, 255)

        error = float(np.median(np.abs(target[sample].astype(float)
                                      - corrected[sample])))
        ok = error <= max_ring_error
        diags.append({"accepted": ok, "ring_error": round(error, 2)})
        if ok:
            candidates.append(corrected)
            validities.append(v)

    if not candidates:
        return target.copy(), np.zeros((h, w), bool), {
            "donors": diags, "accepted": 0}

    stack = np.stack(candidates)
    vstack = np.stack(validities)
    stack = np.where(vstack[..., None], stack, np.nan)

    count = vstack.sum(axis=0)
    stack[:, count == 0, :] = target[count == 0]
    median = np.nanmedian(stack, axis=0)

    covered = erase & (count > 0)
    result = target.copy()
    result[covered] = np.rint(median[covered]).astype(np.uint8)
    return result, covered, {"donors": diags,
                             "accepted": len(candidates)}


# ---------- GND 光影继承 ----------

def blend_gnd_with_shadow(clean_bg_bgr, line_rgb, alpha, wear_level=0.15,
                          protected=None, allowed=None,
                          opacity=0.82, shadow_falloff=0.85):
    """新标线继承树影 + 沥青颗粒（线性空间）。

    POLISH-OPACITY-SHADOW:
      opacity:        全局不透明度（默认 0.82，原 1.0 太"新"）
      shadow_falloff: 树影区额外衰减系数（默认 0.85）
    """
    h, w = clean_bg_bgr.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h))
    if line_rgb.shape[:2] != (h, w):
        line_rgb = cv2.resize(line_rgb, (w, h))

    bg_lab = cv2.cvtColor(clean_bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_bg = bg_lab[:, :, 0]

    # 宏观树影
    L_shadow = cv2.GaussianBlur(L_bg, (51, 51), 0)
    mask_bool = alpha > 0.1
    if not np.any(mask_bool):
        return clean_bg_bgr, alpha
    road_ref = np.percentile(L_shadow[mask_bool], 75) + 1e-5

    # POLISH: shadow_ratio 下限从 0.35 压到 0.28（树影里更暗）
    shadow_ratio = np.clip(L_shadow / road_ref, 0.28, 1.15)

    # 微观沥青
    L_grain = L_bg - cv2.GaussianBlur(L_bg, (7, 7), 0)

    line_lab = cv2.cvtColor(line_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    line_lab[:, :, 0] = np.clip(
        line_lab[:, :, 0] * shadow_ratio + L_grain * 0.45, 0, 255)
    # 色温向路面靠拢
    line_lab[:, :, 1] = line_lab[:, :, 1] * 0.85 + bg_lab[:, :, 1] * 0.15
    line_lab[:, :, 2] = line_lab[:, :, 2] * 0.85 + bg_lab[:, :, 2] * 0.15
    realistic = cv2.cvtColor(line_lab.astype(np.uint8),
                             cv2.COLOR_LAB2BGR)

    # 油漆磨损
    rng = np.random.default_rng(7)
    noise = cv2.resize(
        rng.random((max(h // 4, 2), max(w // 4, 2))).astype(np.float32),
        (w, h))
    eff = alpha * (1.0 - wear_level * noise)

    # POLISH: 树影区额外衰减（L_bg 低于 road_ref 时乘 falloff）
    shadow_dark = np.clip(L_shadow / road_ref, 0.5, 1.0)
    shadow_gain = np.where(shadow_dark < 0.85,
                           1.0 - (1.0 - shadow_dark) * (1.0 - shadow_falloff),
                           1.0).astype(np.float32)
    eff = eff * float(opacity) * shadow_gain

    eff = np.clip(eff, 0.0, 0.92)

    return composite_linear(clean_bg_bgr, realistic, eff,
                            protected=protected, allowed=allowed)


# ---------- 空位直插（OBJ add 模式核心）----------

def find_spawn_region(drivable, existing_boxes, margin=15, check=(60, 40)):
    """在可行驶区域内找空位。

    drivable: (H,W) bool/uint8，路面区域
    existing_boxes: [[x1,y1,x2,y2], ...] 已有目标（含阴影外扩）
    返回 (H,W) uint8 合法放置掩码
    """
    h, w = drivable.shape[:2]
    valid = (drivable > 0).astype(np.uint8) * 255
    for bb in existing_boxes:
        x1 = max(0, int(bb[0]) - margin)
        y1 = max(0, int(bb[1]) - margin)
        x2 = min(w, int(bb[2]) + margin)
        y2 = min(h, int(bb[3]) + margin)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(valid, (x1, y1), (x2, y2), 0, -1)

    # 腐蚀：保证离边缘/其他车有安全距离
    k = cv2.getStructuringElement(cv2.MORPH_RECT, check)
    safe = cv2.erode(valid, k)
    return safe


def build_drivable_from_masks(shape, freespace=None, road_hint=None):
    """从 freespace mask 或简单几何构造可行驶区。"""
    h, w = shape[:2]
    if freespace is not None:
        f = cv2.resize(freespace, (w, h),
                       interpolation=cv2.INTER_NEAREST)
        return (f > 0).astype(np.uint8) * 255
    # 兜底：画面下 40% 视为路面
    m = np.zeros((h, w), np.uint8)
    m[int(h * 0.60):, :] = 255
    return m

def blend_new_gnd_realistic(
    clean_bg_bgr,
    new_gnd_bgr,
    new_gnd_alpha,
    wear_level=0.35,
    dirt_level=0.25,
    optical_blur=1.2,
):
    """物理级 GND 做旧融合（用户方案）。

    修 4 个"一眼假"破绽：
      ① 油漆太白      → 压暗到 0.85 + 继承环境色温
      ② 边缘太锐      → optical_blur 高斯
      ③ 表面太平      → 沥青颗粒透底 + 灰尘/胎印斑块
      ④ 树影不够暗    → shadow_ratio ** 1.35
    """
    h, w = clean_bg_bgr.shape[:2]
    mask_bool = new_gnd_alpha > 0.05
    if not np.any(mask_bool):
        return clean_bg_bgr

    # 1. 原路面宏观光照(树影) + 微观沥青颗粒
    bg_lab = cv2.cvtColor(clean_bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_bg = bg_lab[:, :, 0]
    L_shadow = cv2.GaussianBlur(L_bg, (61, 61), 0)
    L_grain = L_bg - cv2.GaussianBlur(L_bg, (5, 5), 0)

    road_sun_L = np.percentile(L_shadow[mask_bool], 90) + 1e-5
    shadow_ratio = np.clip((L_shadow / road_sun_L) ** 1.35, 0.25, 1.05)

    # 2. 油漆去荧光白 + 继承树影 + 色温
    gnd_lab = cv2.cvtColor(new_gnd_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gnd_lab[:, :, 0] = np.clip(
        gnd_lab[:, :, 0] * 0.85 * shadow_ratio + L_grain * 0.7, 0, 255)
    gnd_lab[:, :, 1] = gnd_lab[:, :, 1] * 0.65 + bg_lab[:, :, 1] * 0.35
    gnd_lab[:, :, 2] = gnd_lab[:, :, 2] * 0.65 + bg_lab[:, :, 2] * 0.35

    # 3. 低频胎印/灰尘脏污
    dirt_noise = cv2.resize(
        np.random.rand(max(h // 16, 2), max(w // 16, 2)).astype(np.float32),
        (w, h))
    dirt_noise = cv2.GaussianBlur(dirt_noise, (15, 15), 0)
    gnd_lab[:, :, 0] = np.clip(
        gnd_lab[:, :, 0] * (1.0 - dirt_level * dirt_noise), 0, 255)

    realistic_gnd = cv2.cvtColor(gnd_lab.astype(np.uint8),
                                 cv2.COLOR_LAB2BGR).astype(np.float32)

    # 4. 油漆剥落 + 粗糙沥青透底
    fine_wear = cv2.resize(
        np.random.rand(max(h // 2, 2), max(w // 2, 2)).astype(np.float32),
        (w, h))
    coarse_wear = cv2.resize(
        np.random.rand(max(h // 12, 2), max(w // 12, 2)).astype(np.float32),
        (w, h))
    wear_map = 0.6 * fine_wear + 0.4 * coarse_wear
    asphalt_pit = np.clip(-L_grain / 20.0, 0, 1.0) * 0.25

    eff_alpha = new_gnd_alpha * (1.0 - wear_level * wear_map - asphalt_pit)
    eff_alpha = np.clip(eff_alpha, 0.0, 0.82).astype(np.float32)

    # 5. 镜头光学退化
    if optical_blur > 0:
        realistic_gnd = cv2.GaussianBlur(realistic_gnd, (0, 0),
                                         sigmaX=optical_blur)
        eff_alpha = cv2.GaussianBlur(eff_alpha, (0, 0),
                                     sigmaX=optical_blur)

    # 6. 传感器 ISO 噪点
    bg_float = clean_bg_bgr.astype(np.float32)
    bg_noise_std = float(np.std(
        bg_float - cv2.GaussianBlur(bg_float, (3, 3), 0)))
    sensor_noise = np.random.normal(
        0, bg_noise_std * 0.8, realistic_gnd.shape).astype(np.float32)
    realistic_gnd = np.clip(realistic_gnd + sensor_noise, 0, 255)

    # 7. 合成
    a3 = eff_alpha[..., None]
    out = realistic_gnd * a3 + bg_float * (1.0 - a3)
    return np.clip(out, 0, 255).astype(np.uint8)


