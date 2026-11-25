"""OBJ 2D 贴片渲染的三处修复：

1. 接地：贴片按 alpha 底边裁紧，quad 底边 = 轮胎着地点
2. 不透明：车身 α 强制 255，只有边缘 1~2px 羽化
3. 接触阴影：阴影由 alpha 底边生成并向上柔和衰减（不是正圆盘）
"""
import cv2
import numpy as np


def crop_sprite_by_alpha(sp):
    """接地 FIX-1：按 alpha>0 裁紧贴片，去掉透明边距。

    返回 (cropped_rgba, (x0, y0, w, h))
    """
    if sp is None or sp.ndim != 3 or sp.shape[2] != 4:
        return sp, (0, 0, 0, 0)
    a = sp[:, :, 3]
    ys, xs = np.where(a > 8)
    if len(ys) == 0:
        return sp, (0, 0, 0, 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop = sp[y0:y1 + 1, x0:x1 + 1]
    return crop, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def solidify_alpha(cropped, edge_px=1, feather_px=1):
    """不透明 FIX-2：车身主体 alpha 拉满 255，仅外侧羽化。

    做法：对 alpha>0 的区域做形态学闭运算填内部空洞，
    再把内部（距边缘 > feather_px）强制 255。
    """
    if cropped is None or cropped.shape[2] != 4:
        return cropped
    out = cropped.copy()
    a = out[:, :, 3]
    binm = (a > 8).astype(np.uint8)

    # 填内部空洞（车身玻璃/轮廓线处 alpha 可能 < 255）
    k = np.ones((2 * edge_px + 1, 2 * edge_px + 1), np.uint8)
    filled = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)

    # 内部区域 = 腐蚀后的 filled
    if feather_px > 0:
        k2 = np.ones((2 * feather_px + 1, 2 * feather_px + 1), np.uint8)
        inner = cv2.erode(filled, k2, iterations=1)
    else:
        inner = filled

    amax = a.max()
    if amax > 0:
        out[:, :, 3] = np.where(inner > 0, 255,
                                (a.astype(np.float32) * (255.0 / amax))
                                ).astype(np.uint8)
    return out


def contact_shadow(h, w, quad, strength=1.0, opacity=0.32,
                   spread=0.18, rise=0.10):
    """接触阴影 FIX-3：紧贴轮胎底边的柔和接触阴影。

    不是正圆盘，而是：
      - 沿四边形底边展开的带状
      - 向上（远处）柔和衰减
      - 左右两侧随距离衰减
    阴天场景下接触阴影很窄（spread 小），不应明显外扩。
    """
    shm = np.zeros((h, w), np.float32)
    base = np.asarray(quad, dtype=np.float32)
    if base.shape[0] < 4:
        return shm

    p2 = base[2]   # 右下
    p3 = base[3]   # 左下
    cx = (p2[0] + p3[0]) / 2.0
    cy = (p2[1] + p3[1]) / 2.0
    half = abs(p2[0] - p3[0]) / 2.0          # 底边半宽
    if half < 2:
        return shm

    # 带状参数
    band_w = half * (1.0 + spread)            # 左右扩展
    band_h = max(half * 0.20, 4.0)            # 上下高度（窄）
    tread = max(half * 0.06, 1.5)             # 胎面厚度（更暗）

    # 1) 主接触带：沿底边的一条窄带
    pt1 = (int(max(cx - band_w, 0)), int(cy))
    pt2 = (int(min(cx + band_w, w - 1)), int(cy))
    thick = max(int(tread * 2), 3)
    cv2.line(shm, pt1, pt2, 1.0, thick)

    # 2) 向上柔和衰减（模拟漫射）
    rise_px = max(int(band_h + tread * 3), 4)
    up = np.zeros((h, w), np.float32)
    cv2.line(up, (int(cx - band_w * 0.85), int(cy - rise_px * 0.5)),
             (int(cx + band_w * 0.85), int(cy - rise_px * 0.5)),
             1.0, max(int(rise_px), 3))
    up = cv2.GaussianBlur(up, (0, 0), sigmaX=max(band_w * 0.35, 2.0),
                          sigmaY=max(rise_px * 0.45, 2.0))
    shm = np.maximum(shm, up * 0.55)

    # 3) 轻微整体模糊，避免硬边
    shm = cv2.GaussianBlur(shm, (0, 0), sigmaX=max(band_w * 0.10, 1.2),
                           sigmaY=max(tread * 0.8, 1.0))

    m = float(shm.max())
    if m > 1e-6:
        shm = shm / m
    return shm * (opacity * strength)


def composite_quad(out_im, sprite_cropped, quad, opacity_cap=1.0):
    """把贴片按四边形贴到图中。alpha 不额外乘全局系数。"""
    h, w = out_im.shape[:2]
    sh, sw = sprite_cropped.shape[:2]
    src = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])
    dst = np.float32(quad)
    Pm = cv2.getPerspectiveTransform(src, dst)

    a = sprite_cropped[:, :, 3].astype(np.float32) / 255.0
    rgb = sprite_cropped[:, :, :3].astype(np.float32)
    prem = rgb * a[..., None]

    a_w = cv2.warpPerspective(a, Pm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
    p_w = cv2.warpPerspective(prem, Pm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    a_w = np.clip(a_w, 0, opacity_cap)
    straight = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
    straight[a_w <= 1e-4] = 0
    return a_w, straight
