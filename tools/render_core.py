#!/usr/bin/env python3
"""完整吸收方案的渲染核心。

模块：
  1) drivable_from_gnd    —— 从 GND freespace_mask 或几何推断路面
  2) shadow_mask_from_sprite —— 从精灵 alpha 推导接触阴影
  3) composite_multiply   —— 正片叠底阴影 + 线性空间合成
  4) build_protected      —— 从检测框生成保护区
  5) validate_spawn       —— 物理尺寸 + 遮挡门控
  6) ring_check           —— 边界环一致性
"""
import cv2
import numpy as np


# ---------- 1) 可行区域 ----------

def drivable_from_gnd_mask(gnd_mask_path, shape):
    """从 GND freespace_mask 构造可行驶区。

    freespace_mask 值：0=不可行, 70/130/180=不同等级可行驶
    """
    h, w = shape[:2]
    m = cv2.imread(gnd_mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    # 非 0 视为可行驶
    return (m > 0).astype(np.uint8) * 255


def drivable_from_calib(calib, shape, cam=None, max_dist=60.0):
    """DRV-COMPAT: 用标定投影生成路面掩码（兼容 OBJ/GND 两种结构）。"""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)

    # 判断 calib 类型
    is_gnd = calib is not None and "R_cam2body" in calib
    is_obj = cam is not None and "R" in cam and "t" in cam

    # DRV-POLY: 用多边形填充（避免采样空洞）
    def _poly_from(project):
        """project(x, y) -> (u,v) 或 None。沿左右边界投影成多边形。"""
        left, right = [], []
        for x in np.arange(3.0, max_dist, 0.5):
            pl = project(x, 15.0)     # 左边界
            pr = project(x, -15.0)    # 右边界
            if pl is not None and pr is not None:
                left.append([pl[0], pl[1]])
                right.append([pr[0], pr[1]])
        if len(left) < 3:
            return None
        poly = np.array(left + right[::-1], np.int32)
        mm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mm, [poly], 255)
        return mm

    if is_gnd:
        from gnd_calib import ground_to_pixel
        def _pj(x, y):
            p = ground_to_pixel(calib, x, y)
            if p is None:
                return None
            if not (np.isfinite(p[0]) and np.isfinite(p[1])):
                return None
            return p
        r = _poly_from(_pj)
        if r is not None:
            m = r
    elif is_obj:
        from real_calib import body_to_pixel_real
        def _pj2(x, y):
            uv, z, ok = body_to_pixel_real(
                cam, np.array([[x, y, 0.0]], float))
            if not ok[0] or not np.isfinite(uv[0]).all():
                return None
            return (float(uv[0][0]), float(uv[0][1]))
        r = _poly_from(_pj2)
        if r is not None:
            m = r
    else:
        return None

    if not np.any(m):
        return None
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         np.ones((25, 25), np.uint8))
    return m


def drivable_geometric(shape, horizon_ratio=0.52):
    """兜底：地平线以下视为路面。"""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    m[int(h * horizon_ratio):, :] = 255
    return m


def build_drivable(shape, gnd_mask_path=None, calib=None, cam=None):
    """按优先级构造可行驶区。"""
    m = None
    if gnd_mask_path:
        m = drivable_from_gnd_mask(gnd_mask_path, shape)
    if m is None:
        m = drivable_from_calib(calib, shape, cam=cam)
    if m is None:
        m = drivable_geometric(shape)
    return m


# ---------- 2) 接触阴影 ----------

def shadow_mask_from_sprite(sprite_rgba, soft=0.55):
    """从精灵 alpha 推导地面接触阴影。

    原理：取精灵底部 18% 的 alpha，向下投影成椭圆软阴影。
    """
    h, w = sprite_rgba.shape[:2]
    a = sprite_rgba[..., 3].astype(np.float32) / 255.0
    if a.max() < 0.05:
        return np.zeros((h, w), np.float32)

    band_h = max(int(h * 0.18), 4)
    band = a[h - band_h:, :]
    # 每列的覆盖宽度
    col = band.max(axis=0)

    sh = np.zeros((h, w), np.float32)
    # 阴影中心在下缘
    for x in range(w):
        if col[x] < 0.1:
            continue
        # 底行位置
        ys = np.where(band[:, x] > 0.15)[0]
        if len(ys) == 0:
            continue
        y_bottom = h - band_h + int(ys.max())
        rw = max(int(w * 0.09), 3)
        rh = max(int(h * 0.035), 2)
        cv2.ellipse(sh, (x, y_bottom), (rw, rh), 0, 0, 360,
                    float(col[x]), -1)

    if sh.max() < 1e-3:
        return sh
    k = max(int(min(h, w) * 0.12) | 1, 5)
    sh = cv2.GaussianBlur(sh, (k, k), 0)
    sh = np.clip(sh / max(sh.max(), 1e-6) * soft, 0.0, 1.0)
    return sh


def sprite_with_shadow(sprite_rgba, shadow_strength=0.65):
    """返回 (sp_rgba, shadow_alpha)。shadow 与 sprite 同尺寸。"""
    sh = shadow_mask_from_sprite(sprite_rgba)
    return sprite_rgba, sh * shadow_strength


# ---------- 3) 合成 ----------

def to_linear(x):
    x = x.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    s = np.where(x <= 0.0031308, x * 12.92,
                 1.055 * x ** (1 / 2.4) - 0.055)
    return np.rint(np.clip(s, 0, 1) * 255).astype(np.uint8)


def composite_multiply(bg_bgr, sprite_rgba, shadow_alpha, quad,
                       scene_light=None, depth_m=None,
                       protected=None, allowed=None,
                       shadow_strength=1.0):
    """正片叠底阴影 + 线性空间前景合成。

    1) 阴影：Multiply 到背景（保留沥青纹理）
    2) 前景：线性空间 alpha 混合 + 羽化
    """
    h, w = bg_bgr.shape[:2]
    sh_, sw_ = sprite_rgba.shape[:2]
    if sw_ < 4 or sh_ < 4:
        return bg_bgr

    quad = np.asarray(quad, np.float32)
    src = np.array([[0, 0], [sw_ - 1, 0], [sw_ - 1, sh_ - 1],
                    [0, sh_ - 1]], np.float32)
    P = cv2.getPerspectiveTransform(src, quad)

    tw = max(4, int(abs(quad[1][0] - quad[0][0])))
    th = max(4, int(abs(quad[3][1] - quad[0][1])))
    # NO-STRETCH-REMOVED: 不做裁剪（内容比例差 <15%，拉伸可接受）

    # ---- ① 阴影：正片叠底 ----
    sh_w = cv2.warpPerspective(shadow_alpha.astype(np.float32), P, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
    # SHADOW-ADAPT: 只在物体底边附近生效（垂直衰减）
    ys = quad[:, 1]
    y_base = float(ys.max())
    y_top = float(ys.min())
    yy = np.arange(h, dtype=np.float32)[:, None]
    span = max(y_base - y_top, 8.0)
    # SPRITE-CLEAN: 阴影只在底部 12%，更窄
    vfall = np.clip(1.0 - (y_base - yy) / (span * 0.12), 0.0, 1.0)
    vfall = (vfall ** 2.2)
    sh_w = sh_w * vfall
    sh_w = np.clip(sh_w * shadow_strength, 0.0, 1.0)
    if protected is not None:
        sh_w = sh_w * (1.0 - protected.astype(np.float32))
    if allowed is not None:
        sh_w = sh_w * allowed.astype(np.float32)

    base_lin = to_linear(bg_bgr)
    if sh_w.max() > 1e-3:
        # SHADOW-ADAPT: 自适应阴影强度（亮处多压、暗处少压）
        lum = bg_bgr.astype(np.float32).mean(axis=2)
        # 场景中位亮度做参考
        ref = float(np.median(lum[lum > 20])) if (lum > 20).any() else 110.0
        # 越亮的地方阴影越明显（真实物理：高光区域阴影对比大）
        adapt = np.clip(lum / max(ref, 1.0), 0.25, 1.35)
        strength = sh_w * adapt * 0.62
        factor = (1.0 - strength[..., None])
        base_lin = base_lin * factor

    # ---- ② 前景 ----
    # SPRITE-MILD-CLEAN: 温和清理（只去半透明噪点 + 轻羽化）
    a = sprite_rgba[..., 3].astype(np.float32) / 255.0
    rgb = sprite_rgba[..., :3].astype(np.float32)

    # 低于 0.25 的半透明视为背景
    a = np.where(a < 0.25, 0.0, a)
    # 轻微羽化边缘
    a = cv2.GaussianBlur(a, (3, 3), 0)

    # 光照匹配
    if scene_light:
        hsv = cv2.cvtColor(rgb.clip(0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        vis = hsv[..., 2] > 30
        cur = float(np.median(hsv[vis][:, 2])) if vis.any() else 150.0
        cur = max(cur, 1.0)
        tgt = np.clip(scene_light.get("lum", 120) * 1.05, 25, 235)
        rgb = rgb * np.clip(tgt / cur, 0.35, 2.2)
        g = scene_light.get("gain")
        if g is not None:
            rgb = rgb * np.asarray(g, np.float32)[None, None, :]

    # 景深模糊
    if depth_m and depth_m > 15:
        sg = min((depth_m - 15) / 40.0, 1.2)
        if sg > 0.2:
            rgb = cv2.GaussianBlur(rgb, (0, 0), sg)
            a = cv2.GaussianBlur(a, (0, 0), sg * 0.6)

    rgb = np.clip(rgb, 0, 255)
    prem = rgb * a[..., None]

    a_w = cv2.warpPerspective(a, P, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
    p_w = cv2.warpPerspective(prem, P, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    a_w = np.clip(a_w, 0, 1)
    fg = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
    fg[a_w <= 1e-4] = 0

    # 羽化
    fe = max(3, int(min(tw, th) * 0.05) | 1)
    keep = cv2.GaussianBlur(a_w, (fe, fe), 0)
    if protected is not None:
        keep = keep * (1.0 - protected.astype(np.float32))
    if allowed is not None:
        keep = keep * allowed.astype(np.float32)
    keep = np.clip(keep, 0, 1)

    fg_lin = to_linear(fg)
    out_lin = base_lin * (1 - keep[..., None]) + fg_lin * keep[..., None]
    res = to_srgb(out_lin)
    untouched = keep <= 1e-4
    res[untouched] = to_srgb(base_lin)[untouched]
    return res


# ---------- 4) 保护区 ----------

def build_protected(shape, boxes, exclude_ids=None, margin=8):
    """从检测框生成保护区（不可改动区域）。"""
    h, w = shape[:2]
    p = np.zeros((h, w), np.uint8)
    for i, bb in enumerate(boxes):
        if exclude_ids and i in exclude_ids:
            continue
        x1 = max(0, int(bb[0]) - margin)
        y1 = max(0, int(bb[1]) - margin)
        x2 = min(w, int(bb[2]) + margin)
        y2 = min(h, int(bb[3]) + margin)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(p, (x1, y1), (x2, y2), 255, -1)
    return p


def build_allowed(shape, quad=None, margin=0):
    """允许区：默认全图，或限制在物体投影区域。"""
    h, w = shape[:2]
    a = np.full((h, w), 255, np.uint8)
    return a


# ---------- 5) 放置门控 ----------

def validate_spawn(spawn_mask, quad, w0, h0, min_area=60):
    """检查落地点是否合法（在空位内且不越界）。"""
    if spawn_mask is None:
        return True
    cx = int(np.clip((quad[2][0] + quad[3][0]) / 2.0, 0, w0 - 1))
    cy = int(np.clip((quad[2][1] + quad[3][1]) / 2.0, 0, h0 - 1))
    if spawn_mask[cy, cx] == 0:
        return False
    xs = quad[:, 0]
    ys = quad[:, 1]
    if (xs.max() - xs.min()) < min_area * 0.3:
        return False
    if (ys.max() - ys.min()) < min_area * 0.3:
        return False
    return True


def bbox_overlap_ratio(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / aa


def filter_boxes_for_spawn(drivable, existing_boxes, margin=18,
                           check=(40, 40)):
    """在可行区域中扣除已有目标，得到空位掩码。"""
    h, w = drivable.shape[:2]
    v = drivable.copy()
    for bb in existing_boxes:
        x1 = max(0, int(bb[0]) - margin)
        y1 = max(0, int(bb[1]) - margin)
        x2 = min(w, int(bb[2]) + margin)
        y2 = min(h, int(bb[3]) + margin)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(v, (x1, y1), (x2, y2), 0, -1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, check)
    return cv2.erode(v, k)


# ---------- 6) 边界环校验 ----------

def ring_error_check(target, donor_warped, ring_mask):
    """边界环一致性。返回 (error, accepted)。"""
    if ring_mask.sum() < 100:
        return 999.0, False
    diff = np.abs(target[ring_mask].astype(np.float32)
                  - donor_warped[ring_mask].astype(np.float32))
    err = float(np.median(diff))
    return err, err <= 20.0
