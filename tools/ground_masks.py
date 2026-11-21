"""路面 / 前景 / 保护区掩码。

GND 标线必须只画在可见路面上，不能覆盖真实车辆、车头、路缘。

做法（纯 CV，无需 UE 深度 pass）：
  1) 路面掩码：HSV 低饱和 + 中低亮度 + 位于画面下半 + 单应诱导的车道带内
  2) 前景掩码：底图上明显比路面亮/饱和的连通块（车、人、路缘石、路牌）
  3) 保护区：画面底部车头区域 + 路缘上方一定高度
"""
import cv2
import numpy as np


def road_mask(bg_bgr, ground_band=None, band_frac=2.2):
    """估计可见路面掩码（0/1 浮点）。

    ground_band: (y0, y1, x_polyline) 可选，来自单应诱导的车道带
    """
    h, w = bg_bgr.shape[:2]
    hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)

    # 路面：低饱和、非极亮
    sat_ok = s < 60
    val_ok = (v > 35) & (v < 200)

    m = (sat_ok & val_ok).astype(np.uint8)

    # 只信画面下半部分（地平线以下）
    yy = np.arange(h)[:, None].repeat(w, axis=1)
    horizon = int(h * 0.52)
    m[:horizon, :] = 0

    # 形态学：连通成片，滤掉细碎纹理
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    # 取最大连通块（主路面）
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = 1 + int(np.argmax(areas))
        m = (lab == big).astype(np.uint8)

    # 与单应车道带取交
    if ground_band is not None:
        band = ground_band.astype(np.uint8)
        m = (m & band).astype(np.uint8)

    # 软边
    soft = cv2.GaussianBlur(m.astype(np.float32), (21, 21), 0)
    return np.clip(soft, 0, 1)


def foreground_mask(bg_bgr, road):
    """车辆/行人/路缘等前景保护掩码。"""
    h, w = bg_bgr.shape[:2]
    hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)

    # 明显不是路面的：高饱和 或 很亮 或 很暗的实心区域
    nonroad = ((s > 80) | (v > 210) | (v < 40)).astype(np.uint8)

    # 紧邻路面之上/之中的高对比块 = 真实物体
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    nonroad = cv2.morphologyEx(nonroad, cv2.MORPH_CLOSE, k, iterations=2)
    nonroad = cv2.morphologyEx(nonroad, cv2.MORPH_OPEN, k, iterations=1)

    # 在路面区域内出现的高对比 = 车/人（非路面覆盖物）
    inside_road = (nonroad & (road > 0.5)).astype(np.uint8)

    # 膨胀一点，避免边缘渗色
    inside_road = cv2.dilate(
        inside_road,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 1)

    soft = cv2.GaussianBlur(inside_road.astype(np.float32), (15, 15), 0)
    return np.clip(soft, 0, 1)


def hood_mask(bg_bgr, frac=0.16):
    """画面底部车头/引擎盖保护区。"""
    h, w = bg_bgr.shape[:2]
    m = np.zeros((h, w), np.float32)
    y0 = int(h * (1.0 - frac))
    m[y0:, :] = 1.0
    # 软过渡
    m = cv2.GaussianBlur(m, (31, 31), 0)
    return np.clip(m, 0, 1)


def compose_ground(bg_bgr, paint_bgr, coverage, road, fg, hood, wear):
    """只在 可见路面 ∧ 非前景 ∧ 非车头 处上漆。

    alpha = coverage * road * (1-fg) * (1-hood) * wear
    """
    alpha = (np.clip(coverage, 0, 1) * np.clip(road, 0, 1)
             * (1.0 - np.clip(fg, 0, 1)) * (1.0 - np.clip(hood, 0, 1))
             * np.clip(wear, 0, 1))

    out = (bg_bgr.astype(np.float32) * (1 - alpha[..., None])
           + paint_bgr.astype(np.float32) * alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), alpha


def lane_band_mask(shape, H, y_left, y_right, x_near, x_far, margin=1.6):
    """用单应把车道带（世界米制）投到像素，得到带掩码。"""
    h, w = shape
    quad = np.array([
        [y_left - margin, x_near], [y_right + margin, x_near],
        [y_right + margin, x_far], [y_left - margin, x_far],
    ], np.float32)
    pts = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return None
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, np.rint(pts).astype(np.int32), 255)
    return m
