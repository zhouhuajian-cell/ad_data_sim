#!/usr/bin/env python3
"""GND seg_lane 生成：把标线像素分成 edge / dash 两类。

判别依据（已用真实数据验证）：
  - edge = 位于可行驶区域【左右边界】的标线（贴墙 / 贴路沿）
  - dash = 位于可行驶区域【内部】的标线（车道分隔线）

真实数据参考（1790569917.088）：
  edge: y 601~823, x 233~1488（左右两侧墙边）
  dash: y 576~760, x 245~895（车道内部分隔）
"""
import cv2
import numpy as np

# 真实文件的 RGB 值（PIL 验证）
PALETTE_RGB = {
    "bg":   (0, 0, 0),
    "edge": (0, 255, 0),
    "dash": (244, 35, 232),
}
PALETTE_BGR = {k: (v[2], v[1], v[0]) for k, v in PALETTE_RGB.items()}


def split_edge_dash(marking_mask, free_mask, img_shape,
                    boundary_band_px=28, min_component_px=8):
    """把标线 mask 拆成 edge / dash 两类。

    参数:
        marking_mask: 标线二值图 (uint8 0/255)，所有标线
        free_mask:    可行驶区域 (uint8 0/255)
        img_shape:    (h, w)
        boundary_band_px: 距可行驶区边界的带宽（像素）
    """
    h, w = img_shape[:2]
    mark = (marking_mask > 127)
    if mark.sum() == 0:
        zero = np.zeros((h, w), np.uint8)
        return zero, zero

    # 1) 可行驶区域边界
    # FREE-VAL-117: 真实数据 freespace_mask 的可行驶值 = 117，不可行驶 = 0
    #   （0: 72.2% 天空/墙/远处，117: 27.8% 路面 y561~1079）
    if free_mask is None:
        fb = np.ones((h, w), np.uint8)
    else:
        free = free_mask.copy()
        if free.shape[:2] != (h, w):
            free = cv2.resize(free, (w, h), interpolation=cv2.INTER_NEAREST)
        fb = (free > 0).astype(np.uint8)

    # 2) 每行的左右边界 x
    left_x = np.full(h, -1, np.int32)
    right_x = np.full(h, -1, np.int32)
    for y in range(h):
        xs = np.where(fb[y] > 0)[0]
        if xs.size:
            left_x[y] = xs.min()
            right_x[y] = xs.max()

    # 3) EDGE-OUTER-LOGIC: 按可行驶区左右边界判定
    #    真实数据规律（已验证）：
    #      edge = 2 个大连通域，贴左右墙
    #      dash = 5 个小连通域，在车道内部
    #    判定：像素到左右边界的横向距离，取两侧外廓 -> edge
    edge = np.zeros((h, w), bool)
    ys, xs = np.where(mark)
    lx = left_x[ys]
    rx = right_x[ys]
    valid = (lx >= 0) & (rx >= 0)
    d_left = np.abs(xs - lx)
    d_right = np.abs(xs - rx)
    near = valid & ((d_left <= boundary_band_px) |
                    (d_right <= boundary_band_px))
    edge[ys[near], xs[near]] = True
    dash = mark & (~edge)

    # 4) 去掉极小噪点
    def clean(m):
        u8 = m.astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(u8, 8)
        out = np.zeros_like(u8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_component_px:
                out[lab == i] = 1
        return out

    return clean(edge) * 255, clean(dash) * 255


def make_seg_lane_png(img_shape, edge_mask, dash_mask):
    """合成 seg_lane 调色板图（返回 BGR，供 cv2.imwrite）。"""
    h, w = img_shape[:2]
    out = np.zeros((h, w, 3), np.uint8)
    if dash_mask is not None:
        out[dash_mask > 127] = PALETTE_BGR["dash"]
    if edge_mask is not None:
        out[edge_mask > 127] = PALETTE_BGR["edge"]
    return out


def verify_palette(png_path):
    """校验生成的 seg_lane 颜色是否合法（按 RGB 比对）。"""
    im = cv2.imread(png_path)
    if im is None:
        return False, "读不到"
    uniq = np.unique(im.reshape(-1, 3), axis=0)
    allowed = set(PALETTE_RGB.values())
    bad = []
    stats = {}
    for c in uniq:
        rgb = (int(c[2]), int(c[1]), int(c[0]))
        if rgb not in allowed:
            bad.append(rgb)
        else:
            stats[rgb] = int((np.all(im == c, axis=2)).sum())
    if bad:
        return False, "非法颜色 %s" % bad[:3]
    return True, stats
