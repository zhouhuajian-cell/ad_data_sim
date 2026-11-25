#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_grow.py — 让 grow 按「框内标牌实际占比」动态调整，避免越界压到相邻标牌

问题定位过程：
  200 张产出里有 52 帧（26%）框内出现多个疑似标牌，可视化后确认
  YOLO 检测框本身是准的，但 grow 固定 1.12 把贴图放大到越过框边界，
  盖住了方底板上下相邻的标牌。

  量了两种「标牌占框比例」：
    Otsu 主连通域面积 / 框面积  → 中位 0.41（失准，圆形牌内部被 抠掉一半）
    红圈直径 / 框边长           → 中位 0.94、75% 分位 1.00（可靠）

  红圈中位 0.94 说明 YOLO 框基本就是圈着红圈的，
  任何大于 1.06 的放大都会出界。

做法：
  优先用红圈比值；红圈测不到（三角牌、方牌）时退回 Otsu fill。
"""
import sys

import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")


def ring_ratio(image, bbox):
    """红圈最大跨度 / 框边长。测不到返回 None。"""
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, (0, 55, 45), (14, 255, 255))
         | cv2.inRange(hsv, (158, 55, 45), (180, 255, 255)))
    if m.sum() < 30:
        return None
    ys, xs = np.where(m > 0)
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return span / float(max(x2 - x1, y2 - y1))


def otsu_fill(image, bbox):
    """Otsu 主连通域 / 框面积。三角/方牌没有红圈时用它。"""
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = image[y1:y2, x1:x2]
    if roi.shape[0] < 16 or roi.shape[1] < 16:
        return None
    h, w = roi.shape[:2]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 10) | 1,
                                                      max(3, h // 10) | 1))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return None
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return float(stats[best, cv2.CC_STAT_AREA]) / float(w * h)


def grow_for_ring(ratio):
    """红圈比值 → grow。

    实测红圈/框中位 0.94、75 分位 1.00。
    比值 1.00 时框刚好贴着红圈，再多放就出界；
    比值 0.85 说明框比红圈大 15%，可以放大 1.08 补齐。
    """
    if ratio is None:
        return 1.06
    if ratio >= 0.97:
        return 1.00
    if ratio >= 0.92:
        return 1.03
    if ratio >= 0.85:
        return 1.08
    return 1.12


def grow_for_frame(image, bbox, fallback=1.06):
    """对外接口：给这一帧这个框算 grow。"""
    r = ring_ratio(image, bbox)
    if r is not None:
        return grow_for_ring(r)
    f = otsu_fill(image, bbox)
    if f is None:
        return fallback
    # 无红圈（三角/方牌）：方形填充率接近 1，圆形接近 0.78
    if f > 0.85:
        return 1.00
    if f > 0.65:
        return 1.05
    return 1.10


if __name__ == "__main__":
    print("ring_ratio 映射：")
    for r in (1.00, 0.95, 0.88, 0.80):
        print("  ratio=%.2f → grow=%.2f" % (r, grow_for_ring(r)))
