#!/usr/bin/env python3
"""GND 旧标线提取 —— 用户方案 v2 原样实现。

核心：
  ① 局部相对亮度比值 (L / L_bg > 1.12) 抓树影里的暗白线
  ② 多尺度 Top-Hat (31 + 65) 兼顾远景细线 + 近景宽线
  ③ 路面 Mask 向下延伸 25px，防近景被切
"""
import cv2
import numpy as np


def extract_all_old_markings_mask_v2(img_bgr, drivable_mask, vehicle_mask):
    """按用户方案原样实现。"""
    # 1. 适当向下延伸路面 Mask
    road_expand_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    expanded_road = cv2.dilate(drivable_mask, road_expand_kernel,
                               iterations=1)
    pure_road = cv2.bitwise_and(expanded_road, cv2.bitwise_not(vehicle_mask))

    # 2. 转 LAB 取 L 通道
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # 3. 局部相对亮度比值（专治树影里的暗白线）
    L_bg_local = cv2.GaussianBlur(L, (51, 51), 0) + 1e-5
    ratio_map = L / L_bg_local
    shadow_robust_mask = (ratio_map > 1.12).astype(np.uint8) * 255

    # 4. 多尺度 Top-Hat（TOPHAT-CONSTRAINED: 提高阈值 + 连通域筛选）
    L_u8 = L.astype(np.uint8)
    th_small = cv2.morphologyEx(
        L_u8, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    th_large = cv2.morphologyEx(
        L_u8, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65)))

    # 阈值提高到 38（原 18 会吞掉树影/纹理）
    th_raw = ((th_small > 38) | (th_large > 38)).astype(np.uint8) * 255

    # 连通域筛选：只保留细长或文字状小块
    hh, ww = img_bgr.shape[:2]
    n_lab, lab_i, st_i, _ = cv2.connectedComponentsWithStats(th_raw, 8)
    keep = np.zeros((hh, ww), np.uint8)
    max_area = int(hh * ww * 0.015)
    for i in range(1, n_lab):
        x, y, bw, bh, area = st_i[i]
        if area < 25 or area > max_area:
            continue
        ratio_wh = max(bw, bh) / float(max(min(bw, bh), 1))
        fill = area / float(max(bw * bh, 1))
        if ratio_wh >= 1.6 or (fill >= 0.45 and area <= 6000):
            keep[lab_i == i] = 255
    tophat_mask = keep

    # 5. 补充黄色标线
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 60, 80]),
                              np.array([35, 255, 255]))

    # 6. 取并集并严格限制在纯路面内
    raw_markings = cv2.bitwise_or(shadow_robust_mask, tophat_mask)
    raw_markings = cv2.bitwise_or(raw_markings, yellow_mask)
    raw_markings = cv2.bitwise_and(raw_markings, pure_road)

    # 7. 去噪 + 膨胀包边
    clean_markings = cv2.morphologyEx(
        raw_markings, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    final_erase_mask = cv2.dilate(
        clean_markings,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1)

    # 严格避开车辆
    final_erase_mask = cv2.bitwise_and(final_erase_mask,
                                       cv2.bitwise_not(vehicle_mask))
    return final_erase_mask


def compute_masks(img_bgr, clip_dir, stem, use_semantic_gt=True):
    """组装 freespace / vehicle / semantic，并调用 v2。"""
    h, w = img_bgr.shape[:2]

    # freespace（含底部延伸）
    free = None
    p = "%s/freespace_mask_ori_3.0.0/%s.png" % (clip_dir, stem)
    import os
    if os.path.isfile(p):
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is not None:
            if m.ndim == 3:
                m = m[..., 0]
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            free = ((m > 0).astype(np.uint8)) * 255
    if free is None:
        free = np.zeros((h, w), np.uint8)
        free[int(h * 0.52):, :] = 255

    # 底部几何延伸（freespace 常被截断）
    rows = np.where(np.any(free > 0, axis=1))[0]
    if len(rows) and rows.max() < h - 3:
        y_bot = int(rows.max())
        band = free[max(0, y_bot - 30):y_bot + 1]
        lefts, rights = [], []
        for yy in range(band.shape[0]):
            cols = np.where(band[yy] > 0)[0]
            if len(cols) > 10:
                lefts.append(cols.min())
                rights.append(cols.max())
        if lefts:
            xl = int(np.median(lefts))
            xr = int(np.median(rights))
            for y in range(y_bot + 1, h):
                dy = y - y_bot
                l = max(0, xl - int(3.0 * dy))
                r = min(w - 1, xr + int(3.0 * dy))
                if r > l:
                    free[y, l:r + 1] = 255

    # 车辆 mask（obj_mask，外扩 11px ≈ 5px 半径）
    veh = np.zeros((h, w), np.uint8)
    p = "%s/obj_mask_ori_3.1.1/%s.png" % (clip_dir, stem)
    if os.path.isfile(p):
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is not None:
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            veh = ((m > 0).astype(np.uint8)) * 255
            veh = cv2.dilate(veh, np.ones((11, 11), np.uint8))

    # 语义真值（可选叠加）
    sem_extra = None
    if use_semantic_gt:
        p = "%s/seg_lane/%s.png" % (clip_dir, stem)
        if os.path.isfile(p):
            seg = cv2.imread(p)
            if seg is not None:
                if seg.shape[:2] != (h, w):
                    seg = cv2.resize(seg, (w, h),
                                     interpolation=cv2.INTER_NEAREST)
                sem_extra = np.zeros((h, w), np.uint8)
                for bgr in ((0, 255, 0), (232, 35, 244), (153, 153, 190)):
                    mm = np.all(np.abs(seg.astype(int) -
                                       np.array(bgr)) < 30, axis=2)
                    sem_extra[mm] = 255

    return free, veh, sem_extra


def build_mask(img_bgr, clip_dir, stem, use_semantic_gt=True):
    """完整入口：v2 + 语义真值并集。"""
    free, veh, sem = compute_masks(img_bgr, clip_dir, stem,
                                   use_semantic_gt)
    m = extract_all_old_markings_mask_v2(img_bgr, free, veh)
    if sem is not None and np.any(sem):
        sem_d = cv2.dilate(sem, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (13, 13)))
        m = cv2.bitwise_or(m, sem_d)
        m = cv2.bitwise_and(m, cv2.bitwise_not(veh))
        m = cv2.bitwise_and(m, free)
    h, w = img_bgr.shape[:2]
    return {
        "mask": m,
        "free": free, "veh": veh, "sem": sem,
        "stats": {
            "erase_px": int(np.count_nonzero(m)),
            "erase_pct": round(np.count_nonzero(m) / float(h * w) * 100, 2),
            "free_pct": round(np.count_nonzero(free) / float(h * w) * 100, 2),
            "veh_pct": round(np.count_nonzero(veh) / float(h * w) * 100, 2),
        },
    }
