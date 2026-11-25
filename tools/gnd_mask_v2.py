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

    # VEH-SLIM: 车辆 mask 用 obj_mask 像素级掩码，外扩严格收敛
    #   原外扩 11px 会把车旁路面吞掉（虚线紧贴前车左侧时被误扣）
    #   新规则：远景 1px / 近中景 2px（按 road_mask 纵坐标分区）
    veh = np.zeros((h, w), np.uint8)
    p = "%s/obj_mask_ori_3.1.1/%s.png" % (clip_dir, stem)
    if os.path.isfile(p):
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if m is not None:
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            veh_raw = ((m > 0).astype(np.uint8)) * 255
            # 分区膨胀
            _ys, _ = np.where(free > 127)
            if len(_ys) > 100:
                _y0, _y1 = int(_ys.min()), int(_ys.max())
                _ysplit = int(_y0 + (_y1 - _y0) * 0.55)
            else:
                _ysplit = int(h * 0.7)
            _k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            _k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            _near = cv2.dilate(veh_raw, _k2)     # 近中景 2px
            _far = cv2.dilate(veh_raw, _k1)      # 远景 1px
            veh = _near.copy()
            if _ysplit > 0:
                veh[:_ysplit, :] = _far[:_ysplit, :]

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

def extract_clean_markings_mask_v3(img_bgr, drivable_mask, vehicle_mask):
    # 1. 纯路面区域（向下延伸防底部截断，扣除车辆）
    expanded_road = cv2.dilate(drivable_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    pure_road = cv2.bitwise_and(expanded_road, cv2.bitwise_not(vehicle_mask))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # 2. 用形态学 Top-Hat 替代 ratio_map（Top-Hat 对阶梯状树影边界不敏感，只对"细条/文字"敏感）
    th_31 = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    th_55 = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55)))

    # 阳光区阈值 18，树影暗区(L < 100) 自动降低阈值到 10，既能抓暗处白线，又不误伤树影边界
    is_shadow = L < 100
    bright_mask = np.where(is_shadow, (th_31 > 10) | (th_55 > 10), (th_31 > 18) | (th_55 > 18)).astype(np.uint8) * 255

    # 3. 黄色标线补充
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([35, 255, 255]))

    raw = cv2.bitwise_and(cv2.bitwise_or(bright_mask, yellow_mask), pure_road)

    # 4. 连通域过滤：剔除极小噪点(< 20px) 和 误检的大片反光/树影块(> 路面面积的 8%)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    road_area = np.count_nonzero(pure_road) + 1
    filtered = np.zeros_like(raw)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 20 < area < road_area * 0.08:
            filtered[labels == i] = 255

    # 5. 适度膨胀包住旧油漆边缘，严格避开车辆
    final_mask = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(vehicle_mask))
    return final_mask

def text_from_unet_and_contour(img_bgr, drivable_mask, vehicle_mask,
                                use_unet=True):
    """方案 C：U-Net 预测 + 轮廓填充 —— 专门补地面文字/箭头。

    流程：
      ① U-Net 预测车道线（4 类）
      ② 把所有细线连通域合并成"路网骨架"
      ③ 对骨架外的孤立亮块（文字/箭头），用轮廓填充补齐
      ④ 严格限制在 free_road 且排除车辆
    """
    h, w = img_bgr.shape[:2]

    # 纯路面
    expanded_road = cv2.dilate(
        drivable_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    pure_road = cv2.bitwise_and(expanded_road,
                                cv2.bitwise_not(vehicle_mask))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # ① U-Net 预测（可选）
    unet_mask = np.zeros((h, w), np.uint8)
    if use_unet:
        try:
            import torch
            _ck = None
            try:
                from gnd_ab_mask import predict_marking
                pred = predict_marking(img_bgr)
                for k in (1, 2, 3):
                    unet_mask[pred == k] = 255
            except Exception:
                pass
        except Exception:
            pass

    # ② Top-Hat 抓所有亮点（v3 逻辑）
    th_31 = cv2.morphologyEx(
        L, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    th_55 = cv2.morphologyEx(
        L, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55)))
    is_shadow = L < 100
    bright = np.where(is_shadow,
                      (th_31 > 10) | (th_55 > 10),
                      (th_31 > 18) | (th_55 > 18)).astype(np.uint8) * 255

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 60, 80]),
                         np.array([35, 255, 255]))

    raw = cv2.bitwise_or(bright, yellow)
    raw = cv2.bitwise_or(raw, unet_mask)
    raw = cv2.bitwise_and(raw, pure_road)

    # ③ 连通域分析
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    road_area = np.count_nonzero(pure_road) + 1
    keep = np.zeros((h, w), np.uint8)

    # 先按 v3 规则保留细长线和中小块
    for i in range(1, n_lab):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        ratio = max(bw, bh) / float(max(min(bw, bh), 1))
        fill = area / float(max(bw * bh, 1))
        # 细长线 或 文字状（填充率低）→ 保留
        if ratio >= 1.6 or fill <= 0.55:
            keep[labels == i] = 255
        elif area <= road_area * 0.08:
            keep[labels == i] = 255

    # ④ C-CONSTRAINED: 文字补全（严格约束）
    #    · 只在"已保留块"内部闭合，不生成新区域
    #    · 闭运算核小（5x5），只连接笔画间隙
    #    · 每个块面积上限 1.5% 路面
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k_close,
                              iterations=1)

    # 轮廓填充，但只填原始 keep 已有的连通域范围内
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((h, w), np.uint8)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 150:
            continue
        if a > road_area * 0.015:      # 严格上限 1.5%
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        fill_ratio = a / float(max(ww * hh, 1))
        # 文字/箭头填充率 0.25~0.75
        if 0.25 <= fill_ratio <= 0.75:
            # 额外要求：框内必须有足够的原始亮点（防止填满空区）
            sub_keep = keep[y:y+hh, x:x+ww]
            if np.count_nonzero(sub_keep) / float(max(ww*hh, 1)) < 0.10:
                continue
            cv2.drawContours(filled, [c], -1, 255, -1)

    # 合并：原始细线 + 文字填充块
    final = cv2.bitwise_or(keep, filled)
    final = cv2.dilate(final, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)))
    final = cv2.bitwise_and(final, cv2.bitwise_not(vehicle_mask))
    final = cv2.bitwise_and(final, pure_road)
    return final
