#!/usr/bin/env python3
"""GND 净面清零管线（三步走）。

① extract_all_old_markings_mask  —— Top-Hat + 黄色 + 语义，全量提取旧标线
② erase_to_clean_base           —— LaMa 一次性清零，得到"净面底图"
③ redraw_on_clean_base          —— 净面重绘 + 车辆遮挡扣除
"""
import os

import cv2
import numpy as np

_LAMA = None


def get_lama():
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()
    return _LAMA


# ---------- 可行驶区 & 车辆掩码 ----------

def build_drivable_mask(calib, shape, max_dist=60.0, half_width=15.0):
    """用标定投影生成可行驶区多边形。"""
    from gnd_calib import ground_to_pixel

    h, w = shape[:2]
    left, right = [], []
    for x in np.arange(3.0, max_dist, 0.5):
        pl = ground_to_pixel(calib, x, half_width)
        pr = ground_to_pixel(calib, x, -half_width)
        if pl is None or pr is None:
            continue
        if not (np.isfinite(pl[0]) and np.isfinite(pr[0])):
            continue
        left.append([pl[0], pl[1]])
        right.append([pr[0], pr[1]])
    m = np.zeros((h, w), np.uint8)
    if len(left) < 3:
        return m
    poly = np.array(left + right[::-1], np.int32)
    cv2.fillPoly(m, [poly], 255)
    return m


def build_vehicle_mask(objects, shape, margin=6):
    """从 3D 标注构造车辆/行人遮挡区。"""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    for o in objects or []:
        b = o.get("bbox2d")
        if not b or len(b) != 4:
            continue
        # 过滤异常框
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        if bw <= 4 or bh <= 4 or bw * bh > w * h * 0.35:
            continue
        x1 = max(0, int(b[0]) - margin)
        y1 = max(0, int(b[1]) - margin)
        x2 = min(w, int(b[2]) + margin)
        y2 = min(h, int(b[3]) + margin)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(m, (x1, y1), (x2, y2), 255, -1)
    return m


def build_vehicle_mask_from_seg(seg_bgr, shape):
    """从 seg_lane 或 obj_mask 构造（备选）。"""
    h, w = shape[:2]
    if seg_bgr is None:
        return np.zeros((h, w), np.uint8)
    return np.zeros((h, w), np.uint8)


# ---------- ① 全量提取旧标线 ----------

def extract_all_old_markings_mask(img_bgr, drivable_mask, vehicle_mask,
                                  tophat_kernel=25, tophat_thresh=15,
                                  dilate_px=9, use_semantic=None):
    """全量提取路面上的所有旧标线（车道线、文字、箭头等）。

    img_bgr:       原图
    drivable_mask: 255=路面
    vehicle_mask:  255=车辆/障碍物（不擦除）
    use_semantic:  语义分割类别图（可选），1/2/3=车道线
    """
    h, w = img_bgr.shape[:2]

    # 1) 纯路面 = 可行驶区 - 车辆
    pure_road = cv2.bitwise_and(drivable_mask, cv2.bitwise_not(vehicle_mask))

    sem_mask = np.zeros((h, w), np.uint8)
    if use_semantic is not None:
        for k in (1, 2, 3):
            sem_mask[use_semantic == k] = 255

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # TOPHAT-V3: 双约束（绝对亮度 + 局部对比度）
    # ① 绝对亮度：路面中位 ~94，白线通常 >170
    road_l = L[pure_road > 0] if np.any(pure_road) else L.ravel()
    L_med = float(np.median(road_l))
    L_p90 = float(np.percentile(road_l, 90))
    abs_thr = max(np.percentile(road_l, 85), L_med + 32, 135)
    bright_abs = ((L > abs_thr) & (pure_road > 0)).astype(np.uint8) * 255

    # ② 局部 Top-Hat（抗树影，但要求同时满足绝对亮度）
    k_top = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (tophat_kernel, tophat_kernel))
    tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k_top)
    _, bright_top = cv2.threshold(tophat, tophat_thresh, 255,
                                  cv2.THRESH_BINARY)

    # ③ 语义 + 亮度 + Top-Hat 三者并集（ERASE-V5-RESIDUAL）
    bright_mask = bright_abs.copy()
    if use_semantic is not None and np.any(sem_mask):
        k_sem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        sem_d = cv2.dilate(sem_mask, k_sem, iterations=1)
        bright_mask = cv2.bitwise_or(bright_mask, sem_d)
    bright_mask = cv2.bitwise_or(bright_mask, bright_top)

    # 3) 黄色标线（HSV）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 60, 80]),
                              np.array([35, 255, 255]))

    # 4) 语义分割补充（如果有）

    # 5) 合并
    raw = cv2.bitwise_or(bright_mask, yellow_mask)
    raw = cv2.bitwise_or(raw, sem_mask)
    raw = cv2.bitwise_and(raw, pure_road)

    # 6) 去噪
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k_open)

    # TOPHAT-TUNED: 连通域筛选，只保留"细长/文字"结构
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(clean, 8)
    keep = np.zeros((h, w), np.uint8)
    max_area = int(h * w * 0.02)      # 单块不超过画面 2%
    for i in range(1, n_lab):
        x, y, ww, hh, area = st[i]
        if area < 8:
            continue
        if area > max_area:
            continue      # 大块（树影/高光）丢弃
        # ERASE-V5-RESIDUAL: 放宽细长判定（停车位短线、箭头碎块）
        ratio = max(ww, hh) / max(min(ww, hh), 1)
        fill = area / float(max(ww * hh, 1))
        if ratio > 1.5 or area < 1500 or fill < 0.55:
            keep[lab == i] = 255
    clean = keep

    # 7) 膨胀包住边缘
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilate_px, dilate_px))
    final = cv2.dilate(clean, k_dil, iterations=1)

    # 8) 绝不碰车辆
    final = cv2.bitwise_and(final, cv2.bitwise_not(vehicle_mask))
    final = cv2.bitwise_and(final, drivable_mask)
    return final


# ---------- ② LaMa 清零 ----------

def erase_to_clean_base(img_bgr, erase_mask, dilate_px=8):
    """LaMa 一次性清零，得到净面底图。"""
    if not np.any(erase_mask):
        return img_bgr.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (dilate_px, dilate_px))
    m = cv2.dilate(erase_mask, k, iterations=1)

    from PIL import Image as _Img
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = get_lama()(_Img.fromarray(rgb), _Img.fromarray(m))
    clean = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    # 只替换 mask 内部
    mm = (m > 127)[..., None]
    res = np.where(mm, clean, img_bgr)
    return res.astype(np.uint8)


# ---------- ③ 净面重绘 ----------

def draw_lanes_on_clean(clean_bgr, calib, cfg, old_mask=None):
    """在净面底图上画新标线（GND-ASSET-RENDER-WIRE: 素材贴图优先）。"""
    from compose_core import blend_new_gnd_realistic

    # ① 人工指定素材模式
    items = cfg.get("gnd_items")
    if items and isinstance(items, list) and len(items) > 0:
        try:
            from gnd_asset_render import render_items
            alpha, line_rgb, _n = render_items(calib, items,
                                               clean_bgr.shape)
            if _n > 0:
                out = blend_new_gnd_realistic(
                    clean_bgr, line_rgb, alpha,
                    wear_level=float(cfg.get("wear", 0.35)),
                    dirt_level=float(cfg.get("dirt", 0.25)),
                    optical_blur=float(cfg.get("optical_blur", 1.2)))
                paint = (alpha * 255).astype(np.uint8)
                print("    素材渲染: %d 项" % _n)
                return out, alpha, paint
            print("    素材渲染失败，回退自动模式")
        except Exception as _e:
            print("    素材渲染异常，回退:", str(_e)[:60])

    # ② 默认自动模式（原逻辑）
    from gnd_fusion import render_lane_alpha
    alpha, paint = render_lane_alpha(calib, cfg, clean_bgr.shape)
    line_rgb = np.full_like(clean_bgr, 236)

    out = blend_new_gnd_realistic(
        clean_bgr, line_rgb, alpha,
        wear_level=float(cfg.get("wear", 0.35)),
        dirt_level=float(cfg.get("dirt", 0.25)),
        optical_blur=float(cfg.get("optical_blur", 1.2)))
    return out, alpha, paint


def apply_vehicle_occlusion(synth_bgr, orig_bgr, vehicle_mask):
    """车辆前景回贴：确保新标线被真实车辆遮挡。"""
    vm = vehicle_mask[..., None] > 127
    out = np.where(vm, orig_bgr, synth_bgr)
    return out.astype(np.uint8)


# ---------- 完整流程 ----------

def run_pipeline(img_bgr, calib, cfg, objects=None, semantic=None,
                 verbose=True, clip_dir=None, stem=None):
    """三步走完整流程。

    返回 dict:
      clean_base: 净面底图（只有沥青、树影、车辆，无标线）
      erase_mask: 旧标线掩码
      final:      最终合成图
      stats:      统计信息
    """
    h, w = img_bgr.shape[:2]

    # 可行驶区
    drv = build_drivable_mask(calib, img_bgr.shape)
    if not np.any(drv):
        # 兜底：地平线以下
        drv[int(h * 0.55):, :] = 255

    # 车辆掩码
    veh = build_vehicle_mask(objects, img_bgr.shape)

    # ① 全量提取（FULL-MASK-WIRED: 方案 B 三重约束）
    if verbose:
        print("    ① 提取旧标线（含文字/箭头）...")
    # WIRE-V2-FINAL: 用户方案 v2（相对亮度 + 多尺度 Top-Hat + 连通域筛选）
    from gnd_mask_v2 import build_mask as _build_v2
    _r = _build_v2(img_bgr, clip_dir, stem,
                   use_semantic_gt=(semantic is not None))
    erase_mask = _r["mask"]
    # 车辆 mask 用 v2 算出的（外扩 11px）
    veh = _r["veh"]
    if verbose:
        print("       统计: %s" % _r["stats"])
    n_erase = int(np.count_nonzero(erase_mask))
    if verbose:
        print("       擦除区: %d 像素 (%.2f%%)" %
              (n_erase, n_erase / float(h * w) * 100))

    # ② LaMa 清零
    if verbose:
        print("    ② LaMa 清零...")
    clean_base = erase_to_clean_base(img_bgr, erase_mask)

    # ③ 净面重绘
    if verbose:
        print("    ③ 净面重绘...")
    synth, alpha, paint = draw_lanes_on_clean(clean_base, calib, cfg)
    n_paint = int(np.count_nonzero(paint))
    if verbose:
        print("       新标线: %d 像素 (%.2f%%)" %
              (n_paint, n_paint / float(h * w) * 100))

    # 车辆遮挡
    final = apply_vehicle_occlusion(synth, img_bgr, veh)

    return {
        "clean_base": clean_base,
        "erase_mask": erase_mask,
        "drivable": drv,
        "vehicle": veh,
        "final": final,
        "stats": {
            "erase_px": n_erase,
            "erase_pct": round(n_erase / float(h * w) * 100, 2),
            "paint_px": n_paint,
            "paint_pct": round(n_paint / float(h * w) * 100, 2),
        },
    }
