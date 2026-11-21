#!/usr/bin/env python3
"""GND 全量标线掩码（含文字/箭头）——方案 B。

三重约束，严格控制误擦：
  ① 空间：只能在 freespace 内，且排除 obj_mask 的实例
  ② 亮度：绝对值高 + 局部对比度高（Top-Hat）
  ③ 形态：细长 or 文字状（面积/密度/长宽比）
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")

_PALETTE = {(0, 255, 0): 1, (232, 35, 244): 2, (153, 153, 190): 3}
_MODEL = None


# ---------- 数据加载 ----------

def load_unet(weight_path="/data/train/gnd_runs/unet/best.pth"):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    from train_gnd_seg import UNet
    ck = torch.load(weight_path, map_location="cuda", weights_only=False)
    m = UNet(3, ck.get("ncls", 4), base=32)
    m.load_state_dict(ck["model"])
    m.eval()
    if torch.cuda.is_available():
        m = m.cuda()
    _MODEL = m
    return m


def predict_marking(img_bgr, size=(512, 288)):
    import torch
    m = load_unet()
    h, w = img_bgr.shape[:2]
    im = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im = (im - 0.45) / 0.25
    x = torch.from_numpy(im.transpose(2, 0, 1).copy()).float()
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        pred = m(x.unsqueeze(0)).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def extend_freespace_bottom(free, veh, img_shape, slope=3.0):
    """FREE-EXTEND-BOTTOM: freespace 底部截断处按梯形向下延伸。

    freespace_mask 在画面底部（近景路面）常被截断，
    按左右边界的斜率线性外推到画面底边。
    """
    h, w = img_shape[:2]
    out = free.copy()

    rows = np.where(np.any(free > 0, axis=1))[0]
    if len(rows) == 0:
        return out
    y_bot = int(rows.max())
    if y_bot >= h - 3:
        return out

    # 取底边往上 30 行，估计左右边界
    y_ref = max(0, y_bot - 30)
    band = free[y_ref:y_bot + 1]
    lefts, rights = [], []
    for yy in range(band.shape[0]):
        cols = np.where(band[yy] > 0)[0]
        if len(cols) > 10:
            lefts.append(cols.min())
            rights.append(cols.max())
    if not lefts:
        return out

    x_l = int(np.median(lefts))
    x_r = int(np.median(rights))

    # 线性外推：每往下 1 行，左右各扩展 slope 像素
    for y in range(y_bot + 1, h):
        dy = y - y_bot
        l = max(0, x_l - int(slope * dy))
        r = min(w - 1, x_r + int(slope * dy))
        if r > l:
            out[y, l:r + 1] = 255

    # 排除车辆
    out = cv2.bitwise_and(out, cv2.bitwise_not(veh))
    return out


def load_masks(clip_dir, stem, shape):
    """加载 freespace / obj_mask / seg_lane。"""
    h, w = shape[:2]
    out = {}

    p = os.path.join(clip_dir, "freespace_mask_ori_3.0.0", stem + ".png")
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED) if os.path.isfile(p) else None
    if m is not None:
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out["free"] = ((m > 0).astype(np.uint8)) * 255
    else:
        m = np.zeros((h, w), np.uint8)
        m[int(h * 0.52):, :] = 255
        out["free"] = m

    p = os.path.join(clip_dir, "obj_mask_ori_3.1.1", stem + ".png")
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED) if os.path.isfile(p) else None
    if m is not None:
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        vm = ((m > 0).astype(np.uint8)) * 255
        vm = cv2.dilate(vm, np.ones((11, 11), np.uint8))
        out["veh"] = vm
    else:
        out["veh"] = np.zeros((h, w), np.uint8)

    p = os.path.join(clip_dir, "seg_lane", stem + ".png")
    seg = cv2.imread(p) if os.path.isfile(p) else None
    if seg is not None:
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        gt = np.zeros((h, w), np.uint8)
        for bgr, k in _PALETTE.items():
            mm = np.all(np.abs(seg.astype(int) - np.array(bgr)) < 30, axis=2)
            gt[mm] = k
        out["gt"] = gt
    else:
        out["gt"] = None

    return out


# ---------- 方案 B：全量标线（含文字/箭头） ----------

def extract_shadow_robust(img_bgr, drivable_mask, vehicle_mask,
                          ratio_thr=1.12, top_small=31, top_large=65,
                          top_thr=18, dilate_px=13):
    """EXACT-USER-V2: 相对亮度比值 + 多尺度 Top-Hat（按用户方案原样）。"""
    # 1. 适当向下延伸路面 Mask
    road_expand_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    expanded_road = cv2.dilate(drivable_mask, road_expand_kernel,
                               iterations=1)
    pure_road = cv2.bitwise_and(expanded_road,
                                cv2.bitwise_not(vehicle_mask))

    # 2. 转 LAB 取 L 通道
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # 3. 局部相对比值（专治树影里的暗白线）
    L_bg_local = cv2.GaussianBlur(L, (51, 51), 0) + 1e-5
    ratio_map = L / L_bg_local
    shadow_robust_mask = (ratio_map > ratio_thr).astype(np.uint8) * 255

    # 4. 多尺度 Top-Hat
    L_u8 = L.astype(np.uint8)
    th_small = cv2.morphologyEx(
        L_u8, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (top_small, top_small)))
    th_large = cv2.morphologyEx(
        L_u8, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (top_large, top_large)))
    tophat_mask = ((th_small > top_thr) |
                   (th_large > top_thr)).astype(np.uint8) * 255

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
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (dilate_px, dilate_px)),
        iterations=1)

    # 严格避开车辆
    final_erase_mask = cv2.bitwise_and(final_erase_mask,
                                       cv2.bitwise_not(vehicle_mask))
    return final_erase_mask, {
        "ratio_map": ratio_map,
        "shadow_robust": shadow_robust_mask,
        "tophat": tophat_mask,
        "yellow": yellow_mask,
    }


def build_full_marking_mask(img_bgr, clip_dir, stem,
                            use_unet=True, use_gt=True,
                            paint_thresh=0.55,
                            local_thresh=12,
                            min_area=20,
                            max_area_ratio=0.02):
    """三重约束提取所有地面标线（线、文字、箭头）。"""
    h, w = img_bgr.shape[:2]
    masks = load_masks(clip_dir, stem, img_bgr.shape)
    free = masks["free"]
    veh = masks["veh"]
    gt = masks["gt"]

    # FREE-EXTEND-BOTTOM: 补全底部截断
    free = extend_freespace_bottom(free, veh, img_bgr.shape, slope=3.0)
    road = cv2.bitwise_and(free, cv2.bitwise_not(veh))
    if not np.any(road):
        return {"mask": np.zeros((h, w), np.uint8),
                "debug": {}, "stats": {"erase_px": 0, "erase_pct": 0}}

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    road_L = L[road > 0]
    L_med = float(np.median(road_L))
    L_p75 = float(np.percentile(road_L, 75))
    L_p90 = float(np.percentile(road_L, 90))

    # ① 绝对亮度：油漆必须比路面中位亮很多
    thr_abs = max(L_p75 + 18, L_med + 30, 120)
    m_abs = ((L > thr_abs) & (road > 0)).astype(np.uint8) * 255

    # ② 局部对比度：Top-Hat（抗树影）
    k_top = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    tophat = cv2.morphologyEx(L.astype(np.uint8), cv2.MORPH_TOPHAT, k_top)
    m_top = (tophat > local_thresh).astype(np.uint8) * 255

    # ③ 语义（U-Net + 真值）
    m_sem = np.zeros((h, w), np.uint8)
    if use_unet:
        pred = predict_marking(img_bgr)
        for k in (1, 2, 3):
            m_sem[pred == k] = 255
    if use_gt and gt is not None:
        for k in (1, 2, 3):
            m_sem[gt == k] = 255

    # MERGE-SHADOW-ROBUST: 语义 ∪ shadow_robust（用你的完整版）
    m_sr, _sr_dbg = extract_shadow_robust(
        img_bgr, free, veh, dilate_px=1)
    m = cv2.bitwise_or(m_sem, m_sr)
    m = cv2.bitwise_and(m, road)

    # ④ 形态筛选
    n, lab_img, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros((h, w), np.uint8)
    max_area = int(h * w * max_area_ratio)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < min_area:
            continue
        if area > max_area:
            continue
        # 细长 or 文字状（填充率适中）
        ratio = max(bw, bh) / float(max(min(bw, bh), 1))
        fill = area / float(max(bw * bh, 1))
        long_thin = ratio >= 1.8
        text_like = (fill >= paint_thresh and 30 <= area <= 8000 and bh >= 6)
        if long_thin or text_like:
            keep[lab_img == i] = 255

    # ⑤ 膨胀包边（SHADOW-ROBUST: 加大到 13x13）
    keep = cv2.dilate(keep, np.ones((13, 13), np.uint8))
    keep = cv2.bitwise_and(keep, cv2.bitwise_not(veh))
    keep = cv2.bitwise_and(keep, free)

    return {
        "mask": keep,
        "debug": {
            "m_abs": m_abs, "m_top": m_top, "m_sem": m_sem,
            "road": road, "free": free, "veh": veh, "gt": gt,
            "thr_abs": thr_abs, "L_med": L_med, "L_p90": L_p90,
        },
        "stats": {
            "erase_px": int(np.count_nonzero(keep)),
            "erase_pct": round(
                np.count_nonzero(keep) / float(h * w) * 100, 2),
            "thr_abs": round(thr_abs, 1),
            "road_pct": round(
                np.count_nonzero(road) / float(h * w) * 100, 2),
        },
    }
