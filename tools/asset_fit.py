#!/usr/bin/env python3
"""素材适配门控：用遮盖关系代替形状猜测。

核心思想：不判断原牌"叫什么形状"，而是判断候选素材能否严丝合缝盖住原牌前景。
三个硬指标：
  1. 前景覆盖率 recall = |原牌前景 ∩ 素材| / |原牌前景|        >= 0.94
  2. 最大外露距离 exposed_px（露角像素到素材边界的最大距离）   <= 2px
  3. IoU = |∩| / |∪|                                          >= 0.85
再叠加底色一致性，避免红圈盖蓝底。
"""
import os

import cv2
import numpy as np

# 素材前景掩码缓存：path -> (mtime, mask_tight, color_family)
_ASSET_CACHE = {}


def _largest_cc_fill(mask):
    """取最大连通域并填内孔。"""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m * 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
        m = filled
    return m


def _tight(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, None
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return mask[y1:y2, x1:x2], (x1, y1, x2, y2)


def color_family(bgr, mask=None):
    """按 HSV 判定底色族：white / black / red / yellow / blue / green / other。"""
    if bgr is None or bgr.size == 0:
        return 'other'
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if mask is not None:
        sel = mask > 0
        if sel.sum() < 10:
            return 'other'
        h, s, v = h[sel], s[sel], v[sel]
    s = s.astype(np.float32)
    v = v.astype(np.float32)
    h = h.astype(np.float32)
    col = (s >= 70) & (v >= 45)
    if col.mean() < 0.12:
        return 'white' if float(np.median(v)) > 110 else 'black'
    hh = h[col]
    med = float(np.median(hh))
    if med < 12 or med >= 168:
        return 'red'
    if 12 <= med < 38:
        return 'yellow'
    if 38 <= med < 95:
        return 'green'
    if 95 <= med < 140:
        return 'blue'
    return 'other'


def asset_fg(path):
    """素材前景掩码（紧外接框），带缓存。"""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None, None, 'other'
    hit = _ASSET_CACHE.get(path)
    if hit and hit[0] == mt:
        return hit[1], hit[2], hit[3]
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        _ASSET_CACHE[path] = (mt, None, None, 'other')
        return None, None, 'other'
    if im.ndim == 3 and im.shape[2] == 4:
        mask = (im[:, :, 3] > 32).astype(np.uint8)
        bgr = im[:, :, :3]
    else:
        bgr = im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]
        corners = np.array([bgr[0, 0], bgr[0, w - 1],
                            bgr[h - 1, 0], bgr[h - 1, w - 1]], np.float32)
        bg = corners.mean(axis=0)
        mask = (np.linalg.norm(bgr.astype(np.float32) - bg, axis=2) > 40).astype(np.uint8)
    mask = _largest_cc_fill(mask)
    tight, box = _tight(mask)
    if tight is None:
        _ASSET_CACHE[path] = (mt, None, None, 'other')
        return None, None, 'other'
    x1, y1, x2, y2 = box
    ring = _ring_family(bgr[y1:y2, x1:x2], tight)
    _ASSET_CACHE[path] = (mt, tight, box, ring)
    return tight, box, ring


def _ring_family(bgr, mask, band=0.28):
    """取形状外环的颜色族（限速牌的红圈、指示牌的蓝底）。"""
    if bgr is None or bgr.size == 0 or mask is None:
        return 'other'
    m = (mask > 0).astype(np.uint8)
    er = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    if er.max() <= 0:
        return 'other'
    thr = max(1.0, er.max() * band)
    ring = (m > 0) & (er <= thr)
    return color_family(bgr, ring.astype(np.uint8))


def orig_fg(bgr_crop, thr=26.0, with_box=False):
    """原牌前景：以外扩环带中位色为背景，Lab 距离阈值分割。"""
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    h, w = bgr_crop.shape[:2]
    if h < 6 or w < 6:
        return None
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    bw = max(2, int(round(min(h, w) * 0.10)))
    ring = np.concatenate([
        lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3),
        lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    m = (dist > thr).astype(np.uint8)
    m = _largest_cc_fill(m)
    t, box = _tight(m)
    if with_box:
        return t, box
    return t


def fit_metrics(orig_mask, asset_mask, allow_scale=(0.92, 1.08)):
    """把素材掩码对齐到原牌前景，返回三项硬指标。"""
    if orig_mask is None or asset_mask is None:
        return None
    oh, ow = orig_mask.shape[:2]
    if oh < 3 or ow < 3:
        return None
    a = asset_mask
    ar = a.shape[1] / float(max(a.shape[0], 1))
    orat = ow / float(max(oh, 1))
    # 只允许近似等比缩放；比例差异过大直接判不合格
    if ar > 0 and (orat / ar > 1.35 or ar / orat > 1.35):
        return {'recall': 0.0, 'iou': 0.0, 'exposed_px': 99.0, 'ratio': round(orat / ar, 3)}
    best = None
    for s in np.linspace(allow_scale[0], allow_scale[1], 5):
        tw = max(2, int(round(ow * s)))
        th = max(2, int(round(oh * s)))
        am = cv2.resize(a, (tw, th), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((oh, ow), np.uint8)
        ox, oy = (ow - tw) // 2, (oh - th) // 2
        x1, y1 = max(0, ox), max(0, oy)
        x2, y2 = min(ow, ox + tw), min(oh, oy + th)
        canvas[y1:y2, x1:x2] = am[(y1 - oy):(y1 - oy) + (y2 - y1),
                                  (x1 - ox):(x1 - ox) + (x2 - x1)]
        inter = int(((orig_mask > 0) & (canvas > 0)).sum())
        osum = int((orig_mask > 0).sum())
        asum = int((canvas > 0).sum())
        if osum == 0 or asum == 0:
            continue
        union = osum + asum - inter
        recall = inter / float(osum)
        iou = inter / float(max(union, 1))
        exposed = (orig_mask > 0) & (canvas == 0)
        if exposed.any():
            inv = (canvas == 0).astype(np.uint8)
            dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
            exposed_px = float(dt[exposed].max())
        else:
            exposed_px = 0.0
        cand = {'recall': recall, 'iou': iou,
                'exposed_px': exposed_px, 'ratio': round(orat / max(ar, 1e-6), 3)}
        if best is None or (cand['recall'] + cand['iou']) > (best['recall'] + best['iou']):
            best = cand
    return best


def check_fit(orig_crop, asset_path, recall_min=0.94, iou_min=0.85,
              exposed_max=2.0, need_color=True):
    """完整判定：返回 (通过, 指标字典)。"""
    om, obox = orig_fg(orig_crop, with_box=True)
    if om is None or obox is None:
        return False, {'reason': 'orig_fg_failed'}
    am, box, ring = asset_fg(asset_path)
    if am is None:
        return False, {'reason': 'asset_mask_failed'}
    m = fit_metrics(om, am)
    if m is None:
        return False, {'reason': 'metrics_failed'}
    info = dict(m)
    if need_color:
        ox1, oy1, ox2, oy2 = obox
        ofam = _ring_family(orig_crop[oy1:oy2, ox1:ox2], om)
        info['orig_color'] = ofam
        info['asset_color'] = ring
        if ofam != ring and 'other' not in (ofam, ring):
            return False, dict(info, reason='color_mismatch')
    if m['recall'] < recall_min:
        return False, dict(info, reason='low_recall')
    if m['exposed_px'] > exposed_max:
        return False, dict(info, reason='corner_exposed')
    if m['iou'] < iou_min:
        return False, dict(info, reason='low_iou')
    return True, dict(info, reason='OK')

# ---------------------------------------------------------------- BODY-COLOR
_BODY_CACHE = {}


def _body_family(bgr, mask):
    """牌面主体色：掩码内缩后的主色（避开外圈边框/描边）。"""
    if bgr is None or mask is None or bgr.size == 0:
        return 'other'
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 10:
        return 'other'
    h, w = m.shape
    k = max(3, int(min(h, w) * 0.22) | 1)
    ker = np.ones((k, k), np.uint8)
    inner = cv2.erode(m, ker, iterations=1)
    if inner.sum() < 10:
        inner = m
    return color_family(bgr, inner)


def asset_body(path):
    """素材牌面主体色（缓存）。"""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return 'other'
    hit = _BODY_CACHE.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    col = 'other'
    try:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is not None:
            if im.ndim == 3 and im.shape[2] == 4:
                mask = (im[:, :, 3] > 32).astype(np.uint8)
                bgr = im[:, :, :3]
            else:
                bgr = im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                h, w = bgr.shape[:2]
                corners = np.array([bgr[0, 0], bgr[0, w - 1],
                                    bgr[h - 1, 0], bgr[h - 1, w - 1]], np.float32)
                bg = corners.mean(axis=0)
                mask = (np.linalg.norm(bgr.astype(np.float32) - bg, axis=2) > 40).astype(np.uint8)
            mask = _largest_cc_fill(mask)
            t, box = _tight(mask)
            if t is not None:
                x1, y1, x2, y2 = box
                col = _body_family(bgr[y1:y2, x1:x2], t)
    except Exception:
        col = 'other'
    _BODY_CACHE[path] = (mt, col)
    return col


def orig_body(crop_bgr):
    """原牌牌面主体色；前景分割失败时退回整块主色。"""
    try:
        om, ob = orig_fg(crop_bgr, with_box=True)
    except Exception:
        om, ob = None, None
    if om is not None and ob is not None:
        a1, b1, a2, b2 = ob
        col = _body_family(crop_bgr[b1:b2, a1:a2], om)
        if col != 'other':
            return col
    return color_family(crop_bgr)
