#!/usr/bin/env python3
"""TSR 形状判定（前景模板 IoU）+ 防露底门控。

替代 BBox 长宽比判形状 —— 圆形/正方形/等边三角形的 BBox 都是 1:1，
光看框分不出来。改为提取原牌 ROI 内的有色前景，与标准形状模板算 IoU。
"""
import cv2
import numpy as np

TEMPLATE_IOU_MIN = 0.65     # 模板 IoU 下限
COVER_LEAK_MAX = 0.12       # 露底率上限

# 类别映射（YOLO 细分时优先用）
CLS_TO_SHAPE = {
    "prohibition": "circle", "speed_limit": "circle",
    "mandatory": "circle", "no_entry": "circle",
    "warning": "triangle", "danger": "triangle",
    "guide": "rect", "direction": "rect", "info": "rect",
}


def _colored_foreground(roi_bgr, include_white=False):
    """FG-COLOR-ONLY: 提取有色前景。

    默认只用红/蓝/黄（标牌的图形部分），不包含白色背板 ——
    否则「白底矩形背板 + 蓝圆标牌」会被判成矩形。
    include_white=True 时才把反光白膜算进去（用于覆盖率检查）。
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (((h <= 12) | (h >= 160)) & (s >= 60) & (v >= 40))
    blue = ((h >= 90) & (h <= 130) & (s >= 60) & (v >= 40))
    yellow = ((h >= 18) & (h <= 38) & (s >= 70) & (v >= 90))
    m = red | blue | yellow
    if include_white:
        m = m | ((s <= 60) & (v >= 160))
    return m


def _templates(h, w):
    """生成三种标准形状模板。"""
    t = {}
    circ = np.zeros((h, w), np.uint8)
    cv2.ellipse(circ, (w // 2, h // 2), (int(w * 0.48), int(h * 0.48)),
                0, 0, 360, 1, -1)
    t["circle"] = circ.astype(bool)

    tri = np.zeros((h, w), np.uint8)
    pts = np.array([[w // 2, int(h * 0.04)],
                    [int(w * 0.04), int(h * 0.94)],
                    [int(w * 0.96), int(h * 0.94)]], np.int32)
    cv2.fillPoly(tri, [pts], 1)
    t["triangle"] = tri.astype(bool)

    rect = np.zeros((h, w), np.uint8)
    cv2.rectangle(rect, (int(w * 0.04), int(h * 0.04)),
                  (int(w * 0.96), int(h * 0.96)), 1, -1)
    t["square"] = rect.astype(bool)
    return t


def detect_sign_color(image_bgr, bbox, cls_hint=None):
    """SIGN-COLOR: 判定原标牌底色语义。

    返回 'blue' / 'yellow' / 'red_white' / 'other'
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return "other"

    roi = image_bgr[y1:y2, x1:x2]
    hh, ww = roi.shape[:2]
    dy, dx = max(1, int(hh * 0.08)), max(1, int(ww * 0.08))
    core = roi[dy:hh - dy, dx:ww - dx]
    if core.shape[0] < 8 or core.shape[1] < 8:
        core = roi

    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    blue = (Hc >= 95) & (Hc <= 130) & (Sc >= 60) & (Vc >= 40)
    red = ((Hc <= 12) | (Hc >= 168)) & (Sc >= 60) & (Vc >= 40)
    yellow = (Hc >= 15) & (Hc <= 38) & (Sc >= 60) & (Vc >= 60)

    tot = float(core.shape[0] * core.shape[1])
    r_blue = float(blue.sum()) / tot
    r_red = float(red.sum()) / tot
    r_yel = float(yellow.sum()) / tot

    if r_blue >= 0.25 and r_blue >= r_red and r_blue >= r_yel:
        return "blue"
    if r_yel >= 0.25 and r_yel > r_blue:
        return "yellow"
    if r_red >= 0.12:
        return "red_white"
    return "other"


def detect_shape_foreground(image_bgr, bbox, cls_hint=None):
    """按前景模板 IoU 判形状。返回 (shape, info)。

    优先级：类别映射 → 前景模板 IoU（>= 0.65）
    判不出返回 ('unknown', {...})
    """
    if cls_hint:
        for k, v in CLS_TO_SHAPE.items():
            if str(cls_hint).lower().startswith(k):
                return v, {"source": "class_hint", "cls": cls_hint}

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return "unknown", {"reason": "too_small"}

    roi = image_bgr[y1:y2, x1:x2]
    # 上下各收 8%，去掉边框外溢
    hh, ww = roi.shape[:2]
    dy, dx = max(1, int(hh * 0.08)), max(1, int(ww * 0.08))
    core = roi[dy:hh - dy, dx:ww - dx]
    if core.shape[0] < 10 or core.shape[1] < 10:
        core = roi

    fg = _colored_foreground(core)
    if fg.sum() < 20:
        return "unknown", {"reason": "no_fg"}

    h, w = fg.shape
    tpls = _templates(h, w)
    ious = {}
    for name, tp in tpls.items():
        inter = np.logical_and(fg, tp).sum()
        union = np.logical_or(fg, tp).sum()
        ious[name] = float(inter / max(union, 1))

    best = max(ious, key=ious.get)
    best_iou = ious[best]

    # SHAPE-VOTE: 小 ROI 上 circle/square 的 IoU 只差零点几，
    # 用颜色语义投票纠正 —— 蓝底/红底标牌基本都是圆形
    hsv_core = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    _h, _s, _v = hsv_core[:, :, 0], hsv_core[:, :, 1], hsv_core[:, :, 2]
    _blue = ((_h >= 90) & (_h <= 130) & (_s >= 60) & (_v >= 40))
    _red = (((_h <= 12) | (_h >= 160)) & (_s >= 70) & (_v >= 40))
    _yellow = ((_h >= 18) & (_h <= 38) & (_s >= 70) & (_v >= 90))
    _tot = max(int(fg.sum()), 1)
    r_blue = float(_blue.sum()) / _tot
    r_red = float(_red.sum()) / _tot
    r_yel = float(_yellow.sum()) / _tot

    # 圆形票：蓝底（指示/强制）或红圈（禁令/限速）
    if (r_blue >= 0.45 or r_red >= 0.45) and ious["circle"] >= best_iou - 0.10:
        best = "circle"
        best_iou = max(ious["circle"], best_iou)
    # 三角票：黄底
    elif r_yel >= 0.45 and ious["triangle"] >= best_iou - 0.10:
        best = "triangle"
        best_iou = max(ious["triangle"], best_iou)

    if best_iou < TEMPLATE_IOU_MIN and best not in ("circle",):
        return "unknown", {"reason": "low_iou",
                           "ious": {k: round(v, 3) for k, v in ious.items()}}
    return best, {"source": "template+vote", "iou": round(best_iou, 3),
                  "blue": round(r_blue, 2), "red": round(r_red, 2),
                  "yellow": round(r_yel, 2),
                  "ious": {k: round(v, 3) for k, v in ious.items()}}


def cover_check(image_bgr, bbox, asset_alpha):
    """防露底：原牌有色前景落在素材 alpha 之外的比例。

    asset_alpha: (h, w) uint8/float，已 resize 到 bbox 尺寸，255=不透明
    返回 (ok, leak_ratio, info)
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return True, 0.0, {"reason": "empty"}

    roi = image_bgr[y1:y2, x1:x2]
    a = asset_alpha
    if a.shape[:2] != roi.shape[:2]:
        a = cv2.resize(a.astype(np.float32), (roi.shape[1], roi.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    cover = (a > 127)

    # COVER-COLOR-ONLY: 只用【彩色】前景（红/蓝/黄），不含白色
    # 否则路边灰色路面/杆子会被当成白底算进"露底"
    fg = _colored_foreground(roi, include_white=False).astype(np.uint8)
    if fg.sum() < 15:
        # 没有彩色 → 说明是白底黑字牌，退化为用亮度突变检测
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg = ((np.abs(g.astype(np.int16) - int(np.median(g))) > 45)
              ).astype(np.uint8)
    if fg.sum() < 15:
        return True, 0.0, {"reason": "no_fg"}
    # 去掉贴着外边框的连通块（框外背景）
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    clean = np.zeros_like(fg)
    h_roi, w_roi = fg.shape
    for i in range(1, n_lab):
        x, y, w, h, area = st[i]
        # 跨越整个边框的细长块 = 背景/杆子，丢弃
        if w >= w_roi * 0.85 and h <= 3:
            continue
        if h >= h_roi * 0.85 and w <= 3:
            continue
        if area < 4:
            continue
        clean[lab == i] = 1
    fg = clean
    if fg.sum() < 15:
        return True, 0.0, {"reason": "no_core_fg"}

    fg_b = fg.astype(bool)
    leak = float((fg_b & (~cover)).sum()) / float(fg_b.sum())
    ok = leak <= COVER_LEAK_MAX
    return ok, round(leak, 4), {"fg_px": int(fg_b.sum()),
                                "leak_px": int((fg_b & (~cover)).sum()),
                                "leak": round(leak, 4),
                                "limit": COVER_LEAK_MAX}
