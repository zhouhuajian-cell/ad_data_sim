#!/usr/bin/env python3
"""GND 素材渲染：贴纹理到地面（投影透视变换）。

支持人工指定：
  items = [
    {"asset": "lane_solid_white", "x": 5.0, "y": -5.25,
     "len_m": 50, "width_m": 0.2, "wear": 0},
    {"asset": "stop_line", "x": 20.0, "y": 0.0,
     "len_m": 3.5, "width_m": 0.4, "wear": 25},
  ]
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")

GND_ASSETS = "/data/gnd_assets"
_META_CACHE = {}


def list_assets():
    """列出所有可用素材及其元数据。"""
    out = []
    if not os.path.isdir(GND_ASSETS):
        return out
    for name in sorted(os.listdir(GND_ASSETS)):
        d = os.path.join(GND_ASSETS, name)
        mp = os.path.join(d, "meta.json")
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        meta["key"] = name
        out.append(meta)
    return out


def load_asset_meta(key):
    if key in _META_CACHE:
        return _META_CACHE[key]
    mp = os.path.join(GND_ASSETS, key, "meta.json")
    if not os.path.isfile(mp):
        return None
    try:
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    meta["key"] = key
    _META_CACHE[key] = meta
    return meta


def load_asset_tex(key, wear=0):
    """加载纹理：wear 0/25/55 三档。"""
    d = os.path.join(GND_ASSETS, key)
    if wear >= 45:
        fn = "tex_wear55.png"
    elif wear >= 15:
        fn = "tex_wear25.png"
    else:
        fn = "tex.png"
    p = os.path.join(d, fn)
    if not os.path.isfile(p):
        p = os.path.join(d, "tex.png")
    if not os.path.isfile(p):
        return None
    return cv2.imread(p, cv2.IMREAD_UNCHANGED)


def _ground_quad(calib, x0, y0, x1, y1):
    """地面上一个矩形的 4 角 → 屏幕坐标 [TL, TR, BR, BL]。"""
    from gnd_calib import ground_to_pixel

    pts = []
    for (gx, gy) in ((x0, y1), (x1, y1), (x1, y0), (x0, y0)):
        uv = ground_to_pixel(calib, gx, gy)
        if uv is None or not np.all(np.isfinite(uv)):
            return None
        pts.append([float(uv[0]), float(uv[1])])
    return np.array(pts, np.float32)


def paint_asset(canvas_alpha, canvas_rgb, calib, item):
    """把一张素材贴到地面上（透视变换）。"""
    key = item.get("asset") or "lane_solid_white"
    meta = load_asset_meta(key)
    if meta is None:
        return False

    x = float(item.get("x", 10.0))       # 前向起点 (m)
    y = float(item.get("y", 0.0))        # 横向中心 (m)
    L = float(item.get("len_m", 0))
    W = float(item.get("width_m", 0))
    wear = float(item.get("wear", 0))

    # 未指定长度/宽度时用 meta 的物理尺寸
    ms = meta.get("size") or [48, 1200]
    horiz = bool(meta.get("horizontal"))
    if L <= 0:
        # 纵向素材：长=高度像素/1000*..., 简化：按 meta 比例给默认
        L = (ms[0] / 1000.0) if horiz else (ms[1] / 1000.0)
    if W <= 0:
        W = (ms[1] / 1000.0) if horiz else (ms[0] / 1000.0)

    tex = load_asset_tex(key, wear)
    if tex is None:
        return False

    # 世界矩形：横向 [y-W/2, y+W/2]，纵向 [x, x+L]
    quad = _ground_quad(calib, x, y - W / 2, x + L, y + W / 2)
    if quad is None:
        return False

    h, w = canvas_alpha.shape[:2]
    tw, th = tex.shape[1], tex.shape[0]

    # 纹理四角 → 屏幕四角
    src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1],
                    [0, th - 1]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, quad)

    # 贴图（warpPerspective 到全画布）
    if tex.ndim == 3 and tex.shape[2] == 4:
        tex_rgb = tex[..., :3]
        tex_a = tex[..., 3].astype(np.float32) / 255.0
    else:
        tex_rgb = tex if tex.ndim == 3 else cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
        tex_a = np.ones(tex.shape[:2], np.float32)

    warped_rgb = cv2.warpPerspective(tex_rgb, Hm, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
    warped_a = cv2.warpPerspective(tex_a, Hm, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    # 叠加（取最大值，避免重叠处叠加过亮）
    canvas_alpha[:] = np.maximum(canvas_alpha, warped_a)
    mask = warped_a > 0.01
    canvas_rgb[mask] = warped_rgb[mask]
    return True


def render_items(calib, items, shape):
    """渲染所有指定素材，返回 (alpha, rgb)。"""
    h, w = shape[:2]
    alpha = np.zeros((h, w), np.float32)
    rgb = np.full((h, w, 3), 236, np.uint8)
    ok = 0
    for it in (items or []):
        try:
            if paint_asset(alpha, rgb, calib, it):
                ok += 1
        except Exception as e:
            print("    素材 %s 失败: %s" % (it.get("asset"), str(e)[:50]))
    return alpha, rgb, ok


def default_items(cfg):
    """没有人工指定时，按 lanes/dash 生成默认线位。"""
    lanes = int(cfg.get("lanes", 3))
    dash = cfg.get("dash", True)
    edge_w = float(cfg.get("edge", 0.20))
    line_w = float(cfg.get("width", 0.15))
    lane_type = str(cfg.get("lane_type") or "lane_dashed_white")

    lane_wid = 3.5
    total = lanes * lane_wid
    ys = [-total / 2 + i * lane_wid for i in range(lanes + 1)]
    items = []
    for i, y_m in enumerate(ys):
        is_edge = (i == 0 or i == len(ys) - 1)
        if is_edge:
            items.append({"asset": "road_edge_white", "x": 4.0,
                          "y": y_m, "len_m": 50.0,
                          "width_m": edge_w, "wear": 0})
        else:
            key = lane_type if dash else "lane_solid_white"
            items.append({"asset": key, "x": 4.0, "y": y_m,
                          "len_m": 50.0, "width_m": line_w, "wear": 0})
    return items
