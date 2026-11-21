#!/usr/bin/env python3
"""统一仿真流水线：检测 → 抹除 → 放置 → 融合 → 打包 clip。

OBJ 流程：
  1) YOLO 检测已有障碍物（车/人）→ bbox
  2) LaMa 抹除这些物体 → 干净背景
  3) 在指定世界坐标放置新素材
  4) 阴影 + 遮挡融合

GND 流程：
  1) IPM 检测已有车道线 → 世界坐标
  2) 沿检测到的线抹除（涂背景色）
  3) 画新车道线（参数化）
  4) 磨损 + 遮挡融合

输出：
  <out>/frames/<stem>_sim.jpg    仿真后的帧
  <out>/frames/<stem>_orig.jpg   原图
  <out>/annotations/<stem>.json  标注
  <out>/clip.mp4                 打包的 clip
  <out>/manifest.json            清理记录
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from asset_lib import OBJ_ROOT, pick, refresh
from data_source import list_clips
from label_gate import list_clip_frames, load_frame_labels
from real_calib import body_to_pixel_real, load_cam_real
from annot_3d import (build_frame_annotation, write_clip,
                      write_frame)

REAL_ASSETS = "/data/obj_assets_real"

try:
    from ground_masks import foreground_mask, hood_mask, road_mask
    HAS_MASKS = True
except ImportError:
    HAS_MASKS = False

# ---- 懒加载 ----
_YOLO = None
_LAMA = None


def get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        # DET-V2: 用 done_data 训练的检测模型
        for mp in ("/opt/ad_mining/lane_obj_v2.pt",
                   "/opt/ad_mining/lane_obj_v1.pt",
                   "/opt/ad_mining/yolov8x.pt"):
            if os.path.isfile(mp):
                _YOLO = YOLO(mp)
                print("    检测模型:", os.path.basename(mp))
                break
        else:
            _YOLO = YOLO("/opt/ad_mining/yolov8x.pt")
    return _YOLO


def get_lama():
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()
    return _LAMA


COCO_WANT = {2: "car", 5: "bus", 7: "truck", 3: "motorcycle", 1: "bicycle"}

# DET-V2: 新模型 8 类（done_data 类别）
NEW_CLASSES = ["car", "pedestrian", "three", "truck",
               "van", "rider", "special", "bus"]
# 需要抹除/统计的类别
DET_KEEP = set(NEW_CLASSES) | {"bus", "motorcycle", "bicycle", "person"}


# ============ OBJ：检测 ============
def detect_obstacles(bg, conf=0.25):
    """YOLO 检测已有障碍物。"""
    m = get_yolo()
    r = m.predict(bg, conf=conf, verbose=False)[0]
    h, w = bg.shape[:2]
    out = []
    names = r.names if hasattr(r, "names") else {}
    for b in r.boxes:
        c = int(b.cls[0])
        cname = names.get(c, "") if isinstance(names, dict) else ""
        # DET-V2: 兼容新模型（类别名）和 COCO（类别 id）
        if cname in DET_KEEP:
            cls_name = cname
        elif c in COCO_WANT:
            cls_name = COCO_WANT[c]
        else:
            continue
        xy = b.xyxy[0].cpu().numpy()
        x0, y0, x1, y1 = [float(v) for v in xy]
        bw = x1 - x0
        bh = y1 - y0
        # 过滤：引擎盖误检（贴底 + 超宽）
        if y1 > h * 0.92 and bw > w * 0.5:
            continue
        # 过滤：太小
        if bw * bh < 100:
            continue
        out.append({"cls": cls_name,
                    "conf": round(float(b.conf[0]), 3),
                    "bbox": [round(x0, 1), round(y0, 1),
                             round(x1, 1), round(y1, 1)]})
    return out


# ============ OBJ：抹除 ============
def refine_mask_in_box(bg, x0, y0, x1, y1, grow=3):
    """INPAINT-V2: 在框内用 GrabCut 细化为物体轮廓（不用整个矩形）。"""
    h, w = bg.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w - 1, int(x1)), min(h - 1, int(y1))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    roi = bg[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]

    # 初值：框内 15% 边距当背景，中心当可能前景
    gc = np.zeros((rh, rw), np.uint8)
    mx = max(2, int(rw * 0.06))
    my = max(2, int(rh * 0.06))
    gc[:] = cv2.GC_BGD
    gc[my:rh - my, mx:rw - mx] = cv2.GC_PR_FGD
    # 中心 50% 当确定前景（更激进些，保证擦干净）
    cx0, cx1 = int(rw * 0.25), int(rw * 0.75)
    cy0, cy1 = int(rh * 0.2), int(rh * 0.85)
    gc[cy0:cy1, cx0:cx1] = cv2.GC_FGD
    # 四周作为确定背景
    bd = max(2, int(min(rw, rh) * 0.03))
    gc[:bd, :] = cv2.GC_BGD
    gc[rh - bd:, :] = cv2.GC_BGD
    gc[:, :bd] = cv2.GC_BGD
    gc[:, rw - bd:] = cv2.GC_BGD

    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(roi, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        # 填洞
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        # 取最大连通块
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            return None
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8) * 255
        frac = (m > 0).mean()
        # 太小或太大都不信，退回整矩形
        if frac < 0.08 or frac > 0.95:
            return None
        if grow:
            m = cv2.dilate(m, np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8))
        full = np.zeros((h, w), np.uint8)
        full[y0:y1, x0:x1] = m
        return full
    except Exception:
        return None


def inpaint_objects(bg, boxes, expand=6):
    """INPAINT-V2: 精确轮廓擦除（不再整框擦成灰色）。"""
    if not boxes:
        return bg, np.zeros(bg.shape[:2], np.uint8)
    h, w = bg.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    for b in boxes:
        bb = b["bbox"]
        x0 = bb[0] - expand
        y0 = bb[1] - expand
        x1 = bb[2] + expand
        y1 = bb[3] + expand
        # 关键：裁剪到图像内（之前越界导致整屏擦除）
        cx0 = max(0, int(x0)); cy0 = max(0, int(y0))
        cx1 = min(w, int(x1)); cy1 = min(h, int(y1))
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            continue
        area_ratio = (cx1 - cx0) * (cy1 - cy0) / float(w * h)
        # 超过画面 25% 的框：物体太大/检测错，跳过擦除（避免毁掉整帧）
        if area_ratio > 0.25:
            continue

        fine = refine_mask_in_box(bg, cx0, cy0, cx1, cy1, grow=3)
        if fine is not None:
            mask = np.maximum(mask, fine)
        else:
            mask[cy0:cy1, cx0:cx1] = 255

    if (mask > 0).sum() < 50:
        return bg, mask

    n_mask = int((mask > 0).sum())
    print("    擦除像素: %d (%.2f%%)" % (n_mask, n_mask / float(w * h) * 100))

    try:
        lama = get_lama()
        from PIL import Image
        rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        out = lama(Image.fromarray(rgb), Image.fromarray(mask))
        out = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        return out, mask
    except Exception as e:
        print("    LaMa 失败，退回 cv2.inpaint:", str(e)[:60])
        out = cv2.inpaint(bg, mask, 7, cv2.INPAINT_TELEA)
        return out, mask


# ============ 素材 ============
def find_sprite(cls, spec_path=None):
    """优先真实抠图素材。SPEC-ASSET-POS: 支持指定 sprite 路径。"""
    # ① 指定素材优先
    if spec_path:
        cand = spec_path
        if not os.path.isabs(cand):
            # 支持 "car/xxx.png" 或 "xxx.png" 相对形式
            for root in (REAL_ASSETS, "/data/obj_assets_pc",
                         "/data/obj_assets", ASSET_ROOT):
                p2 = os.path.join(root, cand)
                if os.path.isfile(p2):
                    cand = p2
                    break
                p3 = os.path.join(root, cls, cand)
                if os.path.isfile(p3):
                    cand = p3
                    break
        if os.path.isfile(cand):
            im = cv2.imread(cand, cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                return im, {"source": "specified",
                            "path": os.path.basename(cand)}

    if os.path.isdir(os.path.join(REAL_ASSETS, cls)):
        fs = sorted(glob.glob(os.path.join(REAL_ASSETS, cls, "*.png")))
        if fs:
            im = cv2.imread(fs[0], cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                return im, {"source": "real_extract"}
    if os.path.isdir(os.path.join(REAL_ASSETS, cls + "s")):
        fs = sorted(glob.glob(os.path.join(REAL_ASSETS, cls + "s", "*.png")))
        if fs:
            im = cv2.imread(fs[0], cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                return im, {"source": "real_extract"}
    path, meta = pick(cls, "front", "day", OBJ_ROOT)
    if path and os.path.isfile(path):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            return im, meta
    return None, None


SIZE_M = {
    "car": (1.80, 1.50), "suv": (1.90, 1.75), "truck": (2.00, 2.30),
    "van": (2.00, 2.20), "pedestrian": (0.55, 1.70), "rider": (0.70, 1.75),
    "three": (1.30, 1.60), "cone": (0.30, 0.70), "bus": (2.55, 3.30),
}


def project_quad(cam, x, y, w_m, h_m, sprite_aspect=None):
    """QUAD-ASPECT-FIX: 按精灵宽高比计算 quad，避免拉伸。

    sprite_aspect = sprite_w / sprite_h，为 None 时用 w_m/h_m。
    """
    quad = []
    for yy in (y + w_m / 2, y - w_m / 2):
        uv, z, ok = body_to_pixel_real(cam, [[x, yy, 0.0]])
        if not ok[0] or not np.isfinite(uv[0]).all():
            return None
        quad.append(uv[0])

    uvl, _, _ = body_to_pixel_real(cam, [[x, y + 0.5, 0.0]])
    uvr, _, _ = body_to_pixel_real(cam, [[x, y - 0.5, 0.0]])
    px_per_m = abs(uvl[0][0] - uvr[0][0])

    left, right = quad[0], quad[1]
    px_w = abs(right[0] - left[0])

    # HEIGHT-PHYS: 高度直接用物理高度 × 像素/米
    px_h = h_m * px_per_m

    # 合理性保护
    px_h = float(np.clip(px_h, px_w * 0.33, px_w * 3.0))

    return np.array([(left[0], left[1] - px_h),
                     (right[0], right[1] - px_h),
                     right, left], np.float32)


# ============ 放置 ============
def sample_scene_light(bg, quad, exclude_mask=None):
    """采样目标区域周围的光照（亮度 + 色温）。"""
    h, w = bg.shape[:2]
    xs = quad[:, 0]
    ys = quad[:, 1]
    x1 = max(0, int(xs.min()) - 40)
    y1 = max(0, int(ys.min()) - 40)
    x2 = min(w, int(xs.max()) + 40)
    y2 = min(h, int(ys.max()) + 40)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = bg[y1:y2, x1:x2]
    # 排除中心（物体自己）
    cx1, cy1 = int(xs.min()) - x1, int(ys.min()) - y1
    cx2, cy2 = int(xs.max()) - x1, int(ys.max()) - y1
    m = np.ones(roi.shape[:2], np.float32)
    m[max(0, cy1):cy2, max(0, cx1):cx2] = 0
    if exclude_mask is not None:
        em = exclude_mask[y1:y2, x1:x2]
        m[em > 0] = 0
    px = roi[m > 0]
    if len(px) < 100:
        px = roi.reshape(-1, 3)
    if len(px) < 20:
        return None
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lum = float(np.median(hsv[:, 2]))
    bgr_mean = px.reshape(-1, 3).astype(np.float32).mean(axis=0)
    mn = max(float(bgr_mean.mean()), 1.0)
    gain = bgr_mean / mn
    # 场景锐度（用于估计景深）
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return {"lum": lum, "gain": gain, "sharp": sharp}


def place_sprite(bg, sp, quad, protect, shadow=0.35, scene_light=None,
                 depth_m=None):
    """COMPOSE-V2: 带羽化 + 色温匹配 + 景深模糊的合成。"""
    h, w = bg.shape[:2]
    sh, sw = sp.shape[:2]
    if sw < 4 or sh < 4:
        return bg

    src = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1],
                    [0, sh - 1]], np.float32)
    Pm = cv2.getPerspectiveTransform(src, quad)

    # 目标尺寸（决定羽化宽度）
    tw = max(4, int(abs(quad[1][0] - quad[0][0])))
    th = max(4, int(abs(quad[3][1] - quad[0][1])))

    a = sp[..., 3].astype(np.float32) / 255.0
    rgb = sp[..., :3].astype(np.float32)

    # ① 内部边缘收缩（去掉抠图毛边）
    erode_px = max(1, int(min(sw, sh) * 0.02))
    a_er = cv2.erode((a * 255).astype(np.uint8),
                     np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8))
    a = np.maximum(a, a_er.astype(np.float32) / 255.0 * 0.0)  # 保留原 alpha
    a = a_er.astype(np.float32) / 255.0

    # ② 色温 / 光照匹配
    if scene_light:
        hsv = cv2.cvtColor(rgb.clip(0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        cur_lum = float(np.median(hsv[hsv[..., 2] > 30][:, 2])) \
            if (hsv[..., 2] > 30).any() else 150.0
        cur_lum = max(cur_lum, 1.0)
        # 目标亮度：环境亮度 × 0.95（物体比环境稍亮）
        tgt = np.clip(scene_light["lum"] * 1.05, 25, 235)
        k = tgt / cur_lum
        rgb = rgb * np.clip(k, 0.35, 2.2)
        # 色温
        rgb = rgb * np.array(scene_light["gain"], np.float32)[None, None, :]

    # ③ 景深模糊（远处物体天然模糊）
    if depth_m and depth_m > 15:
        sigma = min((depth_m - 15) / 40.0, 1.2)
        if sigma > 0.2:
            rgb = cv2.GaussianBlur(rgb, (0, 0), sigma)
            a = cv2.GaussianBlur(a, (0, 0), sigma * 0.6)

    rgb = np.clip(rgb, 0, 255)
    prem = rgb * a[..., None]

    a_w = cv2.warpPerspective(a, Pm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    p_w = cv2.warpPerspective(prem, Pm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    a_w = np.clip(a_w, 0, 1)
    straight = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
    straight[a_w <= 1e-4] = 0

    # ④ 关键：边缘羽化（按目标尺寸的 6%，且至少 2px）
    feather = max(2, int(min(tw, th) * 0.06))
    if feather % 2 == 0:
        feather += 1
    keep = cv2.GaussianBlur(a_w, (feather, feather), 0)
    keep = np.clip(keep, 0, 1)

    if shadow > 0.01:
        base = quad.astype(np.int32)
        cx = int((base[2][0] + base[3][0]) / 2)
        cy = int((base[2][1] + base[3][1]) / 2)
        rw = max(int(abs(base[1][0] - base[0][0]) * 0.55), 4)
        rh = max(int(rw * 0.25), 2)
        shm = np.zeros((h, w), np.float32)
        cv2.ellipse(shm, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)
        shm = cv2.GaussianBlur(shm, (15, 15), 0) * shadow * 0.7
        bg = (bg.astype(np.float32) * (1 - shm[..., None])).astype(np.uint8)

    keep = keep * (1.0 - protect)
    out = (bg.astype(np.float32) * (1 - keep[..., None])
           + straight * keep[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def place_sprite_OLD(bg, sp, quad, protect, shadow=0.35):
    h, w = bg.shape[:2]
    sh, sw = sp.shape[:2]
    src = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1],
                    [0, sh - 1]], np.float32)
    Pm = cv2.getPerspectiveTransform(src, quad)

    a = sp[..., 3].astype(np.float32) / 255.0
    rgb = sp[..., :3].astype(np.float32)
    prem = rgb * a[..., None]

    a_w = cv2.warpPerspective(a, Pm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    p_w = cv2.warpPerspective(prem, Pm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    a_w = np.clip(a_w, 0, 1)
    straight = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
    straight[a_w <= 1e-4] = 0

    if shadow > 0.01:
        base = quad.astype(np.int32)
        cx = int((base[2][0] + base[3][0]) / 2)
        cy = int((base[2][1] + base[3][1]) / 2)
        rw = max(int(abs(base[1][0] - base[0][0]) * 0.55), 4)
        rh = max(int(rw * 0.25), 2)
        shm = np.zeros((h, w), np.float32)
        cv2.ellipse(shm, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)
        shm = cv2.GaussianBlur(shm, (15, 15), 0) * shadow
        bg = (bg.astype(np.float32) * (1 - shm[..., None])).astype(np.uint8)

    keep = a_w * (1.0 - protect)
    out = (bg.astype(np.float32) * (1 - keep[..., None])
           + straight * keep[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def get_protect(bg, cfg):
    if not HAS_MASKS or not cfg.get("occlusion", True):
        return np.zeros(bg.shape[:2], np.float32)
    try:
        road = road_mask(bg)
        fg = foreground_mask(bg, road)
        hood = hood_mask(bg, frac=0.18)
        return np.clip(np.maximum(fg, hood), 0, 1)
    except Exception:
        return np.zeros(bg.shape[:2], np.float32)


# ============ GND：检测 + 抹除 + 绘制 ============
def detect_lanes(bg, cam):
    """IPM 检测车道线（复用 detect_lanes.py）。"""
    try:
        from detect_lanes import cluster_lanes, detect_lanes_ipm
        pts = detect_lanes_ipm(bg, cam, None)
        return cluster_lanes(pts)
    except Exception as e:
        print("    车道线检测失败:", str(e)[:60])
        return []


def erase_lanes(bg, cam, lanes, expand=0.12):
    """沿检测到的车道线抹除（涂周围路面中值色）。"""
    if not lanes:
        return bg
    h, w = bg.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for ln in lanes:
        for x in np.arange(ln["x_min"], ln["x_max"], 0.4):
            y = ln["slope"] * x + (ln["y_at_near"] - ln["slope"] * ln["x_min"])
            wm = ln["width_m"] + expand * 2
            for yy in (y - wm / 2, y, y + wm / 2):
                uv, z, ok = body_to_pixel_real(cam, [[x, yy, 0.0]])
                if ok[0] and np.isfinite(uv[0]).all():
                    cv2.circle(mask, (int(uv[0][0]), int(uv[0][1])), 4, 255, -1)
    if mask.sum() == 0:
        return bg
    try:
        lama = get_lama()
        from PIL import Image
        rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        out = lama(Image.fromarray(rgb), Image.fromarray(mask))
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.inpaint(bg, mask, 7, cv2.INPAINT_TELEA)


def draw_lanes(bg, cam, cfg):
    """画新车道线。"""
    lanes = int(cfg.get("lanes", 3))
    dash = cfg.get("dash", True)
    edge_w = float(cfg.get("edge", 0.20))
    line_w = float(cfg.get("width", 0.15))
    wear = float(cfg.get("wear", 0.25))
    opacity = float(cfg.get("opacity", 1.0))
    protect = get_protect(bg, cfg)

    h, w = bg.shape[:2]
    paint = np.zeros((h, w), np.uint8)

    lane_wid = 3.5
    total = lanes * lane_wid
    ys = [-total / 2 + i * lane_wid for i in range(lanes + 1)]

    DASH_ON, DASH_GAP = 6.0, 9.0
    for i, y in enumerate(ys):
        is_edge = (i == 0 or i == len(ys) - 1)
        wdt = edge_w if is_edge else line_w
        dashed = (not is_edge) and dash
        prev = None
        for x in np.arange(5.0, 55.0, 0.5):
            if dashed and (x % (DASH_ON + DASH_GAP)) > DASH_ON:
                prev = None
                continue
            uvl, _, ok1 = body_to_pixel_real(cam, [[x, y + wdt / 2, 0.0]])
            uvr, _, ok2 = body_to_pixel_real(cam, [[x, y - wdt / 2, 0.0]])
            if not (ok1[0] and ok2[0]):
                prev = None
                continue
            if not (np.isfinite(uvl[0]).all() and np.isfinite(uvr[0]).all()):
                prev = None
                continue
            cur = (uvl[0], uvr[0], x)
            if prev is not None and abs(cur[2] - prev[2]) <= 1.0:
                q = np.array([prev[0], cur[0], cur[1], prev[1]], np.float32)
                if np.all(np.isfinite(q)):
                    cv2.fillConvexPoly(paint, np.rint(q).astype(np.int32), 255)
            prev = cur

    if wear > 0.01:
        rng = np.random.default_rng(hash(str(lanes)) & 0xFFFFFFFF)
        low = rng.random((max(h // 8, 2), max(w // 8, 2))).astype(np.float32)
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
        low = cv2.GaussianBlur(low, (0, 0), max(min(h, w) * 0.02, 1.5))
        thr = 0.10 + 0.75 * wear
        wt = np.ones((h, w), np.float32)
        holes = low < thr
        wt[holes] = np.clip(low[holes] / max(thr, 1e-3), 0.0, 1.0)
        paint = (paint.astype(np.float32) * wt).astype(np.uint8)

    alpha = (paint.astype(np.float32) / 255.0) * (1 - protect) * opacity
    alpha = np.clip(alpha, 0, 1)
    out = (bg.astype(np.float32) * (1 - alpha[..., None])
           + np.array([246, 246, 246], np.float32) * alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), paint


# ============ 主流程 ============
def process_clip(batch, clip_id, module, cfg, out_root, nframes=20):
    # GND-SOURCE: 支持 OBJ(real_clips) 和 GND(gnd_train) 两种数据源
    clip_dir = None
    if module == "gnd":
        cand = os.path.join("/data/test_data/gnd_train", batch, clip_id)
        if os.path.isdir(os.path.join(cand, "img_ori")):
            clip_dir = cand
    if clip_dir is None:
        for name, p, src in list_clips():
            if name != batch:
                continue
            cand = os.path.join(p, clip_id)
            if os.path.isdir(os.path.join(cand, "cam_st_x2f_front")):
                clip_dir = cand
                break
    if not clip_dir:
        raise SystemExit("找不到 clip: %s/%s" % (batch, clip_id))

    # GND-FRAMES-SRC: GND 用 img_ori，OBJ 用 cam_st_x2f_front
    if module == "gnd":
        _id = os.path.join(clip_dir, "img_ori")
        _all = []
        if os.path.isdir(_id):
            for f in sorted(os.listdir(_id)):
                if f.lower().endswith(".jpg"):
                    _all.append((os.path.join(_id, f), None))
    else:
        _all = list_clip_frames(clip_dir)
    frames = _all if (not nframes or nframes <= 0) else _all[:nframes]
    if not frames:
        raise SystemExit("无帧")

    cam = load_cam_real(batch, clip_id)
    if cam is None:
        if module == "gnd":
            cam = {}   # GND 用 gnd_calib，不用 real_calib
        else:
            raise SystemExit("无标定")

    fd = os.path.join(out_root, "frames")
    ad = os.path.join(out_root, "annotations")
    os.makedirs(fd, exist_ok=True)
    os.makedirs(ad, exist_ok=True)

    refresh()
    manifest = []

    clip_frames_ann = []

    # SEQ-CONSIST: 预先确定合成物体的世界位置（整个 clip 固定）
    _seq_positions = None
    if module == "obj" and str(cfg.get("edit_mode") or "add") == "add":
        try:
            import render_core as _rc0
            from real_calib import body_to_pixel_real as _b2p
            # 用第一帧的空位掩码定位置
            _f0 = cv2.imread(frames[0][0])
            if _f0 is not None:
                _H, _W = _f0.shape[:2]
                _drv0 = _rc0.build_drivable(_f0.shape, calib=None, cam=cam)
                # 用第一帧的检测框（粗略：整个中央禁区）
                _obs0 = []
                try:
                    _lab0 = load_frame_labels(clip_dir,
                                              xml_path=frames[0][1])
                    if _lab0:
                        _obs0 = [[o["bbox2d"][0], o["bbox2d"][1],
                                  o["bbox2d"][2], o["bbox2d"][3]]
                                 for o in (_lab0.get("objects") or [])
                                 if (o.get("bbox2d") or [0, 0, 0, 0])[2] > 0]
                except Exception:
                    pass
                _spawn0 = _rc0.filter_boxes_for_spawn(
                    _drv0, _obs0, margin=18, check=(40, 30))
                # 反投影采样 cnt 个点
                _cnt = int(cfg.get("count", 3))
                _dmin = float(cfg.get("dist_min", 15))
                _dmax = float(cfg.get("dist_max", 45))
                _ys, _xs = np.where(_spawn0 > 0)
                if len(_xs) > 300:
                    from real_calib import pixel_to_body_real as _p2b
                    _cands = []
                    # 按距离分桶，取分散的点
                    for _k in range(min(_cnt * 8, 40)):
                        _ii = int(len(_xs) * (_k + 0.5) / min(_cnt * 8, 40))
                        _u, _v = int(_xs[_ii]), int(_ys[_ii])
                        _pb = _p2b(cam, _u, _v)
                        if _pb is None:
                            continue
                        _px, _py = float(_pb[0]), float(_pb[1])
                        if _dmin <= _px <= _dmax:
                            _cands.append((_px, _py))
                    if _cands:
                        # 取 cnt 个分散点
                        _step = max(1, len(_cands) // _cnt)
                        _seq_positions = [_cands[i0 * _step]
                                          for i0 in range(_cnt)
                                          if i0 * _step < len(_cands)]
                        print("    时序一致位置: %s" % [
                            (round(p[0], 1), round(p[1], 1))
                            for p in _seq_positions])
        except Exception as _e:
            print("    时序位置预计算失败:", str(_e)[:60])
            _seq_positions = None

    for i, (img_p, xml_n) in enumerate(frames):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        bg = cv2.imread(img_p)
        if bg is None:
            continue
        h0, w0 = bg.shape[:2]

        # 读真值标注（GND 无 XML 标注，跳过）
        lab = None
        if xml_n is not None:
            lab = load_frame_labels(clip_dir, xml_path=xml_n)
        real_objs = (lab.get("objects") or []) if lab else []

        rec = {"stem": stem, "module": module, "steps": {}}
        synth_list = []
        removed_list = []

        if module == "obj":
            # OBJ-EDIT-MODE: 三任务分流（add / replace / erase）
            edit_mode = str(cfg.get("edit_mode") or "add")
            rec["steps"]["edit_mode"] = edit_mode

            # 1 检测
            obs = detect_obstacles(bg)
            rec["steps"]["detect"] = len(obs)

            if edit_mode == "add":
                # 新增模式：完全不擦除，保留原车（最安全）
                clean = bg.copy()
                mask = np.zeros(bg.shape[:2], np.uint8)
                removed_list = []
                rec["steps"]["erase_px"] = 0
            elif edit_mode == "replace":
                # 替换模式：只擦小目标，大目标跳过（可能无背景观察）
                # MODE-FIX-V2: 大目标跳过（画面 >2% 或高度 >40% 视为无背景观察）
                H0, W0 = bg.shape[:2]
                area_thr = H0 * W0 * 0.02
                small_obs = []
                for o in obs:
                    bb = o["bbox"]
                    bw = max(0, bb[2]-bb[0])
                    bh = max(0, bb[3]-bb[1])
                    a = bw * bh
                    if a > area_thr or bh > H0 * 0.40:
                        continue
                    small_obs.append(o)
                removed_list = [{"cls": o["cls"], "bbox": o["bbox"],
                                 "conf": o["conf"]} for o in small_obs]
                clean, mask = inpaint_objects(bg, small_obs)
                rec["steps"]["erase_px"] = int((mask > 0).sum())
                rec["steps"]["skipped_big"] = len(obs) - len(small_obs)
            else:   # erase（旧行为，兼容）
                removed_list = [{"cls": o["cls"], "bbox": o["bbox"],
                                 "conf": o["conf"]} for o in obs]
                clean, mask = inpaint_objects(bg, obs)
                rec["steps"]["erase_px"] = int((mask > 0).sum())

            # RENDER-CORE-WIRED: 生成保护区和空位
            protected_mask = None
            spawn_mask = None
            try:
                import render_core as _rc
                H1, W1 = bg.shape[:2]
                # 保护区：所有检测框（除要替换的）
                _boxes = [o["bbox"] for o in obs]
                _excl = set(range(len(obs))) if edit_mode == "replace" else None
                protected_mask = _rc.build_protected(
                    bg.shape, _boxes, exclude_ids=_excl, margin=10)

                # 可行驶区：标定投影 > 几何兜底
                _drv = _rc.build_drivable(bg.shape, calib=None, cam=cam)
                # 空位：扣除已有目标
                spawn_mask = _rc.filter_boxes_for_spawn(
                    _drv, _boxes, margin=18, check=(40, 30))
                rec["steps"]["protected_pct"] = round(
                    float(np.count_nonzero(protected_mask)) /
                    protected_mask.size * 100, 1)
                rec["steps"]["spawn_pct"] = round(
                    float(np.count_nonzero(spawn_mask)) /
                    spawn_mask.size * 100, 1)
            except Exception as _e:
                print("    render_core 准备失败:", str(_e)[:60])

            # 3 放置
            sp, meta = find_sprite(cfg.get("cls", "car"),
                                   spec_path=cfg.get("sprite_path"))
            if sp is None:
                out = clean
                rec["steps"]["place"] = 0
            else:
                w_m, h_m = SIZE_M.get(cfg.get("cls", "car"), (1.8, 1.5))
                scl = float(cfg.get("scale", 1.0))
                w_m *= scl
                h_m *= scl
                protect = get_protect(clean, cfg)
                cnt = int(cfg.get("count", 3))
                dmin = float(cfg.get("dist_min", 10))
                dmax = float(cfg.get("dist_max", 35))
                lat = float(cfg.get("lat", 3.0))
                out = clean
                placed = 0

                # OBJ-EDIT-MODE: add 模式用空位检测避开原车
                spawn_ok = spawn_mask   # 用 render_core 生成

                # SPAWN-PICK: 优先从空位掩码采样落地点（add 模式）
                _cands = []
                if edit_mode == "add" and spawn_mask is not None:
                    _ys, _xs = np.where(spawn_mask > 0)
                    if len(_xs) > 200:
                        # 分 cnt 段，各取中位点，保证分散
                        idxs = np.array_split(
                            np.argsort(_ys), max(cnt, 1))
                        for _seg in idxs:
                            if len(_seg) == 0:
                                continue
                            _mid = _seg[len(_seg) // 2]
                            _u, _v = int(_xs[_mid]), int(_ys[_mid])
                            # 反投影到自车坐标
                            try:
                                from real_calib import pixel_to_body_real
                                _pb = pixel_to_body_real(cam, _u, _v)
                                if _pb is not None:
                                    _px, _py = float(_pb[0]), float(_pb[1])
                                    if dmin <= _px <= dmax:
                                        _cands.append((_px, _py))
                            except Exception:
                                pass

                # SPEC-ASSET-POS: 人工指定位置优先
                _manual = cfg.get("manual_pos")
                if _manual and isinstance(_manual, (list, tuple)):
                    _manual = list(_manual)
                else:
                    _manual = []

                for k in range(cnt):
                    # SEQ-CONSIST: 优先用 clip 级固定位置
                    if k < len(_manual) and _manual[k]:
                        _mp = _manual[k]
                        x, y = float(_mp[0]), float(_mp[1])
                    elif _seq_positions and k < len(_seq_positions):
                        x, y = _seq_positions[k]
                    elif _cands and k < len(_cands):
                        x, y = _cands[k]
                    else:
                        x = dmin + (dmax - dmin) * (k + 0.5) / max(cnt, 1)
                        y = (-lat) + 2 * lat * (k / max(cnt - 1, 1))
                    # QUAD-ASPECT-FIX: 传精灵宽高比
                    _asp = (float(sp.shape[1]) / float(sp.shape[0])
                            if sp is not None and sp.shape[0] > 0 else None)
                    quad = project_quad(cam, x, y, w_m, h_m,
                                        sprite_aspect=_asp)
                    if quad is None:
                        continue
                    # OBJ-EDIT-MODE: add 模式检查落地点是否在空位
                    if spawn_ok is not None:
                        # 用底面中心点
                        cx0 = int(np.clip(
                            (quad[2][0] + quad[3][0]) / 2.0, 0, w0 - 1))
                        cy0 = int(np.clip(
                            (quad[2][1] + quad[3][1]) / 2.0, 0, h0 - 1))
                        if spawn_ok[cy0, cx0] == 0:
                            rec["steps"]["skip_occupied"] = \
                                rec["steps"].get("skip_occupied", 0) + 1
                            continue
                    # RENDER-CORE-WIRED: 正片叠底阴影 + 线性空间合成
                    sl = sample_scene_light(clean, quad, protect)
                    try:
                        import render_core as _rc2
                        _sh = _rc2.shadow_mask_from_sprite(sp)
                        out = _rc2.composite_multiply(
                            out, sp, _sh, quad,
                            scene_light=sl, depth_m=x,
                            protected=protected_mask,
                            shadow_strength=float(cfg.get("shadow", 0.65)))
                    except Exception as _e2:
                        print("    composite 失败，回退:", str(_e2)[:50])
                        out = place_sprite(out, sp, quad, protect,
                                           float(cfg.get("shadow", 0.35)),
                                           scene_light=sl, depth_m=x)
                    placed += 1
                    bx = [float(quad[:, 0].min()), float(quad[:, 1].min()),
                          float(quad[:, 0].max()), float(quad[:, 1].max())]
                    synth_list.append({
                        "cls": cfg.get("cls", "car"),
                        "x": x, "y": y,
                        "w_m": w_m, "h_m": h_m,
                        "bbox2d": bx, "meta": meta,
                    })
                rec["steps"]["place"] = placed

        else:  # gnd
            # GND-CLEAN-WIRED: 三步净面管线（全量提取→LaMa清零→净面重绘）
            gnd_mode = str(cfg.get("gnd_mode") or "clean")
            rec["steps"]["gnd_mode"] = gnd_mode

            if gnd_mode in ("clean", "fusion"):
                try:
                    from gnd_clean_pipeline import run_pipeline as _gnd_run
                    from gnd_calib import load_gnd_calib

                    # seg_lane 语义（若有）
                    seg_path = os.path.join(clip_dir, "seg_lane",
                                            stem + ".png")
                    segc = None
                    if os.path.isfile(seg_path):
                        segb = cv2.imread(seg_path)
                        if segb is not None:
                            segc = np.zeros(bg.shape[:2], np.uint8)
                            PAL = {(0, 255, 0): 1, (232, 35, 244): 2,
                                   (153, 153, 190): 3}
                            for bgr, k in PAL.items():
                                mm = np.all(
                                    np.abs(segb.astype(int)
                                           - np.array(bgr)) < 30, axis=2)
                                segc[mm] = k

                    cg = load_gnd_calib(batch, clip_id)
                    if cg is None:
                        raise RuntimeError("无标定")

                    # 三步净面流程
                    _r = _gnd_run(bg, cg, cfg,
                                  objects=real_objs,
                                  semantic=segc,
                                  verbose=False,
                                  clip_dir=clip_dir, stem=stem)
                    clean = _r["clean_base"]
                    out = _r["final"]
                    mask = _r["erase_mask"]
                    rec["steps"]["erase_px"] = _r["stats"]["erase_px"]
                    rec["steps"]["paint_px"] = _r["stats"]["paint_px"]
                    rec["steps"]["clean"] = "ok"

                    # SAVE-CLEAN-IMG: 保存净面底图 + 掩码（供 UI 预览）
                    try:
                        _cdir = os.path.join(out_root, "clean")
                        os.makedirs(_cdir, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(_cdir, stem + "_clean.jpg"),
                            clean, [cv2.IMWRITE_JPEG_QUALITY, 93])
                        cv2.imwrite(
                            os.path.join(_cdir, stem + "_mask.png"),
                            mask)
                        rec["steps"]["clean_saved"] = True
                    except Exception as _se:
                        print("    净面图保存失败:", str(_se)[:50])
                        rec["steps"]["clean_saved"] = False
                except Exception as _e:
                    print("    GND clean 失败，回退旧路径:", str(_e)[:60])
                    lanes = detect_lanes(bg, cam)
                    clean = erase_lanes(bg, cam, lanes)
                    out, paint = draw_lanes(clean, cam, cfg)
                    mask = np.zeros(bg.shape[:2], np.uint8)
                    rec["steps"]["fusion"] = "fallback"
            else:
                lanes = detect_lanes(bg, cam)
                rec["steps"]["detect"] = len(lanes)
                clean = erase_lanes(bg, cam, lanes)
                rec["steps"]["erase_lanes"] = len(lanes)
                out, paint = draw_lanes(clean, cam, cfg)
                rec["steps"]["paint_px"] = int((paint > 0).sum())
                mask = np.zeros(bg.shape[:2], np.uint8)

        # 保存
        sp_path = os.path.join(fd, stem + "_sim.jpg")
        op_path = os.path.join(fd, stem + "_orig.jpg")
        cv2.imwrite(sp_path, out, [cv2.IMWRITE_JPEG_QUALITY, 93])
        cv2.imwrite(op_path, bg, [cv2.IMWRITE_JPEG_QUALITY, 92])

        rec["sim"] = os.path.basename(sp_path)
        rec["orig"] = os.path.basename(op_path)

        # GND-NO-3DANN: GND 无物体标注，写简化帧记录
        if module == "gnd":
            ann = {
                "frame": stem, "batch": batch, "clip": clip_id,
                "image": os.path.basename(sp_path),
                "source_image": os.path.basename(op_path),
                "width": w0, "height": h0, "module": "gnd",
                "camera": "front",
                "objects": [],
                "ignore_regions": [],
                "meta": {"n_real": 0, "n_synth": 0,
                         "training_ready": True,
                         "gnd_params": dict(cfg)},
            }
        else:
            ann = build_frame_annotation(
                stem, batch, clip_id, cam, w0, h0,
                real_objs, synth_list, removed_list, module)

        # CORNERS-WIRED: 每个物体加 8 角点像素坐标（仅 obj）
        if module == "obj":
            from annot_3d import box_corners_pixel
            for o in ann.get("objects") or []:
                b3 = o.get("box3d")
                if not b3:
                    continue
                try:
                    o["corners"] = box_corners_pixel(b3, cam)
                except Exception:
                    o["corners"] = None

        # TRAIN-ANNOT-WIRED: 补齐训练字段（ignore/occlusion/truncated/verified）
        try:
            from annot_train import enhance_frame, export_ignore_mask
            # ERASE-MASK-REF: 用 inpaint_objects 返回的 mask
            _em = None
            try:
                _em = mask   # 597 行: clean, mask = inpaint_objects(...)
            except NameError:
                _em = None
            ann = enhance_frame(ann, erase_mask=_em, w=w0, h=h0)
            # 写 ignore 掩码
            _igdir = os.path.join(out_root, "ignore")
            os.makedirs(_igdir, exist_ok=True)
            _igm = export_ignore_mask(ann, w0, h0)
            cv2.imwrite(os.path.join(_igdir, stem + ".png"), _igm)
        except Exception as _e:
            print("    训练标注增强失败:", str(_e)[:80])
        write_frame(out_root, ann)
        clip_frames_ann.append(ann)
        rec["n_objects"] = len(ann["objects"])
        manifest.append(rec)

        if (i + 1) % 5 == 0:
            print("  [%d/%d] %s" % (i + 1, len(frames), stem))

    # 打包 MP4
    mp4 = os.path.join(out_root, "clip.mp4")
    try:
        fs = sorted(glob.glob(os.path.join(fd, "*_sim.jpg")))
        if fs:
            first = cv2.imread(fs[0])
            h, w = first.shape[:2]
            cmd = ["ffmpeg", "-y", "-framerate", "6",
                   "-pattern_type", "glob", "-i",
                   os.path.join(fd, "*_sim.jpg"),
                   "-vf", "scale=%d:%d" % (w, h),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print("  ffmpeg 失败:", (r.stderr or "")[-200:])
    except Exception as e:
        print("  打包失败:", str(e)[:100])

    # clip 级 3D manifest
    write_clip(out_root, clip_frames_ann, batch, clip_id, module, cfg)

    with open(os.path.join(out_root, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"batch": batch, "clip": clip_id, "module": module,
                   "config": cfg, "n_frames": len(manifest),
                   "frames": manifest}, f, ensure_ascii=False, indent=1)

    # 对比图
    make_sheet(fd, out_root)

    return len(manifest)


def make_sheet(fd, out_root):
    fs = sorted(glob.glob(os.path.join(fd, "*_sim.jpg")))[:6]
    if not fs:
        return
    TW, TH = 400, 225
    rows = []
    for p in fs:
        s = cv2.imread(p)
        o = cv2.imread(p.replace("_sim.jpg", "_orig.jpg"))
        if s is None or o is None:
            continue
        s = cv2.resize(s, (TW, TH), interpolation=cv2.INTER_AREA)
        o = cv2.resize(o, (TW, TH), interpolation=cv2.INTER_AREA)
        rows.append(np.hstack([o, np.full((TH, 3, 3), 60, np.uint8), s]))
    if not rows:
        return
    sheet = np.vstack(rows)
    cv2.imwrite(os.path.join(out_root, "contact_sheet.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--module", default="obj", choices=["obj", "gnd"])
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}")
    a = ap.parse_args()

    cfg = json.loads(a.config) if a.config else {}
    os.makedirs(a.out, exist_ok=True)

    print("=== %s / %s [%s] ===" % (a.batch, a.clip, a.module))
    print("配置:", cfg)
    n = process_clip(a.batch, a.clip, a.module, cfg, a.out, a.frames)
    print("完成 %d 帧 → %s" % (n, a.out))


if __name__ == "__main__":
    main()
