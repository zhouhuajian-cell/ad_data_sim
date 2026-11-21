#!/usr/bin/env python3
"""按参数渲染 OBJ 合成图（前端实时调参用）。

参数：
  count      物体数量
  cls        类别
  dist_min   最近距离 m
  dist_max   最远距离 m
  lat        横向范围 ±m
  scale      尺寸倍率
  shadow     阴影强度
  occlusion  遮挡裁剪
  light      auto/day/night
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from asset_lib import OBJ_ROOT, pick, refresh
from data_source import list_clips
from label_gate import list_clip_frames, load_frame_labels
from real_calib import body_to_pixel_real

REAL_ASSETS = "/data/obj_assets_real"

try:
    from ground_masks import foreground_mask, hood_mask, road_mask
    HAS_MASKS = True
except ImportError:
    HAS_MASKS = False


def find_sprite(cls, light, scale=1.0):
    """优先用真实抠图素材，回退程序化精灵。"""
    # 真实素材
    if os.path.isdir(os.path.join(REAL_ASSETS, cls)):
        fs = sorted(glob.glob(os.path.join(REAL_ASSETS, cls, "*.png")))
        if fs:
            im = cv2.imread(fs[0], cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                return im, {"size_m": _real_size(cls), "source": "real_extract"}

    path, meta = pick(cls, "front", light, OBJ_ROOT)
    if path and os.path.isfile(path):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            return im, meta
    return None, None


def _real_size(cls):
    """真实素材的默认尺寸（米）。"""
    return {
        "car": {"width": 1.80, "height": 1.50, "length": 4.30},
        "suv": {"width": 1.90, "height": 1.75, "length": 4.50},
        "truck": {"width": 2.00, "height": 2.30, "length": 5.20},
        "van": {"width": 2.00, "height": 2.20, "length": 5.00},
        "pedestrian": {"width": 0.55, "height": 1.70, "length": 0.50},
        "rider": {"width": 0.70, "height": 1.75, "length": 1.00},
        "three": {"width": 1.30, "height": 1.60, "length": 1.80},
    }.get(cls, {"width": 1.0, "height": 1.5, "length": 2.0})


def project_quad(cam, x, y, w_m, h_m):
    quad = []
    for yy in (y + w_m / 2, y - w_m / 2):
        uv, z, ok = body_to_pixel_real(cam, [[x, yy, 0.0]])
        if not ok[0] or not np.isfinite(uv[0]).all():
            return None
        quad.append(uv[0])
    uvl, _, _ = body_to_pixel_real(cam, [[x, y + 0.5, 0.0]])
    uvr, _, _ = body_to_pixel_real(cam, [[x, y - 0.5, 0.0]])
    px_per_m = abs(uvl[0][0] - uvr[0][0])
    top_dy = h_m * px_per_m
    left, right = quad[0], quad[1]
    return np.array([(left[0], left[1] - top_dy),
                     (right[0], right[1] - top_dy),
                     right, left], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    batch, clip = cfg["batch"], cfg["clip"]
    pp = cfg["params"]
    out = cfg["out"]

    refresh()

    count = int(pp.get("count", 3))
    cls = str(pp.get("cls", "cone"))
    dmin = float(pp.get("dist_min", 8))
    dmax = float(pp.get("dist_max", 28))
    lat = float(pp.get("lat", 3.0))
    scl = float(pp.get("scale", 1.0))
    shadow = float(pp.get("shadow", 0.35))
    use_occ = bool(pp.get("occlusion", True))
    light_mode = str(pp.get("light", "auto"))

    # 定位 clip
    clips = list_clips()
    clip_dir = None
    for name, p, src in clips:
        if name != batch:
            continue
        cand = os.path.join(p, clip)
        if os.path.isdir(os.path.join(cand, "cam_st_x2f_front")):
            clip_dir = cand
            break
    if not clip_dir:
        raise SystemExit("找不到 clip: %s/%s" % (batch, clip))

    frames = list_clip_frames(clip_dir)[:8]
    if not frames:
        raise SystemExit("无帧")

    # 标定
    lab0 = load_frame_labels(clip_dir, xml_path=frames[0][1])
    if lab0 is None:
        raise SystemExit("无标定")
    cam = lab0["camera"]

    is_night = "night" in batch.lower()
    light = ("night" if (light_mode == "auto" and is_night)
             else ("auto" if light_mode == "auto" else light_mode))
    if light == "auto":
        light = "day"

    sp, meta = find_sprite(cls, light)
    if sp is None:
        raise SystemExit("无素材: " + cls)
    sz = (meta or {}).get("size_m") or _real_size(cls)
    w_m = float(sz.get("width", 1.0)) * scl
    h_m = float(sz.get("height", 1.5)) * scl

    print("渲染: cls=%s count=%d dist=[%.0f,%.0f] lat=±%.1f scale=%.2f sprite=%s"
          % (cls, count, dmin, dmax, lat, scl,
             (meta or {}).get("source", "?")))

    n = 0
    for img_p, xml_n in frames:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        bg = cv2.imread(img_p)
        if bg is None:
            continue
        h, w = bg.shape[:2]

        if HAS_MASKS and use_occ:
            try:
                road = road_mask(bg)
                fg = foreground_mask(bg, road)
                hood = hood_mask(bg, frac=0.18)
                protect = np.clip(np.maximum(fg, hood), 0, 1)
            except Exception:
                protect = np.zeros((h, w), np.float32)
        else:
            protect = np.zeros((h, w), np.float32)

        out_im = bg.copy()

        # 均匀分布在 [dmin, dmax] × [-lat, lat]
        import random
        rng = random.Random(hash(stem) & 0xFFFFFFFF)
        positions = []
        for i in range(count):
            x = dmin + (dmax - dmin) * (i + 0.5) / max(count, 1)
            y = rng.uniform(-lat, lat)
            positions.append((x, y))
        positions.sort(key=lambda p: -p[0])

        for x, y in positions:
            quad = project_quad(cam, x, y, w_m, h_m)
            if quad is None:
                continue
            if quad[:, 0].max() < 0 or quad[:, 0].min() > w:
                continue
            if quad[:, 1].max() < 0 or quad[:, 1].min() > h:
                continue

            sh, sw = sp.shape[:2]
            src = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1],
                            [0, sh - 1]], np.float32)
            Pm = cv2.getPerspectiveTransform(src, quad)

            a = sp[..., 3].astype(np.float32) / 255.0
            rgb = sp[..., :3].astype(np.float32)
            prem = rgb * a[..., None]

            a_w = cv2.warpPerspective(a, Pm, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
            p_w = cv2.warpPerspective(prem, Pm, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
            a_w = np.clip(a_w, 0, 1)
            straight = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
            straight[a_w <= 1e-4] = 0

            # 阴影
            if shadow > 0.01:
                base = quad.astype(np.int32)
                cx = int((base[2][0] + base[3][0]) / 2)
                cy = int((base[2][1] + base[3][1]) / 2)
                rw = max(int(abs(base[1][0] - base[0][0]) * 0.55), 4)
                rh = max(int(rw * 0.25), 2)
                shm = np.zeros((h, w), np.float32)
                cv2.ellipse(shm, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)
                shm = cv2.GaussianBlur(shm, (15, 15), 0) * shadow
                out_im = (out_im.astype(np.float32)
                          * (1 - shm[..., None])).astype(np.uint8)

            keep = a_w * (1.0 - protect)
            out_im = (out_im.astype(np.float32) * (1 - keep[..., None])
                      + straight * keep[..., None])
            out_im = np.clip(out_im, 0, 255).astype(np.uint8)

        op = os.path.join(out, stem + "_obj.jpg")
        cv2.imwrite(op, out_im, [cv2.IMWRITE_JPEG_QUALITY, 93])
        n += 1

    print("产出 %d 帧 → %s" % (n, out))


if __name__ == "__main__":
    main()
