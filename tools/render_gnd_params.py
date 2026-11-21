#!/usr/bin/env python3
"""按参数渲染 GND 标线（前端实时调参用）。

参数：
  lanes      车道数
  dash       中间虚线
  stop       停止线
  edge       边线宽度 m
  width      虚线宽度 m
  wear       磨损 0~1
  opacity    不透明度
  occlusion  遮挡裁剪
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_source import list_clips
from label_gate import list_clip_frames, load_frame_labels
from real_calib import body_to_pixel_real

try:
    from ground_masks import foreground_mask, hood_mask, road_mask
    HAS_MASKS = True
except ImportError:
    HAS_MASKS = False

DASH_ON, DASH_GAP = 6.0, 9.0     # 国标虚线 6 实 9 空


def lane_polyline(cam, y_m, w_m, x0, x1, dashed):
    """生成一条车道线的像素多边形列表。"""
    out = []
    xs = np.arange(x0, x1, 0.5)
    prev = None
    for x in xs:
        if dashed and (x % (DASH_ON + DASH_GAP)) > DASH_ON:
            prev = None
            continue
        uvl, _, ok1 = body_to_pixel_real(cam, [[x, y_m + w_m / 2, 0.0]])
        uvr, _, ok2 = body_to_pixel_real(cam, [[x, y_m - w_m / 2, 0.0]])
        if not (ok1[0] and ok2[0]):
            prev = None
            continue
        if not (np.isfinite(uvl[0]).all() and np.isfinite(uvr[0]).all()):
            prev = None
            continue
        cur = (uvl[0], uvr[0], x)
        if prev is not None and abs(cur[2] - prev[2]) <= 1.0:
            quad = np.array([prev[0], cur[0], cur[1], prev[1]], np.float32)
            if np.all(np.isfinite(quad)):
                out.append(quad)
        prev = cur
    return out


def worn_texture(shape, wear, seed):
    """磨损纹理（块状剥落）。"""
    h, w = shape
    rng = np.random.default_rng(seed)
    # 低频场
    low = rng.random((max(h // 8, 2), max(w // 8, 2))).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    low = cv2.GaussianBlur(low, (0, 0), max(min(h, w) * 0.02, 1.5))
    thr = 0.10 + 0.75 * float(wear)
    a = np.ones((h, w), np.float32)
    holes = low < thr
    a[holes] = np.clip(low[holes] / max(thr, 1e-3), 0.0, 1.0)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    batch, clip = cfg["batch"], cfg["clip"]
    pp = cfg["params"]
    out = cfg["out"]

    lanes = int(pp.get("lanes", 3))
    dash = bool(pp.get("dash", True))
    stop = bool(pp.get("stop", False))
    edge_w = float(pp.get("edge", 0.20))
    line_w = float(pp.get("width", 0.15))
    wear = float(pp.get("wear", 0.25))
    opacity = float(pp.get("opacity", 1.0))
    use_occ = bool(pp.get("occlusion", True))

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
        raise SystemExit("找不到 clip")

    frames = list_clip_frames(clip_dir)[:8]
    lab0 = load_frame_labels(clip_dir, xml_path=frames[0][1])
    if lab0 is None:
        raise SystemExit("无标定")
    cam = lab0["camera"]

    # 车道位置：总宽 3.5m × lanes
    lane_wid = 3.5
    total = lanes * lane_wid
    ys = [-total / 2 + i * lane_wid for i in range(lanes + 1)]

    print("GND: lanes=%d dash=%s stop=%s edge=%.2f line=%.2f wear=%.2f"
          % (lanes, dash, stop, edge_w, line_w, wear))

    n = 0
    for img_p, xml_n in frames:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        bg = cv2.imread(img_p)
        if bg is None:
            continue
        h, w = bg.shape[:2]

        paint = np.zeros((h, w), np.uint8)

        # 边线 + 内部线
        for i, y in enumerate(ys):
            is_edge = (i == 0 or i == len(ys) - 1)
            wdt = edge_w if is_edge else line_w
            dashed = (not is_edge) and dash
            quads = lane_polyline(cam, y, wdt, 5.0, 55.0, dashed)
            for q in quads:
                cv2.fillConvexPoly(paint, np.rint(q).astype(np.int32), 255)

        # 停止线
        if stop:
            qs = []
            for x in (24.0, 24.4):
                uvl, _, ok1 = body_to_pixel_real(
                    cam, [[x, ys[0], 0.0]])
                uvr, _, ok2 = body_to_pixel_real(
                    cam, [[x, ys[-1], 0.0]])
                if ok1[0] and ok2[0]:
                    qs.append((uvl[0], uvr[0]))
            if len(qs) == 2:
                quad = np.array([qs[0][0], qs[1][0], qs[1][1], qs[0][1]],
                                np.float32)
                if np.all(np.isfinite(quad)):
                    cv2.fillConvexPoly(paint, np.rint(quad).astype(np.int32),
                                       255)

        # 磨损
        if wear > 0.01:
            wt = worn_texture((h, w), wear, hash(stem) & 0xFFFFFFFF)
            paint = (paint.astype(np.float32) * wt).astype(np.uint8)

        # 遮挡 + 不透明度
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

        alpha = (paint.astype(np.float32) / 255.0) * (1.0 - protect) * opacity
        alpha = np.clip(alpha, 0, 1)

        out_im = (bg.astype(np.float32) * (1 - alpha[..., None])
                  + np.array([246, 246, 246], np.float32) * alpha[..., None])
        out_im = np.clip(out_im, 0, 255).astype(np.uint8)

        op = os.path.join(out, stem + "_gnd.jpg")
        cv2.imwrite(op, out_im, [cv2.IMWRITE_JPEG_QUALITY, 93])
        n += 1

    print("产出 %d 帧 → %s" % (n, out))


if __name__ == "__main__":
    main()
