#!/usr/bin/env python3
"""用真实素材库（/data/obj_assets）渲染 OBJ 合成图。

替换原来的程序化占位图，改用 asset_lib.pick() 选素材：
  - 多视角：front / rear / side_left
  - 日夜：day / night
  - 真实尺寸：从 meta.json 读
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from asset_lib import OBJ_ROOT, pick, refresh, scan
from data_source import list_clips
from label_gate import list_clip_frames, load_frame_labels
from real_calib import body_to_pixel_real

try:
    from ground_masks import foreground_mask, hood_mask, road_mask
    HAS_MASKS = True
except ImportError:
    HAS_MASKS = False

OUT = "/data/render/verify_batch"


def load_sprite(cls, view="front", light="day"):
    """从素材库取精灵图。"""
    path, meta = pick(cls, view, light, OBJ_ROOT)
    if path is None or not os.path.isfile(path):
        return None, None
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None, None
    return im, meta


def project_quad(cam, x_m, y_m, w_m, h_m):
    """地面点 + 尺寸 → 像素四边形。"""
    quad = []
    for yy in (y_m + w_m / 2, y_m - w_m / 2):
        uv, z, ok = body_to_pixel_real(cam, [[x_m, yy, 0.0]])
        if not ok[0] or not np.isfinite(uv[0]).all():
            return None
        quad.append(uv[0])

    uvl, _, _ = body_to_pixel_real(cam, [[x_m, y_m + 0.5, 0.0]])
    uvr, _, _ = body_to_pixel_real(cam, [[x_m, y_m - 0.5, 0.0]])
    if not (uvl[0] is not None and uvr[0] is not None):
        return None
    px_per_m = abs(uvl[0][0] - uvr[0][0])
    top_dy = h_m * px_per_m

    left, right = quad[0], quad[1]
    return np.array([(left[0], left[1] - top_dy),
                     (right[0], right[1] - top_dy),
                     right, left], np.float32)


def compose(bg, sprite, quad, protect):
    """把精灵图贴到四边形区域。"""
    h, w = bg.shape[:2]
    sh, sw = sprite.shape[:2]
    src = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]],
                   np.float32)
    Pm = cv2.getPerspectiveTransform(src, quad)

    a = sprite[..., 3].astype(np.float32) / 255.0
    rgb = sprite[..., :3].astype(np.float32)
    prem = rgb * a[..., None]

    a_w = cv2.warpPerspective(a, Pm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    p_w = cv2.warpPerspective(prem, Pm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    a_w = np.clip(a_w, 0, 1)
    straight = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
    straight[a_w <= 1e-4] = 0

    keep = a_w * (1.0 - protect)
    out = (bg.astype(np.float32) * (1 - keep[..., None])
           + straight * keep[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def process_clip(batch, clip_dir, nframes=8):
    """处理一个 clip。"""
    clip_id = os.path.basename(clip_dir)
    frames = list_clip_frames(clip_dir)[:nframes]
    if not frames:
        return 0

    # 取第一帧的标定
    lab0 = load_frame_labels(clip_dir, xml_path=frames[0][1])
    if lab0 is None:
        return 0
    cam = lab0["camera"]
    is_night = "night" in batch.lower()
    light = "night" if is_night else "day"

    # 固定世界坐标（跨帧稳定）
    POSITIONS = [
        ("cone", 9.0, -1.2),
        ("cone", 9.0, 0.4),
        ("car", 22.0, -3.4),
        ("truck", 34.0, 4.2),
    ]

    od = os.path.join(OUT, batch)
    os.makedirs(od, exist_ok=True)
    n = 0

    for img_p, xml_n in frames:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        bg = cv2.imread(img_p)
        if bg is None:
            continue
        h, w = bg.shape[:2]

        # 前景保护
        if HAS_MASKS:
            try:
                road = road_mask(bg)
                fg = foreground_mask(bg, road)
                hood = hood_mask(bg, frac=0.18)
                protect = np.clip(np.maximum(fg, hood), 0, 1)
            except Exception:
                protect = np.zeros((h, w), np.float32)
        else:
            protect = np.zeros((h, w), np.float32)

        out = bg.copy()
        objs = []
        # 远到近
        for cls, x_m, y_m in sorted(POSITIONS, key=lambda p: -p[1]):
            sp, meta = load_sprite(cls, "front", light)
            if sp is None:
                continue
            sz = (meta or {}).get("size_m") or {}
            w_m = float(sz.get("width", 1.0))
            h_m = float(sz.get("height", 1.5))

            quad = project_quad(cam, x_m, y_m, w_m, h_m)
            if quad is None:
                continue
            if quad[:, 0].max() < 0 or quad[:, 0].min() > w:
                continue
            if quad[:, 1].max() < 0 or quad[:, 1].min() > h:
                continue

            out = compose(out, sp, quad, protect)
            xs, ys = quad[:, 0], quad[:, 1]
            objs.append({"class": cls,
                         "bbox": [round(float(xs.min()), 1),
                                  round(float(ys.min()), 1),
                                  round(float(xs.max()), 1),
                                  round(float(ys.max()), 1)],
                         "world": {"x_m": x_m, "y_m": y_m},
                         "size_m": [w_m, h_m],
                         "sprite": "%s_%s" % ("front", light)})

        op = os.path.join(od, stem + "_composed.jpg")
        cv2.imwrite(op, out, [cv2.IMWRITE_JPEG_QUALITY, 93])

        # 标注
        ad = os.path.join(od, "annotations")
        os.makedirs(ad, exist_ok=True)
        with open(os.path.join(ad, stem + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump({"image": os.path.basename(op),
                       "batch": batch, "clip": clip_id, "stem": stem,
                       "width": w, "height": h, "module": "OBJ",
                       "lighting": light,
                       "calibration": {"source": "xml",
                                       "fx": cam["fx"], "fy": cam["fy"]},
                       "objects": objs,
                       "training_eligible": False,
                       "training_blockers": ["appearance_unverified",
                                             "multi_camera_missing"]},
                      f, ensure_ascii=False, indent=1)
        n += 1

    return n


def main():
    refresh()
    lib = scan(OBJ_ROOT)
    print("素材库:", {k: len(v) for k, v in lib.items()})

    clips = list_clips()
    total = 0
    for name, p, src in clips:
        subs = sorted(glob.glob(os.path.join(p, "*")))
        subs = [c for c in subs
                if os.path.isdir(os.path.join(c, "cam_st_x2f_front"))]
        print("\n### %s [%s]" % (name, src))
        for c in subs:
            n = process_clip(name, c, 8)
            print("  %-26s %d 帧" % (os.path.basename(c), n))
            total += n

    print("\n共 %d 张 OBJ 合成图 → %s" % (total, OUT))


if __name__ == "__main__":
    main()
