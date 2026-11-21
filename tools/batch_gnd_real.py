"""为真实 clip 批量生成 GND 标线合成图（与 OBJ 同帧对齐）。

输出结构与 verify_batch 一致：
  /data/render/verify_batch/<batch>/<stem>_gnd.jpg
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_source import list_clips
from label_gate import list_clip_frames, load_frame_labels
from real_calib import body_to_pixel_real, load_cam_real

try:
    from ground_masks import road_mask, foreground_mask, hood_mask
    HAS_MASKS = True
except ImportError:
    HAS_MASKS = False


# 车道线规格：(横向位置 m, 线宽 m, 类型)
LANE_SPECS = [
    (-5.25, 0.20, "edge"),
    (-1.75, 0.15, "dash"),
    (1.75, 0.15, "dash"),
    (5.25, 0.20, "edge"),
]

DASH_ON, DASH_OFF = 6.0, 9.0     # 虚线 6m 实 9m 空（相位固定）


def draw_lane(bg, cam, y_m, w_m, kind, x0=5.0, x1=55.0, color=(245, 245, 245)):
    """在路面上画一条车道线，返回 mask。"""
    h, w = bg.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    xs = np.arange(x0, x1, 0.5)
    polys = []
    for x in xs:
        if kind == "dash" and (x % (DASH_ON + DASH_OFF)) > DASH_ON:
            continue
        uv_l, z1, ok1 = body_to_pixel_real(cam, [[x, y_m + w_m / 2, 0.0]])
        uv_r, z2, ok2 = body_to_pixel_real(cam, [[x, y_m - w_m / 2, 0.0]])
        if not (ok1[0] and ok2[0]):
            continue
        if not (np.isfinite(uv_l[0]).all() and np.isfinite(uv_r[0]).all()):
            continue
        polys.append((x, uv_l[0], uv_r[0]))

    if len(polys) < 2:
        return mask

    # 连续段合并成多边形
    for i in range(len(polys) - 1):
        x_a, l_a, r_a = polys[i]
        x_b, l_b, r_b = polys[i + 1]
        if abs(x_b - x_a) > 1.0:      # 断开（虚线间隙）
            continue
        quad = np.array([l_a, l_b, r_b, r_a], np.float32)
        if not np.all(np.isfinite(quad)):
            continue
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
    return mask


def process_frame(img_path, cam, out_path):
    bg = cv2.imread(img_path)
    if bg is None:
        return None
    h, w = bg.shape[:2]

    paint = np.zeros((h, w), np.uint8)
    for y_m, w_m, kind in LANE_SPECS:
        m = draw_lane(bg, cam, y_m, w_m, kind)
        paint = np.maximum(paint, m)

    # 路面前景保护
    if HAS_MASKS:
        road = road_mask(bg)
        fg = foreground_mask(bg, road)
        hood = hood_mask(bg, frac=0.18)
        protect = np.clip(np.maximum(fg, hood), 0, 1)
        # 只在"没被挡"的地方上漆
        keep = (paint.astype(np.float32) / 255.0) * (1.0 - protect)
    else:
        keep = paint.astype(np.float32) / 255.0

    out = (bg.astype(np.float32) * (1 - keep[..., None])
           + np.array([245, 245, 245], np.float32) * keep[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)

    cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"out": out_path, "kept": float(keep.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", type=int, default=8)
    ap.add_argument("--out", default="/data/render/verify_batch")
    a = ap.parse_args()

    clips = list_clips()
    total = 0
    for name, p, src in clips:
        subs = sorted(glob.glob(os.path.join(p, "*")))
        subs = [c for c in subs if os.path.isdir(os.path.join(c, "cam_st_x2f_front"))]
        print("\n### %s" % name)
        for c in subs:
            clip_id = os.path.basename(c)
            cam = load_cam_real(name, clip_id)
            if cam is None:
                print("  %s 无标定" % clip_id)
                continue
            frames = list_clip_frames(c)[:a.per_clip]
            od = os.path.join(a.out, name)
            os.makedirs(od, exist_ok=True)
            n = 0
            for img_p, xml_n in frames:
                stem = os.path.splitext(os.path.basename(img_p))[0]
                op = os.path.join(od, stem + "_gnd.jpg")
                try:
                    r = process_frame(img_p, cam, op)
                    if r:
                        n += 1
                except Exception as e:
                    print("    %s 失败: %s" % (stem, e))
            print("  %s: %d 帧" % (clip_id, n))
            total += n

    print("\n共 %d 张 GND 合成图 → %s" % (total, a.out))


if __name__ == "__main__":
    main()
