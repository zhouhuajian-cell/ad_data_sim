#!/usr/bin/env python3
"""GND 流水线：用 seg_lane 真值做「抹除 → 重绘」。

数据源：/data/test_data/gnd_clips/<batch>/<clip>/
  img_ori/      原图
  seg_lane/     车道线真值（绿=边线, 粉=内部虚线）
  obj_mask/     障碍物掩码（用于遮挡）

流程：
  1) 读 seg_lane 真值 → 提取车道线像素 + 分类
  2) 抹除原车道线（inpaint）
  3) 按参数重绘新车道线
  4) 输出仿真帧 + 标注
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_guard import guard_all_mnt_ops

guard_all_mnt_ops()

GND_ROOT = "/data/test_data/gnd_clips"

# seg_lane 调色板（BGR）
SEG_BG = (0, 0, 0)
SEG_PINK = (232, 35, 244)      # 内部虚线
SEG_GREEN = (0, 255, 0)        # 边线/实线

_LAMA = None


def get_lama():
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()
    return _LAMA


def read_seg(seg_path):
    """读分割掩码，返回 (mask_bg, mask_pink, mask_green)。"""
    seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        return None, None, None
    if seg.ndim == 2:
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)

    def eq(c):
        return np.all(seg == np.array(c, np.uint8), axis=2).astype(np.uint8)

    return eq(SEG_BG), eq(SEG_PINK), eq(SEG_GREEN)


def erase_lanes(bg, masks, expand_px=5):
    """用真值掩码抹除原车道线。"""
    h, w = bg.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for k in masks:
        if k is not None:
            m = np.maximum(m, k * 255)

    if m.sum() == 0:
        return bg, m

    m = cv2.dilate(m, np.ones((expand_px, expand_px), np.uint8),
                   iterations=1)
    try:
        lama = get_lama()
        from PIL import Image
        rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        out = lama(Image.fromarray(rgb), Image.fromarray(m))
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR), m
    except Exception as e:
        print("    LaMa 失败:", str(e)[:50])
        return cv2.inpaint(bg, m, 7, cv2.INPAINT_TELEA), m


def load_gnd_cam(clip_dir):
    """GND 数据的标定（从 parameter_yaml 或 hdmap 推）。"""
    # 尝试找 parameter_yaml
    for p in glob.glob(os.path.join(clip_dir, "parameter_yaml", "*", "*")):
        pass
    # 简化：用固定内参（1080p 前视，fx≈1000）
    return {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0,
            "R": np.eye(3).tolist(), "t": [0, 0, 1.5]}


def draw_lanes(bg, cam, cfg, obj_mask=None):
    """画新车道线（参数化）。"""
    lanes = int(cfg.get("lanes", 3))
    dash = cfg.get("dash", True)
    edge_w = float(cfg.get("edge", 0.20))
    line_w = float(cfg.get("width", 0.15))
    wear = float(cfg.get("wear", 0.25))
    opacity = float(cfg.get("opacity", 1.0))

    h, w = bg.shape[:2]
    paint = np.zeros((h, w), np.uint8)

    fx = cam["fx"]
    cx = cam["cx"]
    cy = cam["cy"]
    cam_h = 1.5         # 相机高度 m

    lane_wid = 3.5
    total = lanes * lane_wid
    ys = [-total / 2 + i * lane_wid for i in range(lanes + 1)]

    DASH_ON, DASH_GAP = 6.0, 9.0

    for i, y_m in enumerate(ys):
        is_edge = (i == 0 or i == len(ys) - 1)
        wdt = edge_w if is_edge else line_w
        dashed = (not is_edge) and dash

        prev = None
        for x_m in np.arange(3.0, 60.0, 0.4):
            if dashed and (x_m % (DASH_ON + DASH_GAP)) > DASH_ON:
                prev = None
                continue

            # 地面点 (x_m, y_m±w/2, 0) → 像素
            uvl, uv1 = proj(cam_h, fx, cx, cy, x_m, y_m + wdt / 2)
            uvr, uv2 = proj(cam_h, fx, cx, cy, x_m, y_m - wdt / 2)
            if uvl is None or uvr is None:
                prev = None
                continue

            cur = (uvl, uvr, x_m)
            if prev is not None and abs(cur[2] - prev[2]) <= 1.0:
                q = np.array([prev[0], cur[0], cur[1], prev[1]], np.float32)
                if np.all(np.isfinite(q)):
                    cv2.fillConvexPoly(paint, np.rint(q).astype(np.int32), 255)
            prev = cur

    # 磨损
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

    # 遮挡
    protect = np.zeros((h, w), np.float32)
    if obj_mask is not None:
        protect = np.clip(obj_mask.astype(np.float32) / 255.0, 0, 1)

    alpha = (paint.astype(np.float32) / 255.0) * (1 - protect) * opacity
    alpha = np.clip(alpha, 0, 1)
    out = (bg.astype(np.float32) * (1 - alpha[..., None])
           + np.array([246, 246, 246], np.float32) * alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), paint


def proj(cam_h, fx, cx, cy, x_m, y_m):
    """地面点 → 像素（简化针孔，相机在原点朝 +x）。"""
    if x_m < 0.5:
        return None, None
    z = -cam_h
    u = cx + fx * (y_m / x_m)
    v = cy + fx * (-z / x_m)
    if not (np.isfinite(u) and np.isfinite(v)):
        return None, None
    return (float(u), float(v)), True


def process_clip(clip_dir, out_dir, cfg, max_frames=20):
    img_dir = os.path.join(clip_dir, "img_ori")
    seg_dir = os.path.join(clip_dir, "seg_lane")
    obj_dir = os.path.join(clip_dir, "obj_mask_ori_3.1.1")

    if not os.path.isdir(img_dir):
        raise SystemExit("无 img_ori: " + clip_dir)

    imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    imgs = imgs[:max_frames]

    fd = os.path.join(out_dir, "frames")
    ad = os.path.join(out_dir, "annotations")
    os.makedirs(fd, exist_ok=True)
    os.makedirs(ad, exist_ok=True)

    cam = load_gnd_cam(clip_dir)
    frames_ann = []
    n = 0

    for i, fn in enumerate(imgs):
        stem = os.path.splitext(fn)[0]
        bg = cv2.imread(os.path.join(img_dir, fn))
        if bg is None:
            continue
        h, w = bg.shape[:2]

        # 真值掩码
        seg_p = os.path.join(seg_dir, stem + ".png")
        bg_m = pink_m = green_m = None
        n_real = 0
        if os.path.isfile(seg_p):
            bg_m, pink_m, green_m = read_seg(seg_p)
            n_real = int((pink_m.sum() if pink_m is not None else 0)
                         + (green_m.sum() if green_m is not None else 0))

        # 障碍物掩码
        obj_p = os.path.join(obj_dir, stem + ".png")
        obj_m = None
        if os.path.isfile(obj_p):
            om = cv2.imread(obj_p, cv2.IMREAD_UNCHANGED)
            if om is not None:
                obj_m = om if om.ndim == 2 else om[..., 0]

        # 1) 抹除原车道线
        clean, erase_m = erase_lanes(bg, [pink_m, green_m])

        # 2) 画新车道线
        sim, paint = draw_lanes(clean, cam, cfg, obj_m)

        # 保存
        cv2.imwrite(os.path.join(fd, stem + "_sim.jpg"), sim,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        cv2.imwrite(os.path.join(fd, stem + "_orig.jpg"), bg,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        ann = {
            "frame": stem,
            "batch": os.path.basename(os.path.dirname(clip_dir)),
            "clip": os.path.basename(clip_dir),
            "module": "gnd",
            "width": w, "height": h,
            "image": "frames/%s_sim.jpg" % stem,
            "source_image": "frames/%s_orig.jpg" % stem,
            "camera": cam,
            "gnd": {
                "n_real_lane_px": n_real,
                "n_erased_px": int((erase_m > 0).sum()),
                "n_paint_px": int((paint > 0).sum()),
                "lanes_param": cfg,
                "lane_classes": {"edge": int(green_m.sum())
                                 if green_m is not None else 0,
                                 "dash": int(pink_m.sum())
                                 if pink_m is not None else 0},
            },
            "meta": {
                "ground_truth": "seg_lane",
                "training_eligible": False,
                "training_blockers": ["camera_extrinsic_unverified"],
            },
        }
        with open(os.path.join(ad, stem + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=1)
        frames_ann.append(ann)
        n += 1

        if (i + 1) % 5 == 0:
            print("  [%d/%d] %s" % (i + 1, len(imgs), stem))

    # manifest
    with open(os.path.join(out_dir, "clip_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "version": "gnd_sim_v1",
            "batch": os.path.basename(os.path.dirname(clip_dir)),
            "clip": os.path.basename(clip_dir),
            "module": "gnd",
            "n_frames": n,
            "config": cfg,
            "frames": [a["frame"] for a in frames_ann],
            "ground_truth": "seg_lane (green=edge, pink=dash)",
            "training_eligible": False,
        }, f, ensure_ascii=False, indent=1)

    # 对比图
    make_sheet(fd, out_dir)
    return n


def make_sheet(fd, out_dir):
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
        rows.append(np.hstack([
            cv2.resize(o, (TW, TH)), np.full((TH, 3, 3), 60, np.uint8),
            cv2.resize(s, (TW, TH))]))
    if rows:
        cv2.imwrite(os.path.join(out_dir, "contact_sheet.jpg"),
                    np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    ap.add_argument("--out", default="/data/render/gnd_pipe")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--config", default="{}")
    a = ap.parse_args()

    cfg = json.loads(a.config) if a.config else {}
    os.makedirs(a.out, exist_ok=True)

    if a.clip:
        clips = [a.clip]
    else:
        clips = sorted(glob.glob(os.path.join(GND_ROOT, "*", "*")))
        clips = [c for c in clips if os.path.isdir(os.path.join(c, "img_ori"))]

    print("GND clip: %d 个" % len(clips))
    total = 0
    for c in clips[:1]:
        print("\n=== %s ===" % os.path.relpath(c, GND_ROOT))
        n = process_clip(c, a.out, cfg, a.frames)
        print("  完成 %d 帧" % n)
        total += n

    print("\n共 %d 帧 → %s" % (total, a.out))


if __name__ == "__main__":
    main()
