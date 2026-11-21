#!/usr/bin/env python3
"""GND 流水线 V2：用训练好的语义模型替代真值掩码。

之前：用 seg_lane 真值抹除（只适用于有真值的数据）
现在：用训练好的 U-Net 预测 → 可以用于任意图像

流程：
  1) U-Net 预测车道线语义
  2) 按预测结果抹除（LaMa）
  3) 按参数重绘新车道士
  4) 输出 + 标注
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_guard import guard_all_mnt_ops
from gnd_calib import load_gnd_calib, ground_to_pixel

guard_all_mnt_ops()

GND_ROOT = "/data/test_data/gnd_train"
MODEL = "/data/train/gnd_runs/unet/best.pth"

# 类别
CLASSES = ["background", "edge_solid", "lane_dash", "far_lane"]
PALETTE = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [232, 35, 244],
    [153, 153, 190],
], np.uint8)

_MODEL = None
_LAMA = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from train_gnd_seg import UNet
        ck = torch.load(MODEL, map_location="cpu", weights_only=False)
        m = UNet(3, ck.get("ncls", 4), 32)
        m.load_state_dict(ck["model"])
        m.eval()
        if torch.cuda.is_available():
            m = m.cuda()
        _MODEL = m
    return _MODEL


def get_lama():
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()
    return _LAMA


def predict_seg(image, size=(512, 288)):
    """U-Net 预测语义。"""
    m = get_model()
    h, w = image.shape[:2]

    im = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im = (im - 0.45) / 0.25
    x = torch.from_numpy(im.transpose(2, 0, 1).copy()).float()
    if torch.cuda.is_available():
        x = x.cuda()

    with torch.no_grad():
        logits = m(x.unsqueeze(0))
        pred = logits.argmax(1)[0].cpu().numpy()

    pred = cv2.resize(pred.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    return pred


def erase_lanes(image, seg, pad=5):
    """按语义预测抹除车道线。"""
    mask = np.zeros(image.shape[:2], np.uint8)
    for k in (1, 2, 3):     # edge/dash/far
        mask[seg == k] = 255
    if mask.sum() == 0:
        return image, mask

    mask = cv2.dilate(mask, np.ones((pad, pad), np.uint8), iterations=1)
    try:
        from PIL import Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out = get_lama()(Image.fromarray(rgb), Image.fromarray(mask))
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR), mask
    except Exception as e:
        print("    LaMa 失败:", str(e)[:50])
        return cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA), mask


def draw_new_lanes(bg, calib, cfg):
    """按参数重绘车道线（用真值标定）。"""
    lanes = int(cfg.get("lanes", 3))
    dash = cfg.get("dash", True)
    edge_w = float(cfg.get("edge", 0.20))
    line_w = float(cfg.get("width", 0.15))
    wear = float(cfg.get("wear", 0.25))
    opacity = float(cfg.get("opacity", 1.0))

    h, w = bg.shape[:2]
    paint = np.zeros((h, w), np.uint8)

    lane_wid = 3.5
    total = lanes * lane_wid
    ys = [-total / 2 + i * lane_wid for i in range(lanes + 1)]
    DASH_ON, DASH_GAP = 6.0, 9.0

    for i, y_m in enumerate(ys):
        is_edge = (i == 0 or i == len(ys) - 1)
        wdt = edge_w if is_edge else line_w
        dashed = (not is_edge) and dash

        prev = None
        for x_m in np.arange(4.0, 55.0, 0.5):
            if dashed and (x_m % (DASH_ON + DASH_GAP)) > DASH_ON:
                prev = None
                continue
            uvl = ground_to_pixel(calib, x_m, y_m + wdt / 2)
            uvr = ground_to_pixel(calib, x_m, y_m - wdt / 2)
            if uvl is None or uvr is None:
                prev = None
                continue
            cur = (uvl, uvr, x_m)
            if prev is not None and abs(cur[2] - prev[2]) <= 1.0:
                q = np.array([prev[0], cur[0], cur[1], prev[1]], np.float32)
                if np.all(np.isfinite(q)):
                    cv2.fillConvexPoly(paint, np.rint(q).astype(np.int32), 255)
            prev = cur

    if wear > 0.01:
        rng = np.random.default_rng(1234)
        low = rng.random((max(h // 8, 2), max(w // 8, 2))).astype(np.float32)
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
        low = cv2.GaussianBlur(low, (0, 0), max(min(h, w) * 0.02, 1.5))
        thr = 0.10 + 0.75 * wear
        wt = np.ones((h, w), np.float32)
        holes = low < thr
        wt[holes] = np.clip(low[holes] / max(thr, 1e-3), 0.0, 1.0)
        paint = (paint.astype(np.float32) * wt).astype(np.uint8)

    alpha = (paint.astype(np.float32) / 255.0) * opacity
    alpha = np.clip(alpha, 0, 1)
    out = (bg.astype(np.float32) * (1 - alpha[..., None])
           + np.array([246, 246, 246], np.float32) * alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), paint


def process_clip(clip_dir, out_dir, cfg, max_frames=12):
    img_dir = os.path.join(clip_dir, "img_ori")
    if not os.path.isdir(img_dir):
        return 0

    imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    imgs = imgs[:max_frames]

    # 标定
    parent = os.path.basename(os.path.dirname(clip_dir))
    veh, date = parent.rsplit("_", 1)
    clip = os.path.basename(clip_dir)
    calib = load_gnd_calib(veh, date + "/" + clip)
    has_calib = calib is not None
    if not has_calib:
        print("    !! 无标定，跳过重绘")

    fd = os.path.join(out_dir, "frames")
    ad = os.path.join(out_dir, "annotations")
    os.makedirs(fd, exist_ok=True)
    os.makedirs(ad, exist_ok=True)

    n = 0
    for i, fn in enumerate(imgs):
        stem = os.path.splitext(fn)[0]
        bg = cv2.imread(os.path.join(img_dir, fn))
        if bg is None:
            continue
        h, w = bg.shape[:2]

        # 1) 预测语义
        seg = predict_seg(bg)
        n_lane_px = int(((seg == 1) | (seg == 2) | (seg == 3)).sum())

        # 2) 抹除
        clean, emask = erase_lanes(bg, seg)

        # 3) 重绘
        if has_calib:
            sim, paint = draw_new_lanes(clean, calib, cfg)
        else:
            sim = clean
            paint = np.zeros((h, w), np.uint8)

        # 保存
        cv2.imwrite(os.path.join(fd, stem + "_sim.jpg"), sim,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        cv2.imwrite(os.path.join(fd, stem + "_orig.jpg"), bg,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        # 语义可视化
        vis = PALETTE[seg]
        cv2.imwrite(os.path.join(fd, stem + "_seg.png"), vis)

        ann = {
            "frame": stem, "module": "gnd",
            "image": "frames/%s_sim.jpg" % stem,
            "source_image": "frames/%s_orig.jpg" % stem,
            "seg_map": "frames/%s_seg.png" % stem,
            "width": w, "height": h,
            "gnd": {
                "model": "unet_gnd_v1",
                "miou": 0.7037,
                "predicted_lane_px": n_lane_px,
                "erased_px": int((emask > 0).sum()),
                "painted_px": int((paint > 0).sum()),
                "params": cfg,
            },
            "meta": {"has_calib": has_calib},
        }
        with open(os.path.join(ad, stem + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=1)
        n += 1
        if (i + 1) % 4 == 0:
            print("  [%d/%d]" % (i + 1, len(imgs)))

    return n


def make_sheet(fd, out_dir):
    fs = sorted(glob.glob(os.path.join(fd, "*_sim.jpg")))[:4]
    if not fs:
        return
    rows = []
    TW = 400
    for p in fs:
        stem = os.path.basename(p)[:-8]
        s = cv2.imread(p)
        o = cv2.imread(os.path.join(fd, stem + "_orig.jpg"))
        g = cv2.imread(os.path.join(fd, stem + "_seg.png"))
        if s is None or o is None:
            continue
        TH = int(TW * s.shape[0] / s.shape[1])
        tiles = [cv2.resize(x, (TW, TH)) for x in (o, s, g) if x is not None]
        if len(tiles) < 3:
            continue
        row = np.hstack([tiles[0], np.full((TH, 3, 3), 40, np.uint8),
                         tiles[1], np.full((TH, 3, 3), 40, np.uint8),
                         tiles[2]])
        rows.append(row)
    if rows:
        cv2.imwrite(os.path.join(out_dir, "contact_sheet.jpg"),
                    np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/render/gnd_model")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=2)
    ap.add_argument("--config",
                    default='{"lanes":3,"dash":true,"edge":0.20,'
                            '"width":0.15,"wear":0.25,"opacity":1.0}')
    a = ap.parse_args()

    cfg = json.loads(a.config)
    os.makedirs(a.out, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(GND_ROOT, "*", "*")))
    clips = [c for c in clips if os.path.isdir(os.path.join(c, "img_ori"))]
    print("GND clip: %d" % len(clips))

    total = 0
    for c in clips[:a.max_clips]:
        print("\n=== %s ===" % os.path.relpath(c, GND_ROOT))
        n = process_clip(c, a.out, cfg, a.frames)
        print("  %d 帧" % n)
        total += n

    make_sheet(os.path.join(a.out, "frames"), a.out)
    print("\n共 %d 帧 → %s" % (total, a.out))


if __name__ == "__main__":
    main()
