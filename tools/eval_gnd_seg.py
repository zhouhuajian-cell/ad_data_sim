#!/usr/bin/env python3
"""语义训练完成后：验证 + 可视化 + 接入流水线。

用法：
  python eval_gnd_seg.py --model /data/train/gnd_runs/unet/best.pth
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, "/opt/ue5_synth/tools")
from train_gnd_seg import UNet, SegDS

CLASSES = ["background", "edge_solid", "lane_dash", "far_lane"]
COLORS = np.array([
    [0, 0, 0],
    [0, 255, 0],       # 边线 绿
    [232, 35, 244],    # 虚线 粉
    [153, 153, 190],   # 远处 灰紫
], np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/train/gnd_runs/unet/best.pth")
    ap.add_argument("--data", default="/data/train/gnd_seg")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="/tmp/gnd_seg_eval")
    a = ap.parse_args()

    if not os.path.isfile(a.model):
        raise SystemExit("模型不存在: " + a.model)

    ck = torch.load(a.model, map_location="cpu")
    ncls = ck.get("ncls", 4)
    model = UNet(3, ncls, 32)
    model.load_state_dict(ck["model"])
    model.eval()
    print("模型: %s  (ep=%s  mIoU=%.4f)"
          % (a.model, ck.get("ep"), ck.get("miou_fg", 0)))

    da = SegDS(a.data, "val", (512, 288))
    if len(da) == 0:
        raise SystemExit("验证集为空")

    os.makedirs(a.out, exist_ok=True)
    rows = []
    ious = np.zeros(ncls)
    un = np.zeros(ncls)

    for i in range(min(a.n, len(da))):
        x, y = da[i]
        with torch.no_grad():
            logits = model(x.unsqueeze(0))
            pr = logits.argmax(1)[0].numpy()
        gt = y.numpy()

        for k in range(ncls):
            ious[k] += ((pr == k) & (gt == k)).sum()
            un[k] += ((pr == k) | (gt == k)).sum()

        # 可视化：原图 | GT | 预测
        im = x.numpy().transpose(1, 2, 0)
        im = (im * 0.25 + 0.45)
        im = np.clip(im * 255, 0, 255).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        gt_v = COLORS[gt]
        pr_v = COLORS[pr]

        row = np.hstack([im, np.full((im.shape[0], 3, 3), 60, np.uint8),
                         gt_v, np.full((im.shape[0], 3, 3), 60, np.uint8),
                         pr_v])
        rows.append(row)

    if rows:
        sheet = np.vstack(rows)
        p = os.path.join(a.out, "eval_sheet.jpg")
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("对比图:", p)

    iou = ious / np.maximum(un, 1)
    print("\n=== 验证集 IoU ===")
    for i, c in enumerate(CLASSES):
        print("  %-14s %.4f" % (c, iou[i]))
    print("  mIoU(前景)   %.4f" % float(np.nanmean(iou[1:])))

    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump({"iou": [float(v) for v in iou],
                   "classes": CLASSES,
                   "miou_fg": float(np.nanmean(iou[1:]))}, f, indent=1)


if __name__ == "__main__":
    main()
