#!/usr/bin/env python3
"""GND 车道线语义分割训练（U-Net）。

数据：/data/train/gnd_seg/
  训练 924 帧 / 验证 234 帧
  4 类：background, edge_solid, lane_dash, far_lane

关键：
  - 类别极度不平衡（背景 99.15%）
  - 用 Dice + CE 组合损失
  - 类别权重按逆频率

输出：/data/train/gnd_runs/unet/best.pth
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2


# ---------- 模型 ----------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=3, ncls=4, base=32):
        super().__init__()
        b = base
        self.d1 = DoubleConv(cin, b)
        self.d2 = DoubleConv(b, b * 2)
        self.d3 = DoubleConv(b * 2, b * 4)
        self.d4 = DoubleConv(b * 4, b * 8)
        self.d5 = DoubleConv(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.c4 = DoubleConv(b * 16, b * 8)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.c3 = DoubleConv(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.c2 = DoubleConv(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.c1 = DoubleConv(b * 2, b)
        self.out = nn.Conv2d(b, ncls, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        d5 = self.d5(self.pool(d4))
        x = self.c4(torch.cat([self.u4(d5), d4], 1))
        x = self.c3(torch.cat([self.u3(x), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)


# ---------- 数据 ----------
class SegDS(Dataset):
    def __init__(self, root, split, size=(512, 288)):
        self.root = root
        self.split = split
        self.size = size
        d = os.path.join(root, "images", split)
        self.names = sorted(
            os.path.splitext(f)[0] for f in os.listdir(d)
            if f.endswith(".jpg"))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        ip = os.path.join(self.root, "images", self.split, n + ".jpg")
        lp = os.path.join(self.root, "labels", self.split, n + ".png")

        im = cv2.imread(ip)
        im = cv2.resize(im, self.size, interpolation=cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im = (im - 0.45) / 0.25

        m = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, self.size, interpolation=cv2.INTER_NEAREST)

        x = torch.from_numpy(im.transpose(2, 0, 1).copy()).float()
        y = torch.from_numpy(m.astype(np.int64))
        return x, y


# ---------- 损失 ----------
def dice_loss(logits, target, ncls, weight=None):
    probs = F.softmax(logits, 1)
    oh = F.one_hot(target, ncls).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * oh).sum(dims)
    card = probs.sum(dims) + oh.sum(dims)
    d = (2 * inter + 1e-6) / (card + 1e-6)
    if weight is not None:
        d = d * weight
    return 1 - d.mean()


class Loss(nn.Module):
    def __init__(self, ncls, cls_weights=None):
        super().__init__()
        self.ncls = ncls
        self.w = (torch.tensor(cls_weights, dtype=torch.float32)
                  if cls_weights else None)

    def forward(self, logits, target):
        w = self.w.to(logits.device) if self.w is not None else None
        ce = F.cross_entropy(logits, target, weight=w)
        dc = dice_loss(logits, target, self.ncls, w)
        return 0.5 * ce + 0.5 * dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/train/gnd_seg")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="/data/train/gnd_runs/unet")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", dev)

    meta = json.load(open(os.path.join(a.data, "classes.json"),
                          encoding="utf-8"))
    ncls = len(meta["classes"])
    px = meta.get("pixel_stats") or {}
    tot = sum(px.values()) or 1
    # 逆频率权重，夹在 [0.2, 30]
    cw = []
    for i in range(ncls):
        f = px.get(str(i), 1) / tot
        w = 1.0 / max(f, 1e-6)
        cw.append(w)
    mx = max(cw)
    cw = [max(0.2, min(30.0, w / mx * ncls)) for w in cw]
    print("类别权重:", [round(v, 2) for v in cw])

    tr = SegDS(a.data, "train")
    va = SegDS(a.data, "val")
    print("训练 %d / 验证 %d" % (len(tr), len(va)))

    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=4,
                    pin_memory=True)

    model = UNet(3, ncls, 32).to(dev)
    crit = Loss(ncls, cw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best = -1.0
    hist = []
    t0 = time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        s_loss = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            s_loss += loss.item()
        sched.step()

        # 验证
        model.eval()
        inter = np.zeros(ncls)
        union = np.zeros(ncls)
        vloss = 0.0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                vloss += crit(logits, y).item()
                pr = logits.argmax(1)
                for k in range(ncls):
                    inter[k] += ((pr == k) & (y == k)).sum().item()
                    union[k] += ((pr == k) | (y == k)).sum().item()

        ious = inter / np.maximum(union, 1)
        miou = float(np.nanmean(ious[1:]))     # 不含背景
        fg_iou = float(np.nanmean(ious[1:]))

        print("ep %2d/%-2d  loss=%.4f  val=%.4f  mIoU(前景)=%.4f  "
              "iours=%s  %.0fs"
              % (ep, a.epochs, s_loss / max(len(tl), 1),
                 vloss / max(len(vl), 1), fg_iou,
                 [round(float(v), 3) for v in ious], time.time() - t0))

        hist.append({"ep": ep, "loss": s_loss / max(len(tl), 1),
                     "val": vloss / max(len(vl), 1),
                     "miou_fg": fg_iou,
                     "iou": [float(v) for v in ious]})

        if fg_iou > best:
            best = fg_iou
            torch.save({"model": model.state_dict(),
                        "ncls": ncls, "classes": meta["classes"],
                        "miou_fg": fg_iou, "ep": ep},
                       os.path.join(a.out, "best.pth"))
            print("   ★ 新最佳 %.4f" % fg_iou)

    with open(os.path.join(a.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)

    print("\n=== 完成 %.1f 分钟，最佳 mIoU(前景)=%.4f ==="
          % ((time.time() - t0) / 60, best))
    print("模型:", os.path.join(a.out, "best.pth"))


if __name__ == "__main__":
    main()
