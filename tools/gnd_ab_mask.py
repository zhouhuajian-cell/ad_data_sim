#!/usr/bin/env python3
"""GND 标线掩码：U-Net 分割 + 实例分割 + 可行驶区（A+B 方案）。

不调阈值，直接用已有模型和数据：
  A) U-Net (mIoU 0.70) 预测 marking
  B) obj_mask 排除车辆/障碍物，freespace 限制在路面
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/opt/ue5_synth/tools")

_MODEL = None
_PALETTE = {(0, 255, 0): 1, (232, 35, 244): 2, (153, 153, 190): 3}


def load_unet(weight_path="/data/train/gnd_runs/unet/best.pth"):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from train_gnd_seg import UNet
    ck = torch.load(weight_path, map_location="cuda")
    ncls = ck.get("ncls", 4)
    m = UNet(3, ncls, base=32)
    m.load_state_dict(ck["model"])
    m.eval()
    if torch.cuda.is_available():
        m = m.cuda()
    _MODEL = m
    print("    U-Net 加载: mIoU=%.3f ep=%s" % (
        ck.get("miou_fg", 0), ck.get("ep", "?")))
    return m


@torch.no_grad()
def predict_marking(img_bgr, size=(512, 288)):
    """A) U-Net 预测标线类别图。"""
    m = load_unet()
    h, w = img_bgr.shape[:2]
    im = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im = (im - 0.45) / 0.25
    x = torch.from_numpy(im.transpose(2, 0, 1).copy()).float()
    if torch.cuda.is_available():
        x = x.cuda()
    logits = m(x.unsqueeze(0))
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred


def load_obj_mask(clip_dir, stem, shape):
    """B) 实例分割 → 车辆/障碍物掩码。"""
    p = os.path.join(clip_dir, "obj_mask_ori_3.1.1", stem + ".png")
    h, w = shape[:2]
    if not os.path.isfile(p):
        return np.zeros((h, w), np.uint8)
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        return np.zeros((h, w), np.uint8)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    # 非 0 = 有实例
    vm = (m > 0).astype(np.uint8) * 255
    # 轻微外扩，保证车身边缘不被擦
    vm = cv2.dilate(vm, np.ones((9, 9), np.uint8))
    return vm


def load_freespace(clip_dir, stem, shape):
    """B) 可行驶区掩码。"""
    p = os.path.join(clip_dir, "freespace_mask_ori_3.0.0", stem + ".png")
    h, w = shape[:2]
    if not os.path.isfile(p):
        # 兜底：地平线以下
        m = np.zeros((h, w), np.uint8)
        m[int(h * 0.52):, :] = 255
        return m
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        m = np.zeros((h, w), np.uint8)
        m[int(h * 0.52):, :] = 255
        return m
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((m > 0).astype(np.uint8)) * 255


def load_seg_lane(clip_dir, stem, shape):
    """真值语义（可选）。"""
    p = os.path.join(clip_dir, "seg_lane", stem + ".png")
    h, w = shape[:2]
    if not os.path.isfile(p):
        return None
    seg = cv2.imread(p)
    if seg is None:
        return None
    if seg.shape[:2] != (h, w):
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((h, w), np.uint8)
    for bgr, k in _PALETTE.items():
        mm = np.all(np.abs(seg.astype(int) - np.array(bgr)) < 30, axis=2)
        out[mm] = k
    return out


def build_erase_mask(img_bgr, clip_dir, stem, use_gt_semantic=True):
    """A+B 组合：U-Net ∪ 真值语义，限定在 freespace 且排除 obj。

    返回 dict。
    """
    h, w = img_bgr.shape[:2]

    # A) U-Net 预测
    pred = predict_marking(img_bgr)
    pred_mask = np.zeros((h, w), np.uint8)
    for k in (1, 2, 3):
        pred_mask[pred == k] = 255

    # 真值语义（有就叠加，更准）
    gt = None
    if use_gt_semantic:
        gt = load_seg_lane(clip_dir, stem, img_bgr.shape)
    if gt is not None:
        gt_mask = np.zeros((h, w), np.uint8)
        for k in (1, 2, 3):
            gt_mask[gt == k] = 255
        sem_mask = cv2.bitwise_or(pred_mask, gt_mask)
    else:
        sem_mask = pred_mask

    # B) 可行驶区 & 车辆掩码
    drv = load_freespace(clip_dir, stem, img_bgr.shape)
    veh = load_obj_mask(clip_dir, stem, img_bgr.shape)

    # 合并约束
    m = cv2.bitwise_and(sem_mask, drv)
    m = cv2.bitwise_and(m, cv2.bitwise_not(veh))

    # 轻微膨胀包住旧漆边缘
    m = cv2.dilate(m, np.ones((11, 11), np.uint8))

    # 再裁一次，绝不碰车辆
    m = cv2.bitwise_and(m, cv2.bitwise_not(veh))
    m = cv2.bitwise_and(m, drv)

    return {
        "erase_mask": m,
        "pred": pred,
        "drivable": drv,
        "vehicle": veh,
        "gt": gt,
        "stats": {
            "pred_px": int(np.count_nonzero(pred_mask)),
            "gt_px": int(np.count_nonzero(gt_mask)) if gt is not None else 0,
            "drivable_px": int(np.count_nonzero(drv)),
            "vehicle_px": int(np.count_nonzero(veh)),
            "erase_px": int(np.count_nonzero(m)),
            "erase_pct": round(
                np.count_nonzero(m) / float(h * w) * 100, 2),
        },
    }
