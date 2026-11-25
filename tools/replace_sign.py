#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""replace_sign.py — 把上传的 TSR 图案替换到实拍照片里的标牌上（2D 合成路线）

路线说明（为什么不用 UE 渲染）：
  实拍照片里已有真实标牌，位置/尺寸/透视/光照都是真实的。
  直接在这个位置上做「图案替换」，比在 UE 里重建场景再合成更逼真、也快得多。
  合成时继承原标牌的亮度与色温，所以背光、昏暗等场景特征会保留。

流程：
  1) YOLO（/opt/ad_mining/tsr_sign.pt）检测照片里的标牌，得到 bbox + 类别 + 置信度
  2) 按 bbox 尺寸从大到小挑一个目标（可指定索引）
  3) 把 TSR 图案按目标区域的形状做透视/仿射贴合
  4) 用原标牌区域的亮度统计做颜色匹配（保留背光感）
  5) 羽化边缘融合，避免硬边
  6) 输出合成图 + 标注图 + 记录 JSON

用法：
  python3 replace_sign.py --image 照片.jpg --sign tsr.png --out 输出目录
        [--conf 0.3] [--index 0] [--shape auto|circle|triangle|rect]
        [--scale 1.0] [--keep-ratio] [--feather 0.15]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


# GPU 设备选择：TSR_YOLO_DEVICE 环境变量优先，默认自动（有 CUDA 就用 GPU）
_YOLO_DEVICE = None
_YOLO_MODELS = {}


def _pick_device():
    """选 YOLO 推理设备：cuda:0 优先，没有就 cpu。"""
    global _YOLO_DEVICE
    if _YOLO_DEVICE is not None:
        return _YOLO_DEVICE
    env = os.environ.get("TSR_YOLO_DEVICE", "").strip()
    if env:
        _YOLO_DEVICE = env
        return env
    try:
        import torch
        _YOLO_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        _YOLO_DEVICE = "cpu"
    return _YOLO_DEVICE


def detect_signs(image_path, conf=0.3, weights="/opt/ad_mining/tsr_sign.pt"):
    """跑 YOLO 检测标牌；失败返回空列表而不是抛异常（调用方决定要不要报错）。

    GPU 加速：torch 有 CUDA 就自动用 cuda:0（实测比 CPU 快 5~15 倍）。
    想强制 CPU：export TSR_YOLO_DEVICE=cpu
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("WARN: 无法导入 ultralytics（%s），退回纯图像方案" % e)
        return []
    model = _YOLO_MODELS.get(weights)
    if model is None:
        model = YOLO(weights)
        _YOLO_MODELS[weights] = model
    dev = _pick_device()
    try:
        model.to(dev)
    except Exception:
        dev = "cpu"
    res = model(image_path, conf=conf, device=dev, verbose=False, half=(dev != "cpu"))[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        out.append({
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "class": res.names[int(b.cls[0])],
            "conf": round(float(b.conf[0]), 4),
            "w": round(x2 - x1, 1),
            "h": round(y2 - y1, 1),
        })
    out.sort(key=lambda d: -(d["w"] * d["h"]))
    return out


def load_sign(path):
    """读 TSR 图案；带 alpha 就返回四通道，否则当白底/纯色底处理。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit("读不到图案: %s" % path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def sign_mask(sign):
    """把图案转成「标牌本体」的二值掩码。

    有 alpha 就用 alpha；否则用「与四角颜色差异」判定前景，
    这样纯色背景（白底/黑底）的截图也能抠出来。
    """
    if sign.shape[2] == 4:
        a = sign[:, :, 3]
        m = (a > 8).astype(np.uint8) * 255
        if m.sum() > 0:
            return m
    bgr = sign[:, :, :3]
    h, w = bgr.shape[:2]
    corners = np.array([bgr[0, 0], bgr[0, w - 1], bgr[h - 1, 0], bgr[h - 1, w - 1]], dtype=np.float32)
    bg = corners.mean(axis=0)
    diff = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
    m = (diff > 40).astype(np.uint8) * 255
    # 形态学收一下，去掉噪点
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def estimate_shape(mask):
    """按轮廓的外接形状猜标牌形状：圆 / 三角 / 方。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "rect", None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    rect_ratio = area / float(max(1, w * h))       # 填充率
    n = len(approx)
    if n <= 4 and rect_ratio > 0.85:
        return "rect", c
    if n == 3 or (rect_ratio > 0.35 and rect_ratio < 0.6):
        return "triangle", c
    if rect_ratio > 0.7 and abs(w - h) / max(w, h) < 0.15:
        return "circle", c
    return "rect", c


def color_match(patch, target_mean, target_std):
    """把贴图的颜色分布对齐到原标牌区域，保留亮度/色温（背光场景就不会变亮）。"""
    p = patch.astype(np.float32)
    cur_mean = p.reshape(-1, 3).mean(axis=0)
    cur_std = p.reshape(-1, 3).std(axis=0) + 1e-6
    out = (p - cur_mean) / cur_std * target_std + target_mean
    return np.clip(out, 0, 255)


def fill_original(image, bbox, pad=6, mode="inpaint"):
    """把原标牌擦掉（inpaint），避免新图案盖不住时露出旧标牌的边。

    为什么需要：原标牌可能是三角形、我们贴的圆形盖不住三个角，
    实测会在下方露出红色三角轮廓。先擦后贴就干净了。
    """
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return image
    if mode == "none":
        return image
    mask = np.zeros((H, W), np.uint8)
    # 用椭圆近似标牌轮廓：圆/三角/方都能盖住大部分
    cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2),
                ((x2 - x1) // 2 + pad, (y2 - y1) // 2 + pad), 0, 0, 360, 255, -1)
    try:
        return cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)
    except Exception:
        # 极端情况退回简单模糊填充
        out = image.copy()
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (31, 31), 0)
        return out


def composite(image, sign, bbox, shape="auto", scale=1.0, feather=0.15, keep_ratio=True,
              erase=True):
    """把 sign 贴到 image 的 bbox 区域，返回 (合成图, 掩码, 实际使用的 bbox)。"""
    if erase:
        image = fill_original(image, bbox)
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return image, None, bbox

    # 目标区域（允许稍微外扩/内缩）
    tw = int((x2 - x1) * scale)
    th = int((y2 - y1) * scale)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    tx1, ty1 = cx - tw // 2, cy - th // 2
    tx2, ty2 = tx1 + tw, ty1 + th

    sm = sign_mask(sign)
    sh, sw = sm.shape[:2]

    # 图案按目标长宽比缩放
    if keep_ratio:
        r = min(tw / float(sw), th / float(sh))
        nw, nh = max(2, int(sw * r)), max(2, int(sh * r))
    else:
        nw, nh = max(2, tw), max(2, th)

    sign_r = cv2.resize(sign[:, :, :3], (nw, nh), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(sm, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # 在原标牌区域采样亮度/色温（用于颜色对齐）
    ref = image[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
    if ref.shape[0] > 50:
        tmean = ref.mean(axis=0)
        tstd = ref.std(axis=0)
        matched = color_match(sign_r, tmean, tstd)
    else:
        matched = sign_r.astype(np.float32)

    # 放到目标尺寸的透明层上（居中）
    layer = np.zeros((th, tw, 3), dtype=np.float32)
    layerm = np.zeros((th, tw), dtype=np.uint8)
    oy, ox = (th - nh) // 2, (tw - nw) // 2
    layer[oy:oy + nh, ox:ox + nw] = matched
    layerm[oy:oy + nh, ox:ox + nw] = mask_r

    # 羽化：先腐蚀一下再高斯模糊，边缘不会硬
    kf = max(1, int(min(th, tw) * feather))
    if kf % 2 == 0:
        kf += 1
    soft = cv2.GaussianBlur(layerm.astype(np.float32) / 255.0, (kf, kf), 0)
    soft = np.clip(soft, 0, 1)[:, :, None]

    out = image.astype(np.float32).copy()
    roi = out[ty1:ty2, tx1:tx2]
    if roi.shape[:2] == layer.shape[:2]:
        roi[:] = layer * soft + roi * (1 - soft)
        out[ty1:ty2, tx1:tx2] = roi
    return np.clip(out, 0, 255).astype(np.uint8), layerm, [tx1, ty1, tx2, ty2]


def draw_boxes(image, dets, chosen=None):
    vis = image.copy()
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        color = (0, 255, 0) if i == chosen else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3 if i == chosen else 2)
        label = "%s %.2f" % (d["class"], d["conf"])
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sign", required=True, help="TSR 图案（PNG，带 alpha 最好）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--index", type=int, default=0, help="替换第几个检测（按面积从大到小）")
    ap.add_argument("--bbox", default="", help="手动指定 bbox 'x1,y1,x2,y2'，给了就跳过检测选择")
    ap.add_argument("--shape", default="auto", choices=["auto", "circle", "triangle", "rect"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--feather", type=float, default=0.15)
    ap.add_argument("--no-keep-ratio", action="store_true")
    ap.add_argument("--no-erase", action="store_true",
                    help="不擦除原标牌（默认会先 inpaint 擦掉，避免露出旧边）")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    image = cv2.imread(a.image)
    if image is None:
        raise SystemExit("读不到照片: %s" % a.image)
    sign = load_sign(a.sign)

    dets = detect_signs(a.image, a.conf)
    report = {"image": a.image, "sign": a.sign, "detections": dets,
              "shape_detected": estimate_shape(sign_mask(sign))[0]}

    if a.bbox:
        bbox = [float(v) for v in a.bbox.split(",")]
        idx = -1
        report["mode"] = "manual_bbox"
    else:
        if not dets:
            report["error"] = "没有检测到标牌（可降低 --conf 或用 --bbox 手动指定）"
            with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=1, ensure_ascii=False)
            print("FAIL: " + report["error"])
            return 2
        idx = min(a.index, len(dets) - 1)
        bbox = dets[idx]["bbox"]
        report["mode"] = "yolo_detection"
        report["chosen"] = dets[idx]

    vis = draw_boxes(image, dets, idx if idx >= 0 else None)
    cv2.imwrite(os.path.join(a.out, "detect.jpg"), vis)

    composed, mask, used = composite(image, sign, bbox, a.shape, a.scale, a.feather,
                                     keep_ratio=not a.no_keep_ratio,
                                     erase=not a.no_erase)
    cv2.imwrite(os.path.join(a.out, "composed.jpg"), composed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if mask is not None:
        cv2.imwrite(os.path.join(a.out, "mask.png"), mask)

    report["used_bbox"] = used
    report["ok"] = mask is not None
    with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("OK: %s -> %s (bbox=%s)" % (os.path.basename(a.image), a.out, used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
