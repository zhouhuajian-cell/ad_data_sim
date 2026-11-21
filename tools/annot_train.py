#!/usr/bin/env python3
"""把仿真产出升级为可训练的标注数据。

补齐：
  1) ignore_regions   —— 擦除/不确定区域，训练时忽略（不算背景）
  2) occlusion        —— 真实物体的遮挡比例
  3) truncated        —— 是否被画面边界截断
  4) renderer/asset_id —— 合成物体溯源
  5) verified / provenance —— 真值 vs 合成
  6) 遮挡关系重建     —— 谁挡住谁
"""
import json
import os

import numpy as np


# ---------- 几何工具 ----------

def bbox_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def bbox_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def clip_bbox(b, w, h):
    return [max(0.0, b[0]), max(0.0, b[1]),
            min(float(w), b[2]), min(float(h), b[3])]


def is_truncated(b, w, h, tol=2):
    """是否被画面边界截断。"""
    return (b[0] <= tol or b[1] <= tol
            or b[2] >= w - tol or b[3] >= h - tol)


def occlusion_ratio(obj, others, depth_eps=0.5):
    """计算 obj 被 others 遮挡的面积比例。

    规则：深度更大的（更远）物体才算被遮挡。
    depth 用 box3d.x（前向距离）。
    """
    b = obj.get("bbox2d")
    if not b:
        return 0.0, []
    d = (obj.get("box3d") or {}).get("x") or 0.0
    area = bbox_area(b)
    if area <= 0:
        return 0.0, []

    covered = []
    for o in others:
        if o is obj:
            continue
        ob = o.get("bbox2d")
        if not ob:
            continue
        od = (o.get("box3d") or {}).get("x") or 0.0
        # 只算比自己近的物体
        if od >= d - depth_eps:
            continue
        inter = bbox_iou(b, ob)  # 近似重叠
        iw = max(0.0, min(b[2], ob[2]) - max(b[0], ob[0]))
        ih = max(0.0, min(b[3], ob[3]) - max(b[1], ob[1]))
        inter_area = iw * ih
        if inter_area / max(area, 1e-6) > 0.02:
            covered.append({
                "by_id": o.get("id"),
                "by_class": o.get("class"),
                "ratio": round(inter_area / max(area, 1e-6), 3),
            })

    if not covered:
        return 0.0, []
    # 粗略合并（多个遮挡者不重复累加，取并集近似）
    total = sum(c["ratio"] for c in covered)
    return min(1.0, total), covered


# ---------- 标注增强 ----------

def enhance_frame(ann, erase_mask=None, w=None, h=None):
    """给一帧标注补齐训练所需字段。"""
    w = w or ann.get("width") or 1920
    h = h or ann.get("height") or 1080
    objs = ann.get("objects") or []

    # 1) truncated
    for o in objs:
        b = o.get("bbox2d")
        if not b:
            o["truncated"] = False
            continue
        cb = clip_bbox(b, w, h)
        o["bbox2d_clipped"] = [round(v, 1) for v in cb]
        o["truncated"] = bool(is_truncated(b, w, h))
        # 出画比例
        full = bbox_area(b)
        inimg = bbox_area(cb)
        o["out_of_frame_ratio"] = round(
            1.0 - inimg / max(full, 1e-6), 3) if full > 0 else 0.0

    # 2) occlusion
    for o in objs:
        ratio, by = occlusion_ratio(o, objs)
        o["occlusion"] = round(ratio, 3)
        o["occluded_by"] = by
        o["visibility"] = round(1.0 - ratio, 3)

    # 3) 溯源
    for o in objs:
        if o.get("source") == "synthetic":
            o["verified"] = False
            o["provenance"] = "synthetic_injection"
            o.setdefault("renderer", ann.get("meta", {}).get(
                "renderer", "sprite_2p5d"))
        else:
            o["verified"] = True
            o["provenance"] = "real_label"

    # 4) ignore_regions
    ig = ann.get("ignore_regions") or []
    if erase_mask is not None and np.any(erase_mask):
        ig.append({
            "type": "erased_region",
            "reason": "background_inpainted_no_supervision",
            "bbox": _mask_bbox(erase_mask),
            "area_px": int(np.sum(erase_mask > 0)),
        })
    # 被截断/重度遮挡的目标区域也标 ignore（不参与背景学习）
    for o in objs:
        if o.get("out_of_frame_ratio", 0) > 0.5:
            ig.append({
                "type": "partial_object",
                "reason": "object_mostly_out_of_frame",
                "bbox": o.get("bbox2d"),
                "object_id": o.get("id"),
            })
    ann["ignore_regions"] = ig

    # 5) 帧级质量
    ann.setdefault("meta", {})
    ann["meta"]["n_real"] = sum(1 for o in objs if o.get("source") == "real")
    ann["meta"]["n_synth"] = sum(
        1 for o in objs if o.get("source") == "synthetic")
    ann["meta"]["training_ready"] = bool(
        ann["meta"]["n_real"] > 0 or ann["meta"]["n_synth"] > 0)
    return ann


def _mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()),
            int(xs.max()) + 1, int(ys.max()) + 1]


# ---------- 导出训练集 ----------

def to_yolo(ann, class_map):
    """标注 → YOLO 行。ignore_regions 返回用于排除。"""
    w = ann.get("width") or 1920
    h = ann.get("height") or 1080
    lines = []
    for o in ann.get("objects") or []:
        if not o.get("verified", True):
            continue  # 合成物体默认不进纯真值集
        b = o.get("bbox2d_clipped") or o.get("bbox2d")
        if not b:
            continue
        cid = class_map.get(o.get("class"))
        if cid is None:
            continue
        cx = (b[0] + b[2]) / 2.0 / w
        cy = (b[1] + b[3]) / 2.0 / h
        bw = (b[2] - b[0]) / w
        bh = (b[3] - b[1]) / h
        if bw <= 0 or bh <= 0:
            continue
        lines.append("%d %.6f %.6f %.6f %.6f" % (cid, cx, cy, bw, bh))
    return lines


def to_yolo_with_synthetic(ann, class_map):
    """标注 → YOLO 行（含合成物体，用于混合训练）。"""
    w = ann.get("width") or 1920
    h = ann.get("height") or 1080
    lines = []
    for o in ann.get("objects") or []:
        b = o.get("bbox2d_clipped") or o.get("bbox2d")
        if not b:
            continue
        cid = class_map.get(o.get("class"))
        if cid is None:
            continue
        cx = (b[0] + b[2]) / 2.0 / w
        cy = (b[1] + b[3]) / 2.0 / h
        bw = (b[2] - b[0]) / w
        bh = (b[3] - b[1]) / h
        if bw <= 0.001 or bh <= 0.001:
            continue
        lines.append("%d %.6f %.6f %.6f %.6f" % (cid, cx, cy, bw, bh))
    return lines


def export_ignore_mask(ann, w, h):
    """生成 ignore 掩码 PNG（训练时该区域不计 loss）。"""
    import cv2
    m = np.zeros((h, w), np.uint8)
    for r in ann.get("ignore_regions") or []:
        b = r.get("bbox")
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(m, (max(0, x1), max(0, y1)),
                      (min(w - 1, x2), min(h - 1, y2)), 255, -1)
    return m
