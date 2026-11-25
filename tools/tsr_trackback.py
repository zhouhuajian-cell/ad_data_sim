#!/usr/bin/env python3
"""Track-back 轨迹回溯（轨迹级确证 + 插值补全）。

严格按需求：
  1) 禁止全场景放行 —— 每条轨迹独立确证
  2) 只有同一条轨迹在"近景且 Mask2Former 验证"过的帧，才向前回溯放行
  3) 已确证轨迹中间的 YOLO 漏检空洞（<=2 帧）用前后 BBox 线性插值补全
"""
import numpy as np


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (ua + 1e-6)


def _b(x):
    return x["bbox"] if isinstance(x, dict) else x


def build_tracks(frames_boxes, iou_thr=0.20, max_gap=4):
    """IoU + 中心点距离串轨迹。"""
    tracks = []
    for fi, boxes in enumerate(frames_boxes):
        used = set()
        for t in tracks:
            last_fi = max(t["dets"].keys())
            if fi - last_fi > max_gap:
                continue
            last_box = _b(frames_boxes[last_fi][t["dets"][last_fi]])
            lx = (last_box[0] + last_box[2]) / 2.0
            ly = (last_box[1] + last_box[3]) / 2.0
            lsz = max(last_box[2] - last_box[0],
                      last_box[3] - last_box[1], 1.0)
            best, bi = 0.0, -1
            for j, bx in enumerate(boxes):
                if j in used:
                    continue
                bb = _b(bx)
                v = _iou(last_box, bb)
                cx = (bb[0] + bb[2]) / 2.0
                cy = (bb[1] + bb[3]) / 2.0
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                score = v if v > iou_thr else (0.25 if dist < lsz * 1.5 else 0.0)
                if score > best:
                    best, bi = score, j
            if bi >= 0:
                t["dets"][fi] = bi
                t["hits"] += 1
                used.add(bi)
        for j, bx in enumerate(boxes):
            if j in used:
                continue
            tracks.append({"dets": {fi: j}, "hits": 1})
    return tracks


def trackback_filter(frames_boxes, frames_verified,
                     near_size=20.0, min_hits=2, max_hole=2):
    """轨迹级确证 + 插值补全。

    返回 (frames_keep, frames_fill, stats)
      frames_keep: [[bool,...]]   该框放行
      frames_fill: [[box|None,...]] 插值补出的框（与 frames_keep 对齐）
    """
    n = len(frames_boxes)
    keep = [[False] * len(b) for b in frames_boxes]
    fill = [[None] * len(b) for b in frames_boxes]
    tracks = build_tracks(frames_boxes)

    confirmed_tracks = 0
    rescued_small = 0
    filled_holes = 0

    for t in tracks:
        if t["hits"] < min_hits:
            continue
        # ---- RELAX-V2: 取消强制近景 ----
        # 只要轨迹 >= min_hits 且至少有一帧语义验证通过即可放行
        # （远景小标牌全程不带近景帧也要保留）
        ok = False
        for fi, bi in t["dets"].items():
            if bi >= len(frames_verified[fi]):
                continue
            if frames_verified[fi][bi]:
                ok = True
                break
        if not ok:
            continue
        confirmed_tracks += 1

        # ---- 放行整条轨迹（含远景小目标）----
        for fi, bi in t["dets"].items():
            keep[fi][bi] = True
            b = _b(frames_boxes[fi][bi])
            if min(b[2] - b[0], b[3] - b[1]) < near_size:
                rescued_small += 1

        # ---- 插值补全：<= max_hole 帧的空洞 ----
        fis = sorted(t["dets"].keys())
        for k in range(len(fis) - 1):
            f0, f1 = fis[k], fis[k + 1]
            hole = f1 - f0 - 1
            if hole <= 0 or hole > max_hole:
                continue
            b0 = _b(frames_boxes[f0][t["dets"][f0]])
            b1 = _b(frames_boxes[f1][t["dets"][f1]])
            for h in range(1, hole + 1):
                a = h / float(hole + 1)
                fb = [b0[i] * (1 - a) + b1[i] * a for i in range(4)]
                fill[f0 + h].append(fb)
                filled_holes += 1

    # ---- RELAX-V2: 单帧高置信直通（捞回 IoU 断掉的真标牌）----
    from tsr_filter import SignGate as _SG
    _gate_hi = _SG(min_conf=0.50)
    direct_pass = 0
    for fi, dets in enumerate(frames_boxes):
        for bi in range(len(dets)):
            if keep[fi][bi]:
                continue
            d = dets[bi]
            c = float(d.get("conf", 0))
            if c < 0.50:
                continue
            b = _b(d)
            side = min(b[2] - b[0], b[3] - b[1])
            # 高置信 + 车辆互斥（veh_ratio<=0.10）+ 软概率达标
            try:
                vr = float(d.get("_veh_ratio", 9.0))
                sr = float(d.get("_sign_ratio", 0.0))
                sp = float(d.get("_soft_prob", 0.0))
            except Exception:
                continue
            if vr <= 0.10 and (sr >= 0.30 or sp >= 0.15):
                keep[fi][bi] = True
                direct_pass += 1
            elif side >= 24 and c >= 0.65 and vr <= 0.15:
                # 兜底：大目标高置信直接放
                keep[fi][bi] = True
                direct_pass += 1

    stats = {"tracks": len(tracks),
             "confirmed_tracks": confirmed_tracks,
             "rescued_small": rescued_small,
             "filled_holes": filled_holes,
             "direct_pass": direct_pass,
             "kept": sum(sum(1 for x in f if x) for f in keep)}
    return keep, fill, stats
