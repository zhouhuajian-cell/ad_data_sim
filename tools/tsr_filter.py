#!/usr/bin/env python3
"""TSR 检测三级过滤漏斗 —— 专治 YOLO 误检。

核心原则：仿真数据生产 **怕误检**（宁可漏 10 个真牌，不可换错 1 个三轮车）。

  第 1 层  硬规则：置信度 / 尺寸 / 长宽比
  第 2 层  语义互斥 + 交叉验证：
           车辆互斥（框内 >30% 是车辆 → 丢）
           Mask2Former Traffic Sign 交叉验证（<25% → 丢）
  第 3 层  时序过滤：连续 ≥3 帧出现才放行

用法：
    from tsr_filter import SignGate
    gate = SignGate()
    dets = gate.filter_frame(img_bgr, dets)          # 单帧（1+2 层）
    dets = gate.filter_sequence(all_dets, min_hits=3)  # clip（+3 层）
"""
import os

import cv2
import numpy as np

# ---------------------------------------------------------------- 配置
MIN_CONF = 0.35          # RELAX-35: 降低防误杀（靠语义拦截）
MIN_SIZE = 24            # 最小边长 px
ASPECT_LO, ASPECT_HI = 0.38, 2.60   # ASPECT-RELAX: 覆盖竖长方形牌
VEH_RATIO_MAX = 0.30     # 框内车辆占比上限（常规）
SMALL_CONF = 0.30        # RELAX-35: 小目标门槛
SMALL_VEH_MAX = 0.10     # 小目标车辆占比（极严）
SMALL_PROB_MIN = 0.08    # 小目标软概率阈值
MIN_SIDE_HARD = 12       # 极小噪点硬下限

SIGN_RATIO_MIN = 0.25    # 框内 TrafficSign 占比下限
EXPAND = 2.0             # 复核外扩倍数

_EXTRACTOR = None


def _get_mapillary():
    """复用 GND 已有的 Mask2Former（零新增显存）。"""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        import sys
        sys.path.insert(0, "/opt/ue5_synth/tools")
        from gnd_mask_4a import MapillaryMarkingExtractor
        _EXTRACTOR = MapillaryMarkingExtractor()

        # 解析 TrafficSign / Vehicle 类别 id
        id2label = {}
        try:
            cfg = _EXTRACTOR.model.config
            for i, lb in enumerate(cfg.id2label.values()
                                   if isinstance(cfg.id2label, dict)
                                   else cfg.id2label):
                id2label[i] = str(lb)
        except Exception:
            pass
        _EXTRACTOR._id2label = id2label
        _EXTRACTOR._sign_ids = [
            i for i, n in id2label.items() if "traffic sign" in n.lower()]
        _EXTRACTOR._veh_ids = [
            i for i, n in id2label.items()
            if any(k in n.lower() for k in (
                "car", "truck", "bus", "motorcycle", "bicycle",
                "vehicle", "rider", "trailer"))]
    return _EXTRACTOR


def _seg_and_prob(img_bgr):
    """SOFTMAP-V2: 一次前向，同时返回 argmax 类别图 + TrafficSign 软概率图。

    软概率图用于小目标（12~24px）兜底 —— argmax 会把这些细线吞掉。
    """
    ext = _get_mapillary()
    import torch
    import torch.nn.functional as F
    from PIL import Image

    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = ext.processor(images=pil, return_tensors="pt").to(ext.device)
    with torch.no_grad():
        out = ext.model(**inputs)

    cp = F.softmax(out.class_queries_logits, dim=-1)[..., :-1]
    mp = out.masks_queries_logits.sigmoid()
    sem = torch.einsum("bqc,bqhw->bchw", cp, mp)
    sem = F.interpolate(sem, size=(h, w), mode="bilinear",
                        align_corners=False)

    seg = sem[0].argmax(dim=0).cpu().numpy().astype(np.int32)

    sign_ids = getattr(ext, "_sign_ids", []) or []
    if sign_ids:
        prob = sem[0, sign_ids, :, :].sum(dim=0).cpu().numpy()
    else:
        prob = np.zeros((h, w), np.float32)
    return seg, prob.astype(np.float32)


def _seg_map(img_bgr):
    """兼容旧调用。"""
    seg, _ = _seg_and_prob(img_bgr)
    return seg


class SignGate:
    """三层漏斗。"""

    def __init__(self, min_conf=MIN_CONF, min_size=MIN_SIZE,
                 use_semantic=True):
        self.min_conf = min_conf
        self.min_size = min_size
        self.use_semantic = use_semantic
        self.stats = {"in": 0, "conf": 0, "size": 0, "aspect": 0,
                      "vehicle": 0, "not_sign": 0, "passed": 0}

    # ---------------- 第 1 层：硬规则 ----------------
    def _layer1(self, d):
        if float(d.get("conf", 0)) < self.min_conf:
            self.stats["conf"] += 1
            return False
        w = float(d.get("w", 0))
        h = float(d.get("h", 0))
        # TIERED: 硬下限 12px（12~24 走档位2）
        _floor = MIN_SIDE_HARD
        if w < _floor or h < _floor:
            self.stats["size"] += 1
            return False
        asp = w / (h + 1e-5)
        if not (ASPECT_LO <= asp <= ASPECT_HI):
            self.stats["aspect"] += 1
            return False
        return True

    # ---------------- 第 2 层：分级语义门控 ----------------
    def filter_frame(self, img_bgr, dets):
        """单帧过滤（第 1 层 + 第 2 层分级门控）。"""
        self.stats["in"] += len(dets)
        out = []
        for d in dets:
            if self._layer1(d):
                out.append(d)
        if not out or not self.use_semantic:
            self.stats["passed"] += len(out)
            return out

        try:
            seg, prob = _seg_and_prob(img_bgr)
            ext = _get_mapillary()
            sign_ids = ext._sign_ids
            veh_ids = ext._veh_ids
        except Exception as e:
            print("    [GATE] 语义分割失败，跳过第2层:", str(e)[:70])
            self.stats["passed"] += len(out)
            return out

        H, W = seg.shape[:2]
        keep = []
        for d in out:
            x1, y1, x2, y2 = d["bbox"]
            w = float(x2 - x1)
            h = float(y2 - y1)
            min_side = min(w, h)
            conf = float(d.get("conf", 0))

            nx1, ny1 = max(0, int(x1)), max(0, int(y1))
            nx2, ny2 = min(W, int(x2)), min(H, int(y2))
            roi_seg = seg[ny1:ny2, nx1:nx2]
            roi_prob = prob[ny1:ny2, nx1:nx2]
            if roi_seg.size == 0:
                continue

            veh_ratio = (np.isin(roi_seg, veh_ids).mean()
                         if veh_ids else 0.0)

            if min_side >= 24:
                # ---- 档位 1：常规目标 ----
                if conf < self.min_conf or veh_ratio > VEH_RATIO_MAX:
                    self.stats["vehicle" if veh_ratio > VEH_RATIO_MAX
                              else "conf"] += 1
                    continue
                # 交叉验证（外扩带上下文）
                cw, ch = w, h
                ex = cw * (EXPAND - 1) / 2.0
                ey = ch * (EXPAND - 1) / 2.0
                ix1 = max(0, int(x1 - ex))
                iy1 = max(0, int(y1 - ey))
                ix2 = min(W, int(x2 + ex))
                iy2 = min(H, int(y2 + ey))
                roi_e = seg[iy1:iy2, ix1:ix2]
                roi_pe = prob[iy1:iy2, ix1:ix2]
                if sign_ids and roi_e.size > 0:
                    _ar = float(np.isin(roi_e, sign_ids).mean())
                    # DUAL-CHANNEL: argmax 不足时看软概率
                    # （远景小牌在 argmax 里会被 Road 吞掉）
                    _sp = (float(np.percentile(roi_pe, 80))
                           if roi_pe.size > 0 else 0.0)
                    if _ar < SIGN_RATIO_MIN and _sp < 0.10:
                        self.stats["not_sign"] += 1
                        continue
                    if _ar < SIGN_RATIO_MIN:
                        self.stats["rescued_soft"] = self.stats.get(
                            "rescued_soft", 0) + 1
                    # ATTACH-METRICS: 供 trackback 高置信直通用
                    d["_veh_ratio"] = float(veh_ratio)
                    d["_sign_ratio"] = float(_ar)
                    d["_soft_prob"] = float(_sp)
                keep.append(d)
            else:
                # ---- 档位 2：远景小目标 (12~24px) ----
                yc = (y1 + y2) / 2.0
                if not (0.15 * H <= yc <= 0.65 * H):
                    self.stats["small_pos"] = self.stats.get(
                        "small_pos", 0) + 1
                    continue
                if conf < SMALL_CONF:
                    self.stats["small_conf"] = self.stats.get(
                        "small_conf", 0) + 1
                    continue
                if veh_ratio > SMALL_VEH_MAX:
                    self.stats["small_veh"] = self.stats.get(
                        "small_veh", 0) + 1
                    continue
                if roi_prob.size > 0:
                    if np.percentile(roi_prob, 80) < SMALL_PROB_MIN:
                        self.stats["small_prob"] = self.stats.get(
                            "small_prob", 0) + 1
                        continue
                self.stats["small_pass"] = self.stats.get(
                    "small_pass", 0) + 1
                d["_veh_ratio"] = float(veh_ratio)
                d["_sign_ratio"] = float(np.isin(roi_seg, sign_ids).mean()
                                         if sign_ids else 0.0)
                d["_soft_prob"] = (float(np.percentile(roi_prob, 80))
                                   if roi_prob.size else 0.0)
                keep.append(d)

        self.stats["passed"] += len(keep)
        return keep

    # ---------------- 第 3 层：时序过滤 ----------------
    @staticmethod
    def filter_sequence(frames_dets, min_hits=3, iou_thr=0.3):
        """SEQ-FIX: clip 内时序过滤，只有连续 >=min_hits 帧出现的框才放行。"""
        n = len(frames_dets)
        if n == 0:
            return frames_dets

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
            ua = (ax2 - ax1) * (ay2 - ay1) + \
                 (bx2 - bx1) * (by2 - by1) - inter
            return inter / (ua + 1e-6)

        # tracks: {"last_box":..., "last_fi":..., "hits":..., "keys": set((fi, ji))}
        tracks = []
        for fi, dets in enumerate(frames_dets):
            used = set()
            for t in tracks:
                best, bi = 0.0, -1
                if t["last_fi"] == fi - 1:
                    for j, d in enumerate(dets):
                        if j in used:
                            continue
                        v = _iou(t["last_box"], d["bbox"])
                        if v > iou_thr and v > best:
                            best, bi = v, j
                if bi >= 0:
                    t["last_box"] = dets[bi]["bbox"]
                    t["last_fi"] = fi
                    t["hits"] += 1
                    t["keys"].add((fi, bi))
                    used.add(bi)
            for j, d in enumerate(dets):
                if j in used:
                    continue
                tracks.append({"last_box": d["bbox"], "last_fi": fi,
                               "hits": 1, "keys": {(fi, j)}})

        good = set()
        for t in tracks:
            if t["hits"] >= min_hits:
                good |= t["keys"]

        out = []
        for fi, dets in enumerate(frames_dets):
            out.append([d for j, d in enumerate(dets) if (fi, j) in good])
        return out
