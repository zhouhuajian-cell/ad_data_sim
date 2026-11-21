#!/usr/bin/env python3
"""训练完成后验证模型：对比旧 YOLO 和新模型在真值上的表现。

用法：
  python compare_det.py --new /opt/ad_mining/lane_obj_v1.pt
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from label_gate import list_clip_frames, load_frame_labels


def iou(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    aa = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    bb = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return inter / (aa + bb - inter)


# 新模型类别 → 真值 kind 映射
NEW2GT = {
    "car": {"car"},
    "truck": {"truck"},
    "van": {"van"},
    "bus": {"bus"},
    "pedestrian": {"pedestrian"},
    "rider": {"rider"},
    "three": {"three"},
    "special": {"special"},
}

# COCO → 真值 kind
COCO2GT = {
    "car": {"car"},
    "truck": {"truck"},
    "bus": {"bus"},
    "motorcycle": {"rider", "three"},
    "bicycle": {"rider"},
}


def eval_model(model_path, samples, is_coco=False):
    """在真值上算 P/R。"""
    from ultralytics import YOLO
    m = YOLO(model_path)
    names = m.names

    tp = fp = fn = 0
    per_class = {}

    for img_p, gt_objs in samples:
        bg = cv2.imread(img_p)
        if bg is None:
            continue
        r = m.predict(bg, conf=0.25, verbose=False)[0]

        preds = []
        for b in r.boxes:
            c = names[int(b.cls[0])]
            allowed = (COCO2GT if is_coco else NEW2GT).get(c)
            if not allowed:
                continue
            xy = [float(v) for v in b.xyxy[0].cpu().numpy()]
            preds.append({"gt_classes": allowed, "bbox": xy,
                          "conf": float(b.conf[0])})

        preds.sort(key=lambda p: -p["conf"])

        used_gt = [False] * len(gt_objs)
        for p in preds:
            best_i = -1
            best_v = 0.3        # IoU 门槛
            for gi, g in enumerate(gt_objs):
                if used_gt[gi]:
                    continue
                if g["kind"] not in p["gt_classes"]:
                    continue
                v = iou(p["bbox"], g["bbox"])
                if v > best_v:
                    best_v = v
                    best_i = gi
            if best_i >= 0:
                used_gt[best_i] = True
                tp += 1
                k = gt_objs[best_i]["kind"]
                d = per_class.setdefault(k, {"tp": 0, "fn": 0})
                d["tp"] += 1
            else:
                fp += 1

        for gi, g in enumerate(gt_objs):
            if not used_gt[gi]:
                fn += 1
                d = per_class.setdefault(g["kind"], {"tp": 0, "fn": 0})
                d["fn"] += 1

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "per_class": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="/opt/ad_mining/lane_obj_v1.pt")
    ap.add_argument("--old", default="/opt/ad_mining/yolov8x.pt")
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()

    # 准备样本（真值）
    REAL = "/data/test_data/real_clips"
    samples = []
    clips = sorted(glob.glob(os.path.join(REAL, "*", "*")))
    clips = [c for c in clips
             if os.path.isdir(os.path.join(c, "cam_st_x2f_front"))]

    for clip in clips:
        for img_p, xml_n in list_clip_frames(clip)[:10]:
            lab = load_frame_labels(clip, xml_path=xml_n)
            if lab is None:
                continue
            gts = []
            for o in lab.get("objects") or []:
                if o.get("delete") or not o.get("in_front"):
                    continue
                b = o.get("bbox2d")
                if not b:
                    continue
                gts.append({"kind": o.get("kind") or "unknown", "bbox": b})
            if gts:
                samples.append((img_p, gts))

    samples = samples[:a.frames]
    print("验证样本: %d 帧" % len(samples))
    total_gt = sum(len(g) for _, g in samples)
    print("真值物体: %d" % total_gt)

    print("\n=== 旧模型 (COCO YOLOv8x) ===")
    r_old = eval_model(a.old, samples, is_coco=True)
    print("  P=%.3f  R=%.3f  F1=%.3f  (TP=%d FP=%d FN=%d)"
          % (r_old["precision"], r_old["recall"], r_old["f1"],
             r_old["tp"], r_old["fp"], r_old["fn"]))

    print("\n=== 新模型 (done_data 训练) ===")
    if os.path.isfile(a.new):
        r_new = eval_model(a.new, samples, is_coco=False)
        print("  P=%.3f  R=%.3f  F1=%.3f  (TP=%d FP=%d FN=%d)"
              % (r_new["precision"], r_new["recall"], r_new["f1"],
                 r_new["tp"], r_new["fp"], r_new["fn"]))

        print("\n  分类别:")
        for k in sorted(r_new["per_class"]):
            d = r_new["per_class"][k]
            t = d["tp"]
            f = d["fn"]
            rec = t / max(t + f, 1)
            print("    %-12s TP=%-4d FN=%-4d R=%.2f" % (k, t, f, rec))
    else:
        print("  模型不存在:", a.new)

    with open("/tmp/det_compare.json", "w", encoding="utf-8") as f:
        json.dump({"old": r_old,
                   "new": r_new if os.path.isfile(a.new) else None},
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
