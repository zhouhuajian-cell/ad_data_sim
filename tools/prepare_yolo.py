#!/usr/bin/env python3
"""把 done_data 的 3D 标注 XML 转成 YOLO 训练格式。

输入：
  /data/test_data/real_clips/<batch>/<clip>/cam_st_x2f_front/*.jpg
  .../label_lidar_*/*.xml

输出：
  /data/train/lane_obj/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    data.yaml

类别（按物体数量排序）：
  0 car  1 pedestrian  2 three  3 truck
  4 van  5 rider  6 special  7 bus
"""
import glob
import json
import os
import random
import shutil
import sys

import cv2

sys.path.insert(0, "/opt/ue5_synth/tools")
from label_gate import list_clip_frames, load_frame_labels

SRC = "/data/test_data/real_clips"
OUT = "/data/train/lane_obj"

CLASSES = ["car", "pedestrian", "three", "truck",
           "van", "rider", "special", "bus"]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}

# 种类映射（XML kind → 训练类别）
KIND_MAP = {
    "car": "car",
    "truck": "truck",
    "van": "van",
    "bus": "bus",
    "pedestrian": "pedestrian",
    "rider": "rider",
    "three": "three",
    "special": "special",
}

MIN_BOX = 8       # 最小框边长（像素）
MAX_BOX_RATIO = 0.95   # 框不能占满全图（过滤误标）


def clip_to_yolo(bbox, w, h):
    """bbox2d [x0,y0,x1,y1] → YOLO (cx, cy, bw, bh) 归一化。"""
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(x0, w - 1))
    y0 = max(0.0, min(y0, h - 1))
    x1 = max(0.0, min(x1, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    bw = x1 - x0
    bh = y1 - y0
    if bw < MIN_BOX or bh < MIN_BOX:
        return None
    if bw > w * MAX_BOX_RATIO or bh > h * MAX_BOX_RATIO:
        return None
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    return (cx, cy, bw / w, bh / h)


def main():
    clips = []
    for b in sorted(glob.glob(os.path.join(SRC, "*"))):
        if not os.path.isdir(b):
            continue
        for c in sorted(glob.glob(os.path.join(b, "*"))):
            if os.path.isdir(os.path.join(c, "cam_st_x2f_front")):
                clips.append((os.path.basename(b), os.path.basename(c), c))

    print("找到 clip: %d 个" % len(clips))
    if not clips:
        raise SystemExit("无数据")

    # 按 clip 划分（防泄漏）
    random.seed(42)
    random.shuffle(clips)
    n_val = max(1, int(len(clips) * 0.2))
    val_clips = set(c[2] for c in clips[:n_val])
    print("训练 clip: %d  验证 clip: %d" % (len(clips) - n_val, n_val))

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    stats = {"train": 0, "val": 0}
    n_obj = 0
    by_kind = {}

    for batch, clip_id, clip_dir in clips:
        split = "val" if clip_dir in val_clips else "train"
        frames = list_clip_frames(clip_dir)

        for img_p, xml_n in frames:
            lab = load_frame_labels(clip_dir, xml_path=xml_n)
            if lab is None:
                continue

            img = cv2.imread(img_p)
            if img is None:
                continue
            h, w = img.shape[:2]

            lines = []
            for o in lab.get("objects") or []:
                if o.get("delete") or not o.get("in_front"):
                    continue
                b = o.get("bbox2d")
                if not b:
                    continue
                k = KIND_MAP.get(o.get("kind") or "")
                if not k:
                    continue
                y = clip_to_yolo(b, w, h)
                if y is None:
                    continue
                cx, cy, bw, bh = y
                lines.append("%d %.6f %.6f %.6f %.6f"
                             % (CLS2ID[k], cx, cy, bw, bh))
                by_kind[k] = by_kind.get(k, 0) + 1
                n_obj += 1

            if not lines:
                continue

            stem = os.path.splitext(os.path.basename(img_p))[0]
            # 加 clip 前缀避免重名
            name = "%s_%s" % (clip_id.replace(".", "_"), stem.replace(".", "_"))

            ip = os.path.join(OUT, "images", split, name + ".jpg")
            lp = os.path.join(OUT, "labels", split, name + ".txt")
            shutil.copy2(img_p, ip)
            with open(lp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            stats[split] += 1

    # data.yaml
    yaml_p = os.path.join(OUT, "data.yaml")
    with open(yaml_p, "w", encoding="utf-8") as f:
        f.write("path: %s\n" % OUT)
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("nc: %d\n" % len(CLASSES))
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))

    print("\n=== 完成 ===")
    print("训练帧: %d" % stats["train"])
    print("验证帧: %d" % stats["val"])
    print("物体总数: %d" % n_obj)
    print("\n按类别:")
    for k in CLASSES:
        v = by_kind.get(k, 0)
        if v:
            print("  %-12s %d" % (k, v))
    print("\ndata.yaml:", yaml_p)

    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"train": stats["train"], "val": stats["val"],
                   "objects": n_obj, "by_kind": by_kind,
                   "classes": CLASSES}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
