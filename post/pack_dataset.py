# -*- coding: utf-8 -*-
# pack_dataset.py — 数据集封装（方案 1.3 / 4.5 第 7 步：转标注格式 + 切分训练验证集）
# 干什么：读逐帧真值 JSON（附录 B）→ 生成 COCO（或 YOLO）标注 + train/val 切分。
# 怎么跑：python pack_dataset.py --meta /data/render/meta --images /data/render/final \
#              --out /data/render/dataset --format coco --val-ratio 0.1 --seed 20261113
# 成功判据：instances_{train,val}.json（或 YOLO labels/ + train.txt/val.txt/data.yaml）
#          的帧数与真值帧数一致，框数与真值框一致；宽高来自真值 JSON，不写死 1920×1080。
# 图像命名契约：后处理成品是 <frame_id>.png，与 <frame_id>.json 同主文件名（方案 4.5 第 7 步）。
# ⚠️ 类别表与 batch_capture.SEMANTIC_STENCIL、queue_gen.CLASSES 同源，改一处必须同步另外两处。
import argparse
import json
import os
import random
import sys

CLASSES = ["road", "vehicle", "pedestrian", "traffic_sign", "cone", "vegetation"]
FALLBACK_SIZE = (1920, 1080)


def load_frames(meta_dir):
    """读逐帧真值；跳过 params_log 之类的非帧 JSON（没有 frame_id 键的）。"""
    frames = []
    skipped = 0
    for fn in sorted(os.listdir(meta_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(meta_dir, fn), encoding="utf-8") as f:
                fr = json.load(f)
        except Exception:
            skipped += 1
            continue
        if not isinstance(fr, dict) or not fr.get("frame_id"):
            skipped += 1
            continue
        frames.append(fr)
    if skipped:
        print("WARN 跳过 %d 个非帧 JSON（无 frame_id 或解析失败）" % skipped, file=sys.stderr)
    return frames


def frame_size(fr):
    w = fr.get("width") or FALLBACK_SIZE[0]
    h = fr.get("height") or FALLBACK_SIZE[1]
    try:
        return max(1, int(w)), max(1, int(h))
    except (TypeError, ValueError):
        return FALLBACK_SIZE


def image_name(fr):
    """成品图文件名：与 meta 同主文件名。"""
    return "%s.png" % fr["frame_id"]


def boxes_of(fr):
    """逐目标返回 (class, [x1,y1,x2,y2], 原对象)：裁到画面内、丢弃退化框。"""
    w, h = frame_size(fr)
    out = []
    for ob in (fr.get("objects") or []):
        if not isinstance(ob, dict):
            continue
        cls = ob.get("class")
        b = ob.get("bbox_2d")
        if cls not in CLASSES or not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in b]
        except (TypeError, ValueError):
            continue
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:     # 不足 1px 的框对检测训练无意义
            continue
        out.append((cls, (x1, y1, x2, y2), ob))
    return out, w, h


def split_ids(ids, val_ratio, seed):
    """确定性切分：返回 (train_ids, val_ids)；样本过少时保证训练集非空。"""
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    if len(ids) <= 1:
        return ids, []
    n_val = int(round(len(ids) * max(0.0, min(0.9, val_ratio))))
    n_val = max(1, min(len(ids) - 1, n_val)) if val_ratio > 0 else 0
    return ids[n_val:], ids[:n_val]


def to_coco(frames, images_dir, out_dir, val_ratio, seed):
    """COCO：instances_all.json + instances_train/val.json。框为 [x,y,w,h]。"""
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]
    cat_id = {c["name"]: i + 1 for i, c in enumerate(cats)}
    images, anns = [], []
    aid = 1
    missing = 0
    for fr in frames:
        name = image_name(fr)
        if images_dir and not os.path.exists(os.path.join(images_dir, name)):
            missing += 1
            print("WARN 缺成品图，跳过该帧: %s" % name, file=sys.stderr)
            continue
        w, h = frame_size(fr)
        im_id = len(images) + 1
        images.append({"id": im_id, "file_name": name, "width": w, "height": h})
        boxes, _w, _h = boxes_of(fr)
        for cls, (x1, y1, x2, y2), ob in boxes:
            anns.append({"id": aid, "image_id": im_id, "category_id": cat_id[cls],
                         "bbox": [x1, y1, x2 - x1, y2 - y1],      # COCO 是 [x,y,w,h]
                         "area": (x2 - x1) * (y2 - y1),
                         "iscrowd": 0,
                         "attributes": {"occlusion": ob.get("occlusion", 0),
                                        "distance": ob.get("distance", 0),
                                        "truncated": bool(ob.get("truncated", False)),
                                        "instance_id": ob.get("instance_id", 0)}})
            aid += 1
    if not images:
        raise SystemExit("没有可用帧（图像缺失或 meta 为空）")
    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    doc = {"info": {"description": "UE5 synthetic perception dataset",
                    "version": "1.0"},
           "licenses": [],
           "images": images, "annotations": anns, "categories": cats}
    all_path = os.path.join(ann_dir, "instances_all.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    train_ids, val_ids = split_ids([im["id"] for im in images], val_ratio, seed)
    for name, ids in (("train", set(train_ids)), ("val", set(val_ids))):
        sub = {"info": doc["info"], "licenses": [],
               "images": [im for im in images if im["id"] in ids],
               "categories": cats}
        sub["annotations"] = [an for an in anns if an["image_id"] in ids]
        with open(os.path.join(ann_dir, "instances_%s.json" % name), "w",
                  encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False, indent=1)
    print("COCO: %d 帧 / %d 框 → %s（train %d / val %d，缺图跳过 %d）" % (
        len(images), len(anns), all_path, len(train_ids), len(val_ids), missing))
    return len(images), len(anns), len(train_ids), len(val_ids)


def to_yolo(frames, images_dir, out_dir, val_ratio, seed):
    """YOLO：labels/<frame_id>.txt（cls cx cy w h 归一化）+ train.txt/val.txt/data.yaml。"""
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    usable = []
    missing = 0
    for fr in frames:
        name = image_name(fr)
        if images_dir and not os.path.exists(os.path.join(images_dir, name)):
            missing += 1
            print("WARN 缺成品图，跳过该帧: %s" % name, file=sys.stderr)
            continue
        usable.append(fr)
    if not usable:
        raise SystemExit("没有可用帧（图像缺失或 meta 为空）")
    idx = {c: i for i, c in enumerate(CLASSES)}
    train_ids, val_ids = split_ids([fr["frame_id"] for fr in usable], val_ratio, seed)
    train, val = set(train_ids), set(val_ids)
    img_root = os.path.abspath(images_dir) if images_dir else os.path.abspath(out_dir)
    n_boxes = 0
    for fr in usable:
        w, h = frame_size(fr)
        lines = []
        for cls, (x1, y1, x2, y2), _ob in boxes_of(fr)[0]:
            lines.append("%d %.6f %.6f %.6f %.6f" % (
                idx[cls], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                (x2 - x1) / w, (y2 - y1) / h))
        n_boxes += len(lines)
        with open(os.path.join(labels_dir, fr["frame_id"] + ".txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines))
    for name, ids in (("train", train), ("val", val)):
        with open(os.path.join(out_dir, "%s.txt" % name), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(os.path.join(img_root, "%s.png" % i) for i in ids)))
            if ids:
                f.write("\n")
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: %s\n" % out_dir.replace("\\", "/"))
        f.write("train: train.txt\nval: val.txt\n")
        f.write("nc: %d\nnames:\n" % len(CLASSES))
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASSES) + "\n")
    print("YOLO: %d 帧 / %d 框 → %s（train %d / val %d，缺图跳过 %d）" % (
        len(usable), n_boxes, out_dir, len(train), len(val), missing))
    return len(usable), n_boxes, len(train), len(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--images", default="", help="成品图目录（用于核对图像存在性）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["coco", "yolo"], default="coco")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    if not (0.0 <= a.val_ratio < 0.9):
        raise SystemExit("--val-ratio 必须在 [0, 0.9)：留出训练集")
    frames = load_frames(a.meta)
    if not frames:
        raise SystemExit("meta 下没有真值 JSON")
    os.makedirs(a.out, exist_ok=True)
    if a.format == "coco":
        to_coco(frames, a.images, a.out, a.val_ratio, a.seed)
    else:
        to_yolo(frames, a.images, a.out, a.val_ratio, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
