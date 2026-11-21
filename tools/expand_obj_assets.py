#!/usr/bin/env python3
"""批量扩充 OBJ 素材库：从所有数据抠图。

改进：
  - 跑全部 clip（不只前几个）
  - 每类最多 60 张
  - 质量过滤（形状/颜色/连通性）
  - 按类别 + 形状分类存储
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_guard import guard_all_mnt_ops
from label_gate import list_clip_frames, load_frame_labels

guard_all_mnt_ops()

OUT = "/data/obj_assets_real"

MAX_PER_CLASS = 60
MIN_AREA = 2000
MAX_AREA = 400000
FRAMES_PER_CLIP = 6


def quality_check(rgba):
    """质量校验。"""
    a = rgba[..., 3]
    rgb = rgba[..., :3]
    h, w = a.shape
    if h < 24 or w < 18:
        return False, "too_small"

    filled = a > 128
    fr = filled.mean()
    if fr < 0.35 or fr > 0.96:
        return False, "fill_%.2f" % fr

    cnts, _ = cv2.findContours(filled.astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "no_contour"
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 400:
        return False, "small"

    peri = cv2.arcLength(c, True)
    compact = (peri * peri) / max(area, 1.0)
    if compact > 55:
        return False, "fragmented"

    bgr = rgb[filled].astype(np.float32)
    if len(bgr) < 80:
        return False, "few_px"
    b, g, r = bgr[:, 0].mean(), bgr[:, 1].mean(), bgr[:, 2].mean()
    # 排除纯绿（树叶）
    if g > b + 15 and g > r + 15:
        return False, "greenish"

    ys = np.where(filled.any(axis=1))[0]
    if len(ys) < 10:
        return False, "thin"
    if not (filled[:h // 2].any() and filled[h // 2:].any()):
        return False, "disconnected"

    return True, "ok"


def grabcut(bg, bbox):
    h, w = bg.shape[:2]
    x0 = max(0, int(bbox[0]))
    y0 = max(0, int(bbox[1]))
    x1 = min(w - 1, int(bbox[2]))
    y1 = min(h - 1, int(bbox[3]))
    if x1 - x0 < 14 or y1 - y0 < 14:
        return None
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bg, mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, 5,
                    cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                 255, 0).astype(np.uint8)
    keep = np.zeros_like(m)
    keep[y0:y1, x0:x1] = 255
    m = cv2.bitwise_and(m, keep)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[big, cv2.CC_STAT_AREA] < MIN_AREA:
        return None
    return (lab == big).astype(np.uint8) * 255


def crop_rgba(bg, mask, pad=3):
    h, w = bg.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y0 = max(0, ys.min() - pad)
    y1 = min(h - 1, ys.max() + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(w - 1, xs.max() + pad)
    a = cv2.GaussianBlur(mask[y0:y1 + 1, x0:x1 + 1], (5, 5), 0)
    return np.dstack([bg[y0:y1 + 1, x0:x1 + 1], a])


def main():
    roots = ["/data/test_data/real_clips",
             "/data/test_data/real_clips_extra"]
    clips = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for b in sorted(glob.glob(os.path.join(r, "*"))):
            if not os.path.isdir(b):
                continue
            for c in sorted(glob.glob(os.path.join(b, "*"))):
                if os.path.isdir(os.path.join(c, "cam_st_x2f_front")):
                    clips.append((os.path.basename(b), c))

    print("clip 总数: %d" % len(clips))

    os.makedirs(OUT, exist_ok=True)
    counts = {}
    saved = 0

    for ci, (bname, cdir) in enumerate(clips):
        frames = list_clip_frames(cdir)[:FRAMES_PER_CLIP]
        for img_p, xml_n in frames:
            lab = load_frame_labels(cdir, xml_path=xml_n)
            if not lab:
                continue
            bg = cv2.imread(img_p)
            if bg is None:
                continue
            h, w = bg.shape[:2]

            for o in lab.get("objects") or []:
                if o.get("delete") or not o.get("in_front"):
                    continue
                kind = o.get("kind") or ""
                if kind not in ("car", "truck", "van", "bus", "pedestrian",
                                "rider", "three", "special"):
                    continue
                cur = counts.get(kind, 0)
                if cur >= MAX_PER_CLASS:
                    continue

                bbox = o.get("bbox2d")
                if not bbox:
                    continue
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                area = bw * bh
                if area < MIN_AREA or area > MAX_AREA:
                    continue
                if (bbox[0] < 2 or bbox[1] < 2 or
                        bbox[2] > w - 3 or bbox[3] > h - 3):
                    continue

                m = grabcut(bg, bbox)
                if m is None:
                    continue
                rgba = crop_rgba(bg, m)
                if rgba is None:
                    continue
                ok, why = quality_check(rgba)
                if not ok:
                    continue

                d = os.path.join(OUT, kind)
                os.makedirs(d, exist_ok=True)
                stem = "%s_%s_%s" % (bname[:20],
                                     os.path.splitext(os.path.basename(img_p))[0][:14],
                                     cur)
                cv2.imwrite(os.path.join(d, stem + ".png"), rgba)
                with open(os.path.join(d, stem + ".json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"src": img_p, "kind": kind,
                               "world": {"x": o["Wx"], "y": o["Wy"]},
                               "size": {"L": o["L"], "W": o["W"],
                                        "H": o["H"]}},
                              f, ensure_ascii=False)
                counts[kind] = cur + 1
                saved += 1

        if (ci + 1) % 20 == 0:
            print("  [%d/%d] 已存 %d 张  %s"
                  % (ci + 1, len(clips), saved,
                     {k: v for k, v in sorted(counts.items())}))

    print("\n=== 完成 ===")
    print("共保存 %d 张" % saved)
    for k in sorted(counts):
        print("  %-12s %d" % (k, counts[k]))

    with open(os.path.join(OUT, "_index.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
