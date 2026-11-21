#!/usr/bin/env python3
"""重建 OBJ 素材库（V2：严格质量过滤）。

质量规则（比之前严格）：
  1) 面积 > 8000 px（远光目标不用）
  2) 填充率 0.55~0.92（排除抠成路面的）
  3) 长宽比符合车辆/行人特征
  4) 形状紧凑度（周长²/面积）
  5) 底部比顶部宽（车辆特征，透视）
  6) 颜色不单调（排除纯灰路面）
"""
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_guard import guard_all_mnt_ops
from label_gate import list_clip_frames, load_frame_labels

guard_all_mnt_ops()

OUT = "/data/obj_assets_v2"
LEGACY = "/data/obj_assets_real"

MAX_PER_CLASS = 80
MIN_AREA = 8000
MAX_AREA = 500000
FRAMES_PER_CLIP = 8

# 类别 → 合理长宽比范围 (min, max)
ASPECT = {
    "car": (1.0, 5.0),
    "truck": (0.9, 6.0),
    "van": (0.9, 5.0),
    "bus": (0.8, 6.0),
    "pedestrian": (0.25, 1.2),
    "rider": (0.3, 2.0),
    "three": (0.6, 3.0),
    "special": (0.5, 5.0),
}


def grabcut(bg, bbox):
    h, w = bg.shape[:2]
    x0 = max(0, int(bbox[0]))
    y0 = max(0, int(bbox[1]))
    x1 = min(w - 1, int(bbox[2]))
    y1 = min(h - 1, int(bbox[3]))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bg, mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, 6,
                    cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                 255, 0).astype(np.uint8)
    keep = np.zeros_like(m)
    keep[y0:y1, x0:x1] = 255
    m = cv2.bitwise_and(m, keep)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8) * 255


def strict_check(rgba, kind):
    """严格质量校验。"""
    a = rgba[..., 3]
    rgb = rgba[..., :3]
    h, w = a.shape
    if h < 32 or w < 28:
        return False, "small"

    filled = a > 140
    n_px = int(filled.sum())
    if n_px < MIN_AREA * 0.5:
        return False, "px_%d" % n_px

    fill = filled.mean()
    if fill < 0.55 or fill > 0.92:
        return False, "fill_%.2f" % fill

    cnts, _ = cv2.findContours(filled.astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "no_cnt"
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    compact = (peri * peri) / max(area, 1.0)
    if compact > 42:
        return False, "compact_%.0f" % compact

    # 长宽比
    ar = w / float(max(h, 1))
    lo, hi = ASPECT.get(kind, (0.3, 6.0))
    if not (lo <= ar <= hi):
        return False, "ar_%.2f" % ar

    # 底部比顶部宽（车辆透视特征，行人除外）
    if kind not in ("pedestrian",):
        top_w = filled[:h // 3].sum(axis=1).mean()
        bot_w = filled[-h // 3:].sum(axis=1).mean()
        if bot_w < top_w * 0.85:
            return False, "inv_tri"

    # 颜色不能单调（路面灰）
    px = rgb[filled]
    if len(px) < 100:
        return False, "few"
    std = px.reshape(-1, 3).std(axis=0).mean()
    if std < 12:
        return False, "flat_%.0f" % std

    # 排除绿色（树/草）
    b, g, r = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    if g > b + 18 and g > r + 18:
        return False, "green"

    return True, "ok"


def crop(rgba_full, mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    p = 2
    y0, y1 = max(0, ys.min() - p), min(h - 1, ys.max() + p)
    x0, x1 = max(0, xs.min() - p), min(w - 1, xs.max() + p)
    a = cv2.GaussianBlur(mask[y0:y1 + 1, x0:x1 + 1], (3, 3), 0)
    return np.dstack([rgba_full[y0:y1 + 1, x0:x1 + 1], a])


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

    print("clip: %d" % len(clips))
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    counts = {}
    rejects = {}
    saved = 0

    for ci, (bname, cdir) in enumerate(clips):
        for img_p, xml_n in list_clip_frames(cdir)[:FRAMES_PER_CLIP]:
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
                if kind not in ASPECT:
                    continue
                if counts.get(kind, 0) >= MAX_PER_CLASS:
                    continue

                bbox = o.get("bbox2d")
                if not bbox:
                    continue
                bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if bw * bh < MIN_AREA or bw * bh > MAX_AREA:
                    rejects["area"] = rejects.get("area", 0) + 1
                    continue
                if (bbox[0] < 2 or bbox[1] < 2 or
                        bbox[2] > w - 3 or bbox[3] > h - 3):
                    rejects["edge"] = rejects.get("edge", 0) + 1
                    continue

                m = grabcut(bg, bbox)
                if m is None:
                    rejects["grabcut"] = rejects.get("grabcut", 0) + 1
                    continue
                rgba = crop(bg, m)
                if rgba is None:
                    continue
                ok, why = strict_check(rgba, kind)
                if not ok:
                    k2 = why.split("_")[0]
                    rejects[k2] = rejects.get(k2, 0) + 1
                    continue

                d = os.path.join(OUT, kind)
                os.makedirs(d, exist_ok=True)
                stem = "%s_%s_%d" % (
                    bname[:18],
                    os.path.splitext(os.path.basename(img_p))[0][:12],
                    counts.get(kind, 0))
                cv2.imwrite(os.path.join(d, stem + ".png"), rgba)
                with open(os.path.join(d, stem + ".json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"src": img_p, "kind": kind,
                               "world": {"x": o["Wx"], "y": o["Wy"]},
                               "size": {"L": o["L"], "W": o["W"],
                                        "H": o["H"]},
                               "quality": {"fill": round(float(
                                   (rgba[..., 3] > 140).mean()), 3)}},
                              f, ensure_ascii=False)
                counts[kind] = counts.get(kind, 0) + 1
                saved += 1

        if (ci + 1) % 10 == 0:
            print("  [%d/%d] %d 张  %s" % (ci + 1, len(clips), saved,
                                           dict(sorted(counts.items()))))

    print("\n=== 完成 ===")
    print("保存 %d 张" % saved)
    for k in sorted(counts):
        print("  %-12s %d" % (k, counts[k]))
    print("\n剔除原因:")
    for k, v in sorted(rejects.items(), key=lambda x: -x[1])[:8]:
        print("  %-12s %d" % (k, v))

    with open(os.path.join(OUT, "_index.json"), "w", encoding="utf-8") as f:
        json.dump({"counts": counts, "rejects": rejects,
                   "total": saved}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
