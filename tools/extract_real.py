#!/usr/bin/env python3
"""从真实 clip 抠车 → 真实精灵图库。

原理：
  真实 clip 有 3D 标注框（含 objWx/Wy/Wz/objLength/Width/Height）
  → 投影到 2D 得到精确 bbox
  → 用「框内 vs 框外」的差异做前景分割
  → 抠出车辆像素 + alpha

关键难点：框内还有背景（路面/天空），需要分离。
做法：
  1) GrabCut（用 bbox 做初始种子）
  2) 或 用颜色聚类 + 边缘
  3) 或 用 lidar 点云投影（最准，标注里有 lidarPtsCnt）

先用 GrabCut（最稳，不依赖点云）。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_source import list_clips
from label_gate import list_clip_frames, load_frame_labels

OUT = "/data/obj_assets_real"
MIN_AREA_PX = 1200          # 太小抠不准
MAX_AREA_PX = 400000


def grabcut_object(bg, bbox, iters=5):
    """用 GrabCut 抠出 bbox 内前景。"""
    h, w = bg.shape[:2]
    x0 = max(0, int(bbox[0]))
    y0 = max(0, int(bbox[1]))
    x1 = min(w - 1, int(bbox[2]))
    y1 = min(h - 1, int(bbox[3]))

    bw = x1 - x0
    bh = y1 - y0
    if bw < 12 or bh < 12:
        return None

    rect = (x0, y0, bw, bh)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bg, mask, rect, bgd, fgd, iters,
                    cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                 255, 0).astype(np.uint8)

    # 只保留 bbox 内
    keep = np.zeros_like(m)
    keep[y0:y1, x0:x1] = 255
    m = cv2.bitwise_and(m, keep)

    # 形态学清理
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    # 取最大连通块
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    if stats[big, cv2.CC_STAT_AREA] < MIN_AREA_PX:
        return None
    m = (lab == big).astype(np.uint8) * 255

    return m


def crop_rgba(bg, mask, bbox, pad=3):
    """裁出物体 + alpha。"""
    h, w = bg.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, None
    y0 = max(0, ys.min() - pad)
    y1 = min(h - 1, ys.max() + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(w - 1, xs.max() + pad)

    rgb = bg[y0:y1 + 1, x0:x1 + 1]
    a = mask[y0:y1 + 1, x0:x1 + 1]

    # 边缘羽化
    a = cv2.GaussianBlur(a, (5, 5), 0)
    rgba = np.dstack([rgb, a])
    return rgba, (x0, y0, x1, y1)


def extract_from_clip(batch, clip_dir, max_per_class=4, frames_per_clip=8):
    """从一个 clip 抠出物体。"""
    frames = list_clip_frames(clip_dir)[:frames_per_clip]
    got = {}

    for img_p, xml_n in frames:
        lab = load_frame_labels(clip_dir, xml_path=xml_n)
        if lab is None:
            continue
        bg = cv2.imread(img_p)
        if bg is None:
            continue
        h, w = bg.shape[:2]

        for o in lab.get("objects") or []:
            if o.get("delete"):
                continue
            kind = o.get("kind") or ""
            bbox = o.get("bbox2d")
            if not bbox or not o.get("in_front"):
                continue
            if kind not in ("car", "truck", "van", "bus", "suv",
                            "pedestrian", "rider", "three", "special"):
                continue

            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            area = bw * bh
            if area < MIN_AREA_PX or area > MAX_AREA_PX:
                continue
            # 必须在画面内
            if bbox[0] < 2 or bbox[1] < 2 or bbox[2] > w - 3 or bbox[3] > h - 3:
                continue

            cur = got.setdefault(kind, [])
            if len(cur) >= max_per_class:
                continue

            mask = grabcut_object(bg, bbox)
            if mask is None:
                continue

            rgba, rect = crop_rgba(bg, mask, bbox)
            if rgba is None:
                continue

            # 质量检查：alpha 占比
            a = rgba[..., 3]
            fill = (a > 128).mean()
            if fill < 0.12 or fill > 0.92:
                continue

            cur.append({
                "rgba": rgba,
                "bbox": bbox,
                "world": {"x": o["Wx"], "y": o["Wy"], "depth": o.get("depth_m")},
                "size": {"L": o["L"], "W": o["W"], "H": o["H"]},
                "src": "%s/%s" % (os.path.basename(clip_dir),
                                  os.path.basename(img_p)),
                "fill": round(float(fill), 3),
            })

    return got


def save_assets(all_got, out_dir):
    """按类别保存。"""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for kind, items in all_got.items():
        d = os.path.join(out_dir, kind)
        os.makedirs(d, exist_ok=True)
        for i, it in enumerate(items):
            fn = "v%02d_%s.png" % (i, kind)
            cv2.imwrite(os.path.join(d, fn), it["rgba"])
            meta_item = {
                "file": fn,
                "src": it["src"],
                "world": it["world"],
                "size_m": it["size"],
            }
            with open(os.path.join(d, "v%02d.json" % i), "w",
                      encoding="utf-8") as f:
                json.dump(meta_item, f, ensure_ascii=False, indent=1)
            n += 1
        print("  %-12s %d 张" % (kind, len(items)))
    return n


def main():
    clips = list_clips()
    if not clips:
        raise SystemExit("无本地数据")

    all_got = {}
    for name, p, src in clips:
        subs = sorted(glob.glob(os.path.join(p, "*")))
        subs = [c for c in subs
                if os.path.isdir(os.path.join(c, "cam_st_x2f_front"))]
        for c in subs:
            print("### %s" % os.path.basename(c))
            got = extract_from_clip(name, c)
            for k, v in got.items():
                all_got.setdefault(k, []).extend(v)
            for k, v in got.items():
                print("    %s: %d" % (k, len(v)))

    print("\n=== 汇总 ===")
    total = 0
    for k in sorted(all_got):
        print("  %-12s %d" % (k, len(all_got[k])))
        total += len(all_got[k])

    n = save_assets(all_got, OUT)
    print("\n共保存 %d 张 → %s" % (n, OUT))


if __name__ == "__main__":
    main()
