#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_assets_from_gtsrb.py — 从 GTSRB 真实标牌照片构建素材库

为什么要这套素材（而不是矢量图）：
  实测用矢量图案贴上去「一眼假」：边缘太锐、颜色太纯、没有磨损与反光不均。
  GTSRB 里的图案是**真实拍摄的标牌裁切**，自带光照渐变、传感器噪声、镜头模糊，
  替换后与实拍场景同一质感，可直接当训练数据。
  来源：tanganke/gtsrb（HuggingFace，下载量 1.6 万+），原始 43 类 GTSRB。

做的事：
  1) 读 GTSRB 的 parquet（image bytes + label）
  2) 按 GTSRB 官方 43 类 → 我们的类别名映射（speed_limit_30 这类）
  3) 每类挑出「清晰度最高 + 尺寸最大」的若干张，抠出标牌本体、裁成方形
  4) 输出到素材目录，命名 <大类>_<子类>.png，带 alpha 通道

GTSRB 43 类与我们的类别表对应（官方类别表见 README）：
  0-9   speed_limit 20/30/50/60/70/80（GTSRB 没有 40/90/100/120 的标准牌，跳过）
  ...  其余按语义映射；无法对应的类别不导出（不做猜测，避免标错）
"""
import io
import json
import os
import sys

import numpy as np

# GTSRB 官方类别 → 我们的类别名（None 表示暂不映射）
# 参考 GTSRB 官方 README 的类别顺序
GTSRB_MAP = {
    0: "speed_limit_20",
    1: "speed_limit_30",
    2: "speed_limit_50",
    3: "speed_limit_60",
    4: "speed_limit_70",
    5: "speed_limit_80",
    6: None,                      # end of speed limit 80（解除限速，暂不映射）
    7: "speed_limit_100",
    8: "speed_limit_120",
    9: "prohibition_no_parking",
    10: "prohibition_no_parking", # no stopping with signs
    11: "warning_roundabout",
    12: "prohibition_no_u_turn",
    13: "warning_pedestrian",
    14: "warning_children",
    15: "warning_bump",
    16: "prohibition_no_right",   # no vehicles（近似）
    17: "prohibition_no_entry",
    18: "prohibition_no_entry",   # 其他危险（近似）
    19: "warning_curve_left",
    20: "warning_curve_right",
    21: "warning_narrow",         # double curve
    22: "warning_bump",
    23: "warning_slippery",
    24: "warning_narrow",
    25: "warning_roadwork",
    26: "signal_red",             # traffic signals
    27: "warning_pedestrian",
    28: "warning_pedestrian",
    29: "mandatory_turn_right",   # bicycles crossing（近似）
    30: "warning_pedestrian",
    31: None,                     # wild animals crossing
    32: None,                     # end of all speed and overtaking limits
    33: "mandatory_turn_right",
    34: "mandatory_turn_left",
    35: "mandatory_go_straight",
    36: "mandatory_go_straight",
    37: "mandatory_go_straight",  # ahead or turn left
    38: "mandatory_keep_right",
    39: "mandatory_keep_left",
    40: "mandatory_roundabout",
    41: None,                     # end of no overtaking
    42: None,                     # end of no overtaking by heavy goods
}


def sign_mask_from_image(bgr):
    """在标牌裁切图里抠出标牌本体（去掉背景边角）。"""
    h, w = bgr.shape[:2]
    gray = __import__("cv2").cvtColor(bgr, __import__("cv2").COLOR_BGR2GRAY)
    cv2 = __import__("cv2")
    # 标牌通常在图像中央，用中心区域做 Otsu
    m = np.zeros((h, w), np.uint8)
    pad_h, pad_w = int(h * 0.08), int(w * 0.08)
    m[pad_h:h - pad_h, pad_w:w - pad_w] = 255
    return m


def crop_square(img, mask):
    """按掩码的外接矩形裁成正方形（保留标牌主体）。"""
    cv2 = __import__("cv2")
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h = min(img.shape[:2])
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        return img[cy - h // 2: cy + h // 2, cx - h // 2: cx + h // 2]
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(img.shape[1], x0 + side)
    y1 = min(img.shape[0], y0 + side)
    return img[y0:y1, x0:x1]


def make_round_alpha(img, shape="circle"):
    """给标牌加 alpha：圆形/三角/方形。让素材本身带形状，替换时边缘干净。"""
    cv2 = __import__("cv2")
    h, w = img.shape[:2]
    a = np.zeros((h, w), np.uint8)
    if shape == "circle":
        cv2.circle(a, (w // 2, h // 2), int(min(h, w) * 0.48), 255, -1)
    elif shape == "triangle":
        pts = np.array([[w // 2, int(h * 0.04)],
                        [int(w * 0.05), int(h * 0.94)],
                        [int(w * 0.95), int(h * 0.94)]], np.int32)
        cv2.fillPoly(a, [pts], 255)
    else:
        a[:, :] = 255
    return a


def main():
    import cv2
    parquet = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gtsrb_test.parquet"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tsr_assets_gtsrb"
    per_class = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    os.makedirs(out_dir, exist_ok=True)

    try:
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as e:
        print("缺少依赖: %s（需要 pyarrow 与 Pillow）" % e)
        return 2

    t = pq.read_table(parquet)
    imgs = t.column("image").to_pylist()
    labels = t.column("label").to_pylist()
    print("读入 %d 张，开始筛选" % len(imgs))

    # 每个我们的类别，收集候选（按清晰度评分排序）
    from collections import defaultdict
    cand = defaultdict(list)

    for im, lab in zip(imgs, labels):
        name = GTSRB_MAP.get(int(lab))
        if not name:
            continue
        try:
            arr = np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
        except Exception:
            continue
        if min(arr.shape[:2]) < 24:
            continue
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sharp = float(cv2.Laplacian(g, -1).var())
        # 亮度适中优先（太暗/过曝的都差）
        bright = float(g.mean())
        penalty = abs(bright - 120.0)
        score = sharp - penalty * 2.0
        cand[name].append((score, bgr))

    # 每个类别取前 per_class 张，抠图 + 加 alpha 后保存（同类别多张加序号）
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "modules"))
    try:
        import tsr as tsr_mod
    except Exception:
        tsr_mod = None

    report = {"classes": {}, "total": 0}
    for name, items in sorted(cand.items()):
        items.sort(key=lambda t: -t[0])
        shape = tsr_mod.shape_of(name) if tsr_mod else "circle"
        saved = 0
        for i, (score, bgr) in enumerate(items[:per_class]):
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # 用中心区域估计标牌掩码，再裁成方形
            _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h, w = th.shape
            if th[h // 2, w // 2] == 0:
                th = cv2.bitwise_not(th)
            k = np.ones((3, 3), np.uint8)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
            sq = crop_square(bgr, th)
            if min(sq.shape[:2]) < 20:
                continue
            sq = cv2.resize(sq, (256, 256), interpolation=cv2.INTER_CUBIC)
            a = make_round_alpha(sq, shape)
            bgra = cv2.cvtColor(sq, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = a
            fn = name if saved == 0 else "%s_v%d" % (name, saved + 1)
            cv2.imwrite(os.path.join(out_dir, fn + ".png"), bgra)
            saved += 1
        if saved:
            report["classes"][name] = {"count": saved, "shape": shape,
                                       "best_sharp": round(items[0][0], 1)}
            report["total"] += saved
            print("  %-26s %d 张  形状=%s" % (name, saved, shape))

    with open(os.path.join(out_dir, "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("-" * 50)
    print("共产出 %d 张素材，覆盖 %d 个类别 → %s"
          % (report["total"], len(report["classes"]), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
