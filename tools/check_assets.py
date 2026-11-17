#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_assets.py — 素材自检：形状、类别名、尺寸是否合规

为什么需要：批量替换用「严格同形状匹配」（圆对圆、三角对三角、矩形对矩形），
形状标错的素材会被静默跳过，导致以为上传成功了却大量不出图。
本工具在上传后立刻给出核对清单。

检查四件事：
  1) 文件名 → 类别名能否解析（speed_limit_60 这类）
  2) 图案实际形状 vs 该类别应有的形状是否一致
  3) 图案是否可读、是否带透明通道（带 alpha 抠图更干净）
  4) 尺寸是否过小（<128px 缩放到远处会糊）

用法：
  python3 check_assets.py 素材目录/
  python3 check_assets.py 素材目录/ --json 结果.json
返回码：0=全部通过，1=有问题（清单见输出）
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

_MODULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules")
if os.path.isdir(_MODULES):
    sys.path.insert(0, _MODULES)
try:
    import tsr as tsr_mod
except Exception:
    tsr_mod = None

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def sign_mask(img):
    """抠出图案本体：有 alpha 用 alpha，否则按四角底色做差。"""
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[:, :, 3]
        m = (a > 8).astype(np.uint8) * 255
        if m.sum() > 0:
            return m
    bgr = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    corners = np.array([bgr[0, 0], bgr[0, w - 1], bgr[h - 1, 0], bgr[h - 1, w - 1]],
                       dtype=np.float32)
    bg = corners.mean(axis=0)
    diff = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
    m = (diff > 40).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)


def detect_shape(img):
    """判定图案形状：circle / triangle / rect / unknown。"""
    m = sign_mask(img)
    if m.sum() < 100:
        return "unknown", 0.0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "unknown", 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    _, _, bw, bh = cv2.boundingRect(c)
    if bw < 8 or bh < 8:
        return "unknown", 0.0
    fill = area / float(bw * bh)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    n = len(approx)
    aspect = bw / float(bh)

    if n == 3 or (0.35 <= fill <= 0.62 and 0.75 <= aspect <= 1.35):
        return "triangle", fill
    if fill >= 0.68 and 0.78 <= aspect <= 1.28:
        return "circle", fill
    if fill >= 0.85:
        return "rect", fill
    return "rect", fill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="素材目录")
    ap.add_argument("--json", default="", help="把结果写到 JSON")
    ap.add_argument("--min-size", type=int, default=128, help="最小边长（默认 128）")
    a = ap.parse_args()

    if not os.path.isdir(a.dir):
        print("目录不存在: %s" % a.dir)
        return 2

    files = sorted(f for f in os.listdir(a.dir) if f.lower().endswith(EXTS))
    if not files:
        print("目录里没有图片: %s" % a.dir)
        return 2

    rows = []
    bad = 0
    for fn in files:
        p = os.path.join(a.dir, fn)
        item = {"file": fn, "problems": []}
        stem = os.path.splitext(fn)[0]
        cat = tsr_mod.category_of(stem) if tsr_mod else ""
        item["class_name"] = tsr_mod.normalize_class_name(stem) if tsr_mod else stem
        item["category"] = cat

        if not cat:
            item["problems"].append("文件名无法识别大类")
            bad += 1

        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            item["problems"].append("图像无法读取")
            bad += 1
            rows.append(item)
            continue

        h, w = img.shape[:2]
        item["size"] = [w, h]
        item["has_alpha"] = bool(img.ndim == 3 and img.shape[2] == 4)
        if min(w, h) < a.min_size:
            item["problems"].append("尺寸过小 %dx%d（建议 ≥%d）" % (w, h, a.min_size))
            bad += 1
        if not item["has_alpha"]:
            item["problems"].append("无透明通道（靠底色抠图，边缘可能不干净）")

        shape, fill = detect_shape(img)
        item["shape"] = shape
        item["fill"] = round(float(fill), 3)
        expect = tsr_mod.shape_of(stem) if cat and tsr_mod else ""
        item["expect_shape"] = expect
        if expect and shape != "unknown" and shape != expect:
            item["problems"].append("形状不符：实际 %s，该类应为 %s" % (shape, expect))
            bad += 1

        rows.append(item)

    # 输出
    print("素材自检：%d 个文件，%d 个有问题" % (len(files), bad))
    print("=" * 96)
    print("%-34s %-20s %-9s %-9s %-6s %s" %
          ("文件名", "类别名", "实际形状", "应为", "尺寸", "问题"))
    print("-" * 96)
    for r in rows:
        size = "%dx%d" % tuple(r["size"]) if "size" in r else "-"
        issues = "；".join(r["problems"]) if r["problems"] else "OK"
        print("%-34s %-20s %-9s %-9s %-6s %s" %
              (r["file"][:34], r.get("class_name", "")[:20], r.get("shape", "-"),
               r.get("expect_shape", "-") or "-", size, issues[:46]))

    # 汇总：按形状统计
    print("-" * 96)
    from collections import Counter
    cnt = Counter(r.get("shape", "unknown") for r in rows)
    print("形状分布：圆 %d / 三角 %d / 矩形 %d / 未知 %d"
          % (cnt.get("circle", 0), cnt.get("triangle", 0),
             cnt.get("rect", 0), cnt.get("unknown", 0)))
    cats = Counter(r.get("category", "") for r in rows if r.get("category"))
    print("大类别覆盖：%s" % "、".join("%s×%d" % (k, v) for k, v in cats.items()))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"dir": a.dir, "total": len(files), "problems": bad,
                       "items": rows}, f, indent=1, ensure_ascii=False)
        print("结果已写入 %s" % a.json)

    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
