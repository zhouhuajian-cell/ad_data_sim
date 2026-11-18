#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sim_report.py — 仿真产出质量自检报告

对一批合成结果做量化评估，判断哪些指标合格、哪些需要改进。

检查项：
  1) 亮度一致性   仿真图标牌区域 vs 原图同区域，亮度差应在 ±25 内
  2) 边缘过渡     标牌边缘梯度应该平滑，硬边（梯度突变）会暴露"贴纸感"
  3) 遮挡保留率   有多少帧保留了前景遮挡
  4) 类别均衡度   类别分布是否过度集中（单一类别占比 > 40% 算偏斜）
  5) 标牌尺寸分布 太小（<40px）的帧占多少
  6) 失败归因     跳过原因的分布

用法：
  python3 sim_report.py --out /data/render/tsr_1000
"""
import argparse
import json
import os
import sys
from collections import Counter


def main():
    import cv2
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="批量输出目录")
    ap.add_argument("--sample", type=int, default=200, help="抽样检查多少张")
    a = ap.parse_args()

    ann_dir = os.path.join(a.out, "annotations")
    if not os.path.isdir(ann_dir):
        print("找不到 annotations 目录：%s" % ann_dir)
        return 2

    files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".json"))
    if not files:
        print("没有标注文件")
        return 2
    print("标注总数: %d，抽样检查 %d" % (len(files), min(a.sample, len(files))))

    step = max(1, len(files) // a.sample)
    picked = files[::step][:a.sample]

    bright_diffs = []
    edge_scores = []
    occl_count = 0
    sizes = []
    classes = Counter()
    n_obj = 0

    for fn in picked:
        with open(os.path.join(ann_dir, fn), encoding="utf-8") as f:
            d = json.load(f)
        src = d.get("source_path") or ""
        comp = os.path.join(a.out, d.get("image", ""))
        if not os.path.isfile(comp) or not os.path.isfile(src):
            continue
        orig = cv2.imread(src)
        sim = cv2.imread(comp)
        if orig is None or sim is None or orig.shape != sim.shape:
            continue

        for o in d.get("objects", []):
            n_obj += 1
            classes[o["class"]] += 1
            x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
            w, h = x2 - x1, y2 - y1
            sizes.append(min(w, h))
            if w < 8 or h < 8:
                continue
            pad = 2
            # 标牌区域亮度对比（用内缩 2px 避免边缘干扰）
            ox = orig[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
            sx = sim[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
            if ox.size == 0 or sx.size == 0:
                continue
            go = cv2.cvtColor(ox, cv2.COLOR_BGR2GRAY).mean()
            gs = cv2.cvtColor(sx, cv2.COLOR_BGR2GRAY).mean()
            bright_diffs.append(float(gs - go))

            # 边缘平滑度：用「仿真图梯度 / 原图梯度」这个比值判断，
            # 而不是绝对值 —— 实测背景梯度在 50~185 之间波动，
            # 绝对值阈值根本没意义；关键是合成后有没有比原来更硬。
            def _g(im):
                g = cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
                gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
                return float(np.sqrt(gx * gx + gy * gy).mean())
            gs_ = _g(sim)
            go_ = _g(orig)
            if go_ > 1.0:
                # grad_ratio < 1 表示比原图更平滑（好）；> 1.5 表示更硬（差）
                edge_scores.append(gs_ / go_)

            if o.get("occluder_ratio", 0) > 0.01:
                occl_count += 1

    print()
    print("=" * 60)
    print("仿真质量报告")
    print("=" * 60)
    print("帧数 %d，标牌实例 %d" % (len(picked), n_obj))

    # 1) 亮度
    if bright_diffs:
        arr = np.array(bright_diffs)
        ok = (np.abs(arr) <= 25).mean() * 100
        print("\n[1] 亮度一致性")
        print("    平均偏差 %+.1f，中位数 %+.1f，标准差 %.1f"
              % (arr.mean(), np.median(arr), arr.std()))
        print("    |偏差| <= 25 的比例: %.1f%%  %s"
              % (ok, "合格" if ok >= 75 else "偏低，建议调 brightness_gain"))
    else:
        print("\n[1] 亮度一致性：无法计算")

    # 2) 边缘
    if edge_scores:
        e = np.array(edge_scores)
        print("\n[2] 边缘过渡（仿真图梯度 / 原图梯度）")
        print("    比值均值 %.2f（<1 表示比原图更平滑；>1.5 才有硬边问题）" % e.mean())
        hard = (e > 1.5).mean() * 100
        print("    比值 > 1.5 的比例: %.1f%%  %s"
              % (hard, "有硬边痕迹，建议加大 feather_ratio"
                 if hard > 30 else "边缘自然，无明显贴纸感"))

    # 3) 遮挡
    print("\n[3] 遮挡保留")
    print("    保留前景遮挡的实例: %d / %d (%.1f%%)"
          % (occl_count, n_obj, 100.0 * occl_count / max(1, n_obj)))
    print("    （小标牌 <40px 不启用遮挡检测，属正常）")

    # 4) 类别均衡
    print("\n[4] 类别均衡度")
    total = sum(classes.values())
    for c, n in classes.most_common(10):
        pct = 100.0 * n / total
        flag = "  << 偏斜" if pct > 40 else ""
        print("    %-30s %5d  %5.1f%%%s" % (c, n, pct, flag))
    top_pct = 100.0 * classes.most_common(1)[0][1] / total if total else 0
    print("    结论: %s" % ("均衡" if top_pct < 40 else "类别偏斜，建议开 --vary-class"))

    # 5) 尺寸
    if sizes:
        s = np.array(sizes)
        print("\n[5] 标牌尺寸")
        print("    中位 %d px，最小 %d，最大 %d" % (np.median(s), s.min(), s.max()))
        print("    <40px 占 %.1f%%（太小的合成效果看不出来）"
              % ((s < 40).mean() * 100))

    # 6) summary 信息
    sp = os.path.join(a.out, "summary.json")
    if os.path.isfile(sp):
        with open(sp, encoding="utf-8") as f:
            sm = json.load(f)
        print("\n[6] 运行统计")
        for k in ("ok", "skip_no_sign", "skip_small", "fail", "elapsed_sec"):
            if k in sm:
                print("    %-16s %s" % (k, sm[k]))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
