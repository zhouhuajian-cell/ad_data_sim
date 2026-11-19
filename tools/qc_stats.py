#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qc_stats.py — QC 原因分布统计，一眼看出瓶颈在哪一环

为什么需要这个：
  单看「通过率 56%」只知道一半样本不合格，但不知道是哪道门控在拖后腿。
  把 qc_info["reason"] 按 Gate 做 Counter 统计，瓶颈立刻清楚：
    Gate1 占大头 → 形状检测/几何对齐有问题，该调 std_shape 或容差
    Gate2 占大头 → 边缘融合有问题，该调羽化或 grow
    Gate3 占大头 → 光度对齐有问题，该调 brightness_gain 或映射

输出：
  1) 各 Gate 的失败数、占比
  2) 每个 Gate 内部最频繁的具体原因（如 Gate1_fill_0.42）
  3) 按原因分组的样本缩略图（可选 --sheet），直接看长什么样

用法：
  python3 qc_stats.py --out /data/render/xxx
  python3 qc_stats.py --out /data/render/xxx --sheet   # 额外生成失败样张
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import cv2
    import numpy as np

    from auto_qc import qc_check
    from batch_replace import std_shape

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="产出目录")
    ap.add_argument("--limit", type=int, default=0, help="最多统计多少张（0=全部）")
    ap.add_argument("--edge-jump", type=float, default=38.0)
    ap.add_argument("--bright-diff", type=float, default=38.0)
    ap.add_argument("--sheet", action="store_true", help="生成各失败原因的样张图")
    ap.add_argument("--sheet-per-reason", type=int, default=3)
    a = ap.parse_args()

    ann_dir = os.path.join(a.out, "annotations")
    if not os.path.isdir(ann_dir):
        print("找不到 annotations 目录：%s" % ann_dir)
        return 2

    files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if a.limit:
        files = files[:a.limit]
    print("统计 %d 个样本（%s）" % (len(files), a.out))

    gate_counter = Counter()          # Gate1 / Gate2 / Gate3
    reason_counter = Counter()        # Gate1_fill_0.42 这类完整原因
    shape_counter = Counter()         # 失败样本的形状分布
    size_bucket = Counter()           # 失败样本的尺寸分布
    samples = {}                      # 每个原因收集几张样本图
    passed = 0

    for fn in files:
        try:
            with open(fn, encoding="utf-8") as f:
                rec = json.load(f)
            objs = rec.get("objects", [])
            if not objs:
                continue
            src = rec.get("source_path", "")
            comp = os.path.join(a.out, rec.get("image", ""))
            if not (os.path.isfile(src) and os.path.isfile(comp)):
                continue
            orig = cv2.imread(src)
            composed = cv2.imread(comp)
            if orig is None or composed is None or orig.shape != composed.shape:
                continue

            frame_fail = None
            for o in objs:
                bb = o.get("bbox")
                if not bb:
                    continue
                sh = std_shape(orig, bb)
                if sh == "unknown":
                    sh = "circle"
                ok, info = qc_check(orig, composed, bb, sh,
                                    edge_jump_max=a.edge_jump,
                                    bright_diff_max=a.bright_diff)
                if not ok:
                    frame_fail = (info.get("reason", "unknown"), bb, sh)
                    break

            if frame_fail is None:
                passed += 1
                continue

            reason, bb, sh = frame_fail
            gate = reason.split("_")[0] if "_" in reason else reason
            # 归一化：去掉具体数值，聚合同类问题
            # 归一化：把带具体数值的原因聚成同类
            if reason.startswith("Gate1_contour_small"):
                key = "Gate1_contour_small"
            elif reason.startswith("Gate1_fill_extreme"):
                key = "Gate1_fill_extreme"
            elif reason.startswith("Gate1_no_contour"):
                key = "Gate1_no_contour"
            elif reason.startswith("Gate1_contour_tiny"):
                key = "Gate1_contour_tiny"
            elif reason.startswith("Gate2_edge_jump"):
                key = "Gate2_edge_jump"
            elif reason.startswith("Gate3_bright"):
                key = "Gate3_brightness"
            else:
                key = reason
            gate_counter[gate] += 1
            reason_counter[key] += 1
            shape_counter[sh] += 1
            w = int(bb[2] - bb[0])
            bucket = "20-40" if w < 40 else "40-60" if w < 60 else "60-80" if w < 80 else "80+"
            size_bucket[bucket] += 1

            if a.sheet and len(samples.get(key, [])) < a.sheet_per_reason:
                x1, y1, x2, y2 = [int(v) for v in bb]
                pad = max(8, int((x2 - x1) * 0.4))
                H, W = composed.shape[:2]
                crop = composed[max(0, y1 - pad):min(H, y2 + pad),
                                max(0, x1 - pad):min(W, x2 + pad)]
                if crop.size:
                    samples.setdefault(key, []).append((crop.copy(), sh, w))
        except Exception:
            continue

    total = passed + sum(gate_counter.values())
    if total == 0:
        print("没有可统计的样本")
        return 2

    print()
    print("=" * 62)
    print("QC 瓶颈分析")
    print("=" * 62)
    print("总数 %d　通过 %d（%.1f%%）　失败 %d"
          % (total, passed, 100.0 * passed / total, total - passed))

    print()
    print("【各门控占比】按失败数排序，占比最高的就是当前瓶颈")
    if not gate_counter:
        print("  没有失败样本")
    for gate, n in gate_counter.most_common():
        pct = 100.0 * n / max(total - passed, 1)
        bar = "█" * int(pct / 3)
        tag = ""
        if gate == "Gate1":
            tag = "→ 形状检测/几何对齐有问题，调 std_shape 的 fill 容差"
        elif gate == "Gate2":
            tag = "→ 边缘融合有问题，调 feather_ratio 或 grow"
        elif gate == "Gate3":
            tag = "→ 光度对齐有问题，调 brightness_gain 或对比度映射"
        print("  %-7s %5d  %5.1f%%  %-28s %s" % (gate, n, pct, bar, tag))

    print()
    print("【具体失败原因】")
    for r, n in reason_counter.most_common(10):
        print("  %-30s %4d" % (r, n))

    print()
    print("【失败样本的形状分布】")
    for s_, n in shape_counter.most_common():
        print("  %-12s %4d" % (s_, n))

    print()
    print("【失败样本的尺寸分布】")
    for b_, n in size_bucket.most_common():
        print("  %-10spx %4d" % (b_, n))

    # 生成样张
    if a.sheet and samples:
        tiles = []
        for key, items in samples.items():
            for crop, sh, w in items:
                S = 165
                c = cv2.resize(crop, (S, S), interpolation=cv2.INTER_NEAREST)
                cv2.putText(c, key.replace("Gate", "G"), (2, 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 255), 1)
                cv2.putText(c, "%s %dpx" % (sh[:4], w), (2, S - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 220), 1)
                tiles.append(c)
            # 每个原因后补一个空位分隔
            tiles.append(np.full((165, 8, 3), 40, np.uint8))
        if tiles:
            per_row = 5
            while len(tiles) % per_row:
                tiles.append(np.full((165, 165, 3), 40, np.uint8))
            rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
            out = np.vstack(rows)
            sp = os.path.join(a.out, "qc_fail_samples.jpg")
            cv2.imwrite(sp, out)
            print()
            print("失败样张已保存: %s" % sp)

    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
