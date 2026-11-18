#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_alpha.py — 用素材的真实透明度重设 alpha，而不是按形状猜

问题（实测）：
  build_library_official.py 按「类别→形状」生成几何 alpha（圆/三角/矩形），
  但很多素材的 SVG 图形本身不等于那个几何形状：
    prohibition_yield_x4  八角形，按圆形 alpha → 边缘一堆透明区被当成不透明
    mandatory_turn_right  SVG 图形比画布小，按整圆 alpha → 四周多出一圈
    warning_railway       实际图形占比 41%，按三角形 alpha → 上缘不对

  贴图时 alpha 盖到了透明区，等于把素材的空白背景一起贴上去 —— 表现为重影。

本脚本：
  对每个 PNG，用「自身 RGB 与白色的差异」重建 alpha：
    - 先排除 SVG 自带 alpha 已透明的像素
    - 再用「饱和度 + 与纯白距离」判断哪些是图形本体
    - 形态学闭运算填掉图形内部的小洞（如数字笔画中间），保留外轮廓
  这样 alpha 严格跟随素材的实际形状。

用法：
  python3 fix_alpha.py --library /data/tsr_library_v2 --countries BR,AR,CL
"""
import argparse
import os
import sys


def rebuild_alpha(rgba, white_thresh=246, min_cov=0.15):
    """用自身像素重建 alpha。rgba 是 BGRA。

    返回新的 uint8 alpha。
    """
    import cv2
    import numpy as np

    bgr = rgba[:, :, :3]
    src_a = rgba[:, :, 3]
    h, w = src_a.shape

    # 1) 原 alpha 认为透明的，保持透明
    base = (src_a > 20).astype(np.uint8) * 255

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # 2) 图形本体 = 有颜色（饱和度高）或 不是纯白
    colored = sat > 28
    not_white = g < white_thresh
    body = ((colored | not_white) & (base > 0)).astype(np.uint8) * 255

    # 3) 闭运算填掉内部小洞（数字/字母笔画间的缝隙），但不跨出外轮廓
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 40) | 1,
                                                      max(3, h // 40) | 1))
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k, iterations=2)

    # 4) 取最大连通域（去掉角落杂点）
    n, lab, stats, _ = cv2.connectedComponentsWithStats(body, connectivity=8)
    if n <= 1:
        return src_a
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
    areas.sort(reverse=True)
    biggest = areas[0][1]
    keep = (lab == biggest).astype(np.uint8) * 255

    # 覆盖太小时（说明判断失败）退回原 alpha
    cov = keep.mean() / 255.0
    if cov < min_cov:
        return src_a

    # 5) 内外填充：把最大连通域的外轮廓填实（有些图形内部是白底，会被 above 判掉）
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        filled = np.zeros_like(keep)
        cv2.drawContours(filled, cnts, -1, 255, -1)
        # 只在 base 范围内生效（保留 SVG 原本的锯齿边）
        keep = cv2.bitwise_and(filled, base) if base.mean() > 0 else filled

    # 6) 轻羽化 1px，避免硬锯齿
    keep = cv2.GaussianBlur(keep, (3, 3), 0.6)
    return keep


def crop_to_content(rgba, pad_ratio=0.02):
    """按 alpha 的实际外接矩形裁紧，再把内容居中放到正方形画布。

    为什么必须做：
      实测 prohibition_yield_x4 图形只占画布 39%、mandatory_turn_right 55%，
      贴图时把整个画布缩到 bbox，实际图形就比原标牌小一圈；
      四周空白一旦 alpha 有偏差，就会盖在背景上 —— 这就是重影的来源。
      裁紧后图形贴合 bbox，尺寸才对得上。
    """
    import cv2
    import numpy as np

    h, w = rgba.shape[:2]
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 30)
    if len(ys) == 0:
        return rgba

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    # 已经基本铺满就跳过
    if cw >= w * 0.92 and ch >= h * 0.92:
        return rgba

    crop = rgba[y0:y1 + 1, x0:x1 + 1]
    # 放到正方形画布，留 2% 内边距，保持长宽比
    side = int(max(cw, ch) * (1 + pad_ratio * 2))
    side = max(side, 16)
    canvas = np.zeros((side, side, 4), np.uint8)
    ox = (side - cw) // 2
    oy = (side - ch) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return canvas


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--countries", default="", help="留空=全部")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.countries:
        ccs = [c.strip().upper() for c in a.countries.split(",") if c.strip()]
    else:
        ccs = [d for d in os.listdir(a.library)
               if os.path.isdir(os.path.join(a.library, d)) and len(d) == 2]

    tot = fixed = 0
    for cc in sorted(ccs):
        d = os.path.join(a.library, cc)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".png"):
                continue
            p = os.path.join(d, fn)
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim != 3 or im.shape[2] != 4:
                continue
            tot += 1
            old_cov = (im[:, :, 3] > 127).mean()
            # 先按内容裁紧，再修 alpha
            cropped = crop_to_content(im)
            if cropped.shape != im.shape:
                im = cropped
                old_cov = (im[:, :, 3] > 127).mean()
            new_a = rebuild_alpha(im)
            new_cov = (new_a > 127).mean()
            if abs(new_cov - old_cov) > 0.02:
                if not a.dry_run:
                    im[:, :, 3] = new_a
                    cv2.imwrite(p, im)
                fixed += 1
        if fixed:
            print("  %-3s 已处理" % cc, flush=True)

    print("-" * 50)
    print("检查 %d 张，%s %d 张" % (tot, "需修正" if a.dry_run else "修正", fixed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
