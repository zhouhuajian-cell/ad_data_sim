#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_asset_grid.py — 把素材目录拼成总览图，便于快速看质量

用法：python3 make_asset_grid.py <素材目录> [输出jpg] [最多几张]
"""
import glob
import os
import sys


def main():
    import cv2
    import numpy as np

    src = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/asset_grid.jpg"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    files = sorted(glob.glob(os.path.join(src, "*.png")))
    # 每个类别取第一张（不带 _v 后缀的）
    firsts = [f for f in files if "_v" not in os.path.basename(f)]
    if not firsts:
        firsts = files
    firsts = firsts[:limit]
    print("候选文件 %d 个，取 %d 个" % (len(files), len(firsts)))

    S = 120
    tiles = []
    for f in firsts:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.ndim == 3 and im.shape[2] == 4:
            a = im[:, :, 3:4].astype(np.float32) / 255.0
            comp = (im[:, :, :3].astype(np.float32) * a + 232 * (1 - a)).astype(np.uint8)
        elif im.ndim == 3:
            comp = im
        else:
            comp = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        comp = cv2.resize(comp, (S, S), interpolation=cv2.INTER_AREA)
        name = os.path.basename(f).replace(".png", "")[:17]
        cv2.putText(comp, name, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 190), 1)
        tiles.append(comp)

    print("成功合成 %d 个格子" % len(tiles))
    if not tiles:
        print("没有可用的图")
        return 1
    while len(tiles) % 6:
        tiles.append(np.full((S, S, 3), 245, np.uint8))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    grid = np.vstack(rows)
    cv2.imwrite(out, grid)
    print("已写出 %s  尺寸 %s" % (out, grid.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
