#!/usr/bin/env python3
"""Pandar128 点云解析（正确格式）。

格式： [4字节点数 uint32] + [N × 16字节 (intensity, x, y, z) float32]
"""
import glob
import os
import struct

import numpy as np


def load_lidar(path):
    """读点云。返回 (N,4) 数组 [i, x, y, z]。"""
    sz = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(4)
        n = struct.unpack("<I", head)[0]
        rest = f.read()
    if n * 16 == len(rest):
        arr = np.frombuffer(rest, dtype=np.float32).reshape(n, 4)
        return arr
    # 兜底
    arr = np.fromfile(path, dtype=np.float32)[1:]
    m = (len(arr) // 4) * 4
    return arr[:m].reshape(-1, 4)


def describe(pts):
    i, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    print("点数: %d" % len(pts))
    for name, a in (("i", i), ("x", x), ("y", y), ("z", z)):
        print("  %s: [%.2f, %.2f]  median=%.2f" % (
            name, a.min(), a.max(), np.median(a)))
    d = np.linalg.norm(pts[:, 1:4], axis=1)
    print("  距离: [%.2f, %.2f]" % (d.min(), d.max()))


if __name__ == "__main__":
    p = ("/data/test_data/real_clips/"
         "20261120_HS5001_changchun_1v_extremely_closetruck/"
         "133913-1788137262.664/lidar_hs_pandar128_topmiddle/"
         "1788137262.664.bin")
    pts = load_lidar(p)
    describe(pts)
    print("\n前 6 点:")
    for r in pts[:6]:
        print("  i=%.2f x=%.2f y=%.2f z=%.2f" % tuple(r))
