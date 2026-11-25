#!/usr/bin/env python3
"""解析 lidar .bin 点云格式。"""
import glob
import os
import struct
import sys

import numpy as np


def probe(path):
    """探测点云格式。"""
    sz = os.path.getsize(path)
    print("文件: %s" % os.path.basename(path))
    print("大小: %d 字节" % sz)

    # 常见格式：float32 x,y,z,intensity = 16 字节
    for rec in (16, 20, 24, 32):
        if sz % rec == 0:
            n = sz // rec
            print("  %d 字节/点 → %d 点" % (rec, n))

    # 试 float32 x,y,z,i
    raw = np.fromfile(path, dtype=np.float32)
    if len(raw) % 4 == 0:
        pts = raw.reshape(-1, 4)
        print("\n按 float32(x,y,z,i) 解析: %d 点" % len(pts))
        print("  x: [%.2f, %.2f]" % (pts[:, 0].min(), pts[:, 0].max()))
        print("  y: [%.2f, %.2f]" % (pts[:, 1].min(), pts[:, 1].max()))
        print("  z: [%.2f, %.2f]" % (pts[:, 2].min(), pts[:, 2].max()))
        print("  i: [%.2f, %.2f]" % (pts[:, 3].min(), pts[:, 3].max()))
        # 距离范围
        d = np.linalg.norm(pts[:, :3], axis=1)
        print("  距离: [%.2f, %.2f] m" % (d.min(), d.max()))
        # 前 5 点
        print("\n前 5 点:")
        for r in pts[:5]:
            print("  (%.2f, %.2f, %.2f) i=%.2f" % tuple(r))
        return pts
    return None


def main():
    D = ("/data/test_data/real_clips/"
         "20261120_HS5001_changchun_1v_extremely_closetruck/"
         "133913-1788137262.664/lidar_hs_pandar128_topmiddle")
    fs = sorted(f for f in glob.glob(os.path.join(D, "*.bin")))[:3]
    if not fs:
        raise SystemExit("无点云")

    for f in fs[:1]:
        probe(f)

    # 与标注对比：车在哪
    print("\n=== 与 XML 标注对比 ===")
    sys.path.insert(0, "/opt/ue5_synth/tools")
    from label_gate import load_frame_labels

    clip = os.path.dirname(D)
    lab = load_frame_labels(
        clip, xml_path=os.path.join(
            clip, "label_lidar_hs_pandar128_topmiddle",
            os.path.basename(fs[0]).replace(".bin", ".xml")))
    if lab:
        print("前方物体:")
        for o in (lab.get("objects") or [])[:6]:
            if o.get("delete") or not o.get("in_front"):
                continue
            print("  %-10s Wx=%.1f Wy=%.1f Wz=%.1f L=%.1f W=%.1f H=%.1f" % (
                o.get("kind"), o.get("Wx", 0), o.get("Wy", 0),
                o.get("Wz", 0), o.get("L", 0), o.get("W", 0),
                o.get("H", 0)))


if __name__ == "__main__":
    main()
