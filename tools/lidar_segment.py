#!/usr/bin/env python3
"""点云辅助分割：用点云精确勾出物体轮廓。

原理：
  1) 点云转到相机像素
  2) 落在 3D 框内的点云 = 物体本体
  3) 在图像上用点云密度做 GrabCut 的初始化
  4) 得到精确掩码

解决了纯 GrabCut 的「包含背景」问题。
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from label_gate import load_frame_labels
from lidar_io import load_lidar
from real_calib import body_to_pixel_real, load_cam_real


def find_frame_files(clip, stem):
    """找对应的 lidar / image / xml。"""
    ld = os.path.join(clip, "lidar_hs_pandar128_topmiddle")
    cd = os.path.join(clip, "cam_st_x2f_front")
    xd = os.path.join(clip, "label_lidar_hs_pandar128_topmiddle")
    bp = os.path.join(ld, stem + ".bin")
    ip = os.path.join(cd, stem + ".jpg")
    xp = os.path.join(xd, stem + ".xml")
    return (bp if os.path.isfile(bp) else None,
            ip if os.path.isfile(ip) else None,
            xp if os.path.isfile(xp) else None)


def points_in_box(pts, o):
    """筛出 3D 框内的点云。

    框定义：中心 (Wx, Wy, Wz)，尺寸 (L, W, H)，朝向 yaw
    """
    X = pts[:, 0]     # 前后
    Y = pts[:, 1]     # 左右
    Z = pts[:, 2]     # 高度

    cx = o.get("Wx") or 0.0
    cy = o.get("Wy") or 0.0
    cz = o.get("Wz") or 0.0
    L = o.get("L") or 2.0
    W = o.get("W") or 1.8
    H = o.get("H") or 1.5
    yaw = o.get("yaw") or 0.0

    # 转到框坐标系
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx = X - cx
    dy = Y - cy
    # 旋转（绕 Z）
    lx = c * dx - s * dy     # 框系纵向
    ly = s * dx + c * dy     # 框系横向

    m = ((np.abs(lx) <= L / 2) & (np.abs(ly) <= W / 2)
         & (np.abs(Z - cz) <= H / 2))
    return m


def build_mask_from_points(pts, o, cam, img_shape, dilate=3):
    """点云 → 像素掩码。"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), np.uint8)

    m = points_in_box(pts, o)
    if m.sum() < 5:
        return None, 0

    box_pts = pts[m][:, :3]  # (x, y, z) in body
    uv, zc, ok = body_to_pixel_real(cam, box_pts)
    valid = ok & np.isfinite(uv).all(axis=1)
    if valid.sum() < 5:
        return None, int(valid.sum())

    u = uv[valid, 0]
    v = uv[valid, 1]

    # 限制在图像内
    inb = (u >= -20) & (u < w + 20) & (v >= -20) & (v < h + 20)
    u, v = u[inb], v[inb]
    if len(u) < 5:
        return None, len(u)

    # 画成掩码（点在物体表面，需要填充内部）
    for uu, vv in zip(u, v):
        cv2.circle(mask, (int(uu), int(vv)), dilate, 255, -1)

    # 形态学闭运算填洞
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

    # 取最大连通块
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, len(u)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8) * 255

    return mask, len(u)


def main():
    clip = ("/data/test_data/real_clips/"
            "20261120_HS5001_changchun_1v_extremely_closetruck/"
            "133913-1788137262.664")
    cam = load_cam_real(os.path.basename(os.path.dirname(clip)),
                        os.path.basename(clip))
    if cam is None:
        raise SystemExit("无标定")

    stems = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(
                       clip, "lidar_hs_pandar128_topmiddle", "*.bin")))[:3]

    out_dir = "/tmp/lidar_seg"
    os.makedirs(out_dir, exist_ok=True)

    for stem in stems:
        bp, ip, xp = find_frame_files(clip, stem)
        if not (bp and ip and xp):
            print("缺文件:", stem)
            continue

        pts = load_lidar(bp)
        img = cv2.imread(ip)
        lab = load_frame_labels(clip, xml_path=xp)
        if lab is None:
            continue

        print("\n=== %s ===" % stem)
        print("点云: %d  图像: %s" % (len(pts), img.shape))

        vis = img.copy()
        n_ok = 0
        for o in (lab.get("objects") or []):
            if o.get("delete") or not o.get("in_front"):
                continue
            bbox = o.get("bbox2d")
            if not bbox:
                continue
            kind = o.get("kind") or "?"

            mask, n_pts = build_mask_from_points(pts, o, cam, img.shape)
            if mask is None:
                print("  %-10s 点云不足 (%d 点)" % (kind, n_pts))
                continue

            # 覆盖率：点云掩码覆盖了 bbox 多少
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            bm = mask[y1:y2, x1:x2]
            cov = (bm > 0).mean()
            print("  %-10s 点云 %4d 个  掩码覆盖 bbox %.0f%%  bbox=%s"
                  % (kind, n_pts, cov * 100, [round(v) for v in bbox]))

            if cov < 0.15:
                continue

            # 画出来
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            color = (0, 255, 0) if kind == "truck" else (255, 128, 0)
            cv2.drawContours(vis, cnts, -1, color, 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 128, 255), 1)
            n_ok += 1

        op = os.path.join(out_dir, stem + "_seg.jpg")
        cv2.imwrite(op, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("  → %s (成功 %d 个)" % (op, n_ok))

    print("\n产出:", out_dir)


if __name__ == "__main__":
    main()
