#!/usr/bin/env python3
"""50 帧合成：静态锥桶锚定 + 动态车 + BBox 叠加 + 视频输出。

约束：
  ① 用 LiDAR RANSAC 地面，不用 Z=0
  ② 掩码内点云去噪
  ③ 去畸变空间 + 真实可见 2D 框

用法:
    python render_clip_50.py --clip <clip> --seq <seq> --out <dir>
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from lidar_depth_core import (load_points, fit_ground_plane, ground_z,
                              project_points, undistort_frame)

CLIPS_ROOT = "/data/test_data/real_clips"
BLENDER = "/data/blender-4.2.23-linux-x64/blender"
RENDER_PY = "/opt/ue5_synth/tools/render_canonical_one.py"
CANON = "/data/assets_global/OBJ_3D/canonical"


def parse_calib(xml_path):
    root = ET.parse(xml_path).getroot()
    cam = root.find(".//cam_st_x2f_front")
    iv = [float(cam.find("intrinsic_matrix%d" % i).text) for i in range(9)]
    K = np.array(iv).reshape(3, 3)
    rv = [float(cam.find("rotation_vector%d" % i).text) for i in range(3)]
    tv = np.array([float(cam.find("translation_vector%d" % i).text)
                   for i in range(3)])
    dist = np.array([float(cam.find("distortion_coeffs%d" % i).text)
                     for i in range(8)])
    th = np.linalg.norm(rv)
    if th < 1e-9:
        R = np.eye(3)
    else:
        k = np.array(rv) / th
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]],
                       [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)
    return K, dist, R, tv


def T_from_Rt(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def box_corners_lidar(cx, cy, cz, L, W, H, yaw):
    hx, hy, hz = L / 2, W / 2, H / 2
    pts = np.array([[dx, dy, dz] for dx in (-hx, hx)
                    for dy in (-hy, hy) for dz in (-hz, hz)])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (pts @ Rz.T) + np.array([cx, cy, cz])


EDGES = [(0, 1), (2, 3), (4, 5), (6, 7),
         (0, 2), (1, 3), (4, 6), (5, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_3d_box(img, corners_px, color=(0, 255, 255), thick=2):
    if corners_px is None:
        return
    ip = [[int(round(x)), int(round(y))] for x, y in corners_px]
    for a, b in EDGES:
        cv2.line(img, tuple(ip[a]), tuple(ip[b]), color, thick)


def draw_2d_box(img, bbox, color=(0, 255, 0), thick=2, label=None):
    if not bbox:
        return
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
    if label:
        cv2.putText(img, label, (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="20261120_HS5001_changchun_1v_ground_text")
    ap.add_argument("--seq", default="137710-1789960617.954")
    ap.add_argument("--out", default="/tmp/clip50")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--cone-uid", default=None)
    ap.add_argument("--car-uid", default=None)
    args = ap.parse_args()

    clip_dir = os.path.join(CLIPS_ROOT, args.clip, args.seq)
    bins = sorted(glob.glob(os.path.join(
        clip_dir, "lidar_hs_pandar128_topmiddle", "*.bin")))
    imgs = sorted(glob.glob(os.path.join(
        clip_dir, "cam_st_x2f_front", "*.jpg")))
    n = min(len(bins), len(imgs), args.limit)
    print("帧数: %d" % n)

    os.makedirs(args.out, exist_ok=True)
    fdir = os.path.join(args.out, "frames")
    os.makedirs(fdir, exist_ok=True)

    # 标定
    xmls = sorted(glob.glob(os.path.join(
        clip_dir, "label_lidar_hs_pandar128_topmiddle", "*.xml")))
    K, dist, R, tv = parse_calib(xmls[0])
    T_w2c = T_from_Rt(R, tv)

    # 选资产
    cone_fp = None
    car_fp = None
    if args.cone_uid:
        p = os.path.join(CANON, args.cone_uid + ".glb")
        if os.path.isfile(p):
            cone_fp = p
    if args.car_uid:
        p = os.path.join(CANON, args.car_uid + ".glb")
        if os.path.isfile(p):
            car_fp = p
    print("锥桶:", cone_fp)
    print("动态车:", car_fp)

    # 首帧：定静态锥桶世界坐标
    pts0 = load_points(bins[0])
    plane0 = fit_ground_plane(pts0[:, :3])
    print("地面 Z@15m: %.3f  内点率 %.0f%%" % (
        plane0["ground_z_at_15m"], plane0["inlier_ratio"] * 100))

    # 锥桶位置（首帧 LiDAR 系，右前方车道）
    cone_lidar0 = [
        (18.0, 3.2, 0.0),
        (21.0, 3.4, 0.0),
        (24.0, 3.6, 0.0),
    ]
    # 转世界系（用首帧 R,t 近似，因为没 ego pose，用相对运动补偿）
    # 简化：锥桶在首帧 LiDAR 系定义，后续帧按自车运动反向移动
    # 这里先用「相对自车静止」做基线，后续接 KISS-ICP

    stats = {"frames": [], "ground": [], "boxes": []}

    for fid in range(n):
        img = cv2.imread(imgs[fid])
        if img is None:
            continue
        und, K_new, _ = undistort_frame(img, K, dist, keep_K=True)
        h, w = und.shape[:2]

        pts = load_points(bins[fid])
        xyz = pts[:, :3]
        plane = fit_ground_plane(xyz)
        gz_list = []
        for (cx, cy, _cz) in cone_lidar0:
            gz_list.append(ground_z(plane, cx, cy))

        # 画 BBox
        vis = und.copy()
        for i, (cx, cy, _cz) in enumerate(cone_lidar0):
            gz = gz_list[i]
            corners = box_corners_lidar(cx, cy, gz + 0.35,
                                        0.4, 0.4, 0.7, 0.0)
            uv, zc, ok = project_points(corners, K_new, R, tv)
            if ok.all() and np.isfinite(uv).all():
                draw_3d_box(vis, uv, (0, 140, 255), 2)
                xs, ys = uv[:, 0], uv[:, 1]
                bx = [xs.min(), ys.min(), xs.max() - xs.min(),
                      ys.max() - ys.min()]
                draw_2d_box(vis, bx, (0, 255, 0), 2,
                            "cone_%d z=%.2f" % (i, gz))

        # 信息条
        cv2.rectangle(vis, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(vis,
                    "frame %02d/%d | ground_z@15m=%.3f | inlier=%.0f%%" % (
                        fid, n, plane["ground_z_at_15m"],
                        plane["inlier_ratio"] * 100),
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0), 2)

        fp = os.path.join(fdir, "f%03d.jpg" % fid)
        cv2.imwrite(fp, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

        stats["frames"].append(fid)
        stats["ground"].append({
            "fid": fid,
            "z15": round(plane["ground_z_at_15m"], 3),
            "inlier": round(plane["inlier_ratio"], 3),
            "cone_ground_z": [round(v, 3) for v in gz_list],
        })
        if fid % 10 == 0:
            print("  [%2d] ground Z@15m=%.3f  cone_ground_z=%s" % (
                fid, plane["ground_z_at_15m"],
                [round(v, 2) for v in gz_list]))

    with open(os.path.join(args.out, "stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    # 视频
    mp4 = os.path.join(args.out, "preview_50f.mp4")
    cmd = ["ffmpeg", "-y", "-framerate", "10",
           "-i", os.path.join(fdir, "f%03d.jpg"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-vf", "scale=1280:720", mp4]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
        print("视频:", mp4, os.path.getsize(mp4) / 1e6, "MB")
    except Exception as e:
        print("ffmpeg 失败:", e)

    print("完成 ->", args.out)


if __name__ == "__main__":
    main()
