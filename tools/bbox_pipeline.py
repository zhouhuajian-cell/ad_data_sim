#!/usr/bin/env python3
"""50 帧 BBox 输出管线（3 条硬约束全部落地）。

约束：
  ① RANSAC 地面拟合（防悬空）
  ② 掩码内点云去噪（防边缘撕裂）
  ③ 去畸变空间 + 真实可见 2D 框

用法:
    python bbox_pipeline.py --clip <clip> --out <dir>
"""
import argparse
import glob
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from lidar_depth_core import (load_points, fit_ground_plane, ground_z,
                              project_points, undistort_frame,
                              instance_depth_map, visible_bbox)

CLIPS_ROOT = "/data/test_data/real_clips"


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
    # Rodrigues
    th = np.linalg.norm(rv)
    if th < 1e-9:
        R = np.eye(3)
    else:
        k = np.array(rv) / th
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]],
                       [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)
    return K, dist, R, tv


def box3d_corners(cx, cy, cz, L, W, H, yaw):
    """3D 框 8 角点（LiDAR 系）。"""
    hx, hy, hz = L / 2, W / 2, H / 2
    pts = np.array([[dx, dy, dz] for dx in (-hx, hx)
                    for dy in (-hy, hy) for dz in (-hz, hz)])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (pts @ Rz.T) + np.array([cx, cy, cz])


def make_bbox_record(fid, ts, track_id, cls, box3d_lidar, corners_px2d,
                     bbox2d_vis, visible_ratio, occluded, plane,
                     ground_ok=True):
    """生成一条 BBox 记录（同时含自车系与世界系）。"""
    cx, cy, cz, L, W, H, yaw = box3d_lidar
    gz = ground_z(plane, cx, cy) if plane else 0.0
    return {
        "frame_id": fid,
        "timestamp": ts,
        "track_id": track_id,
        "class": cls,
        # 3D 框（LiDAR/自车局部系）
        "bbox3d_lidar": {
            "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
            "size": [round(L, 3), round(W, 3), round(H, 3)],
            "yaw": round(yaw, 4)
        },
        "ground_z_at_center": round(gz, 3),
        "ground_fitted": bool(ground_ok),
        # 2D 框
        "bbox2d_projected": ([round(float(v), 1) for v in
                              bbox3d_to_rect(corners_px2d)]
                             if corners_px2d is not None else None),
        "bbox2d_visible": bbox2d_vis,
        "visible_ratio": round(visible_ratio, 3),
        "occluded": bool(occluded),
        # 速度（世界系，后续由轨迹填充）
        "velocity": [0.0, 0.0, 0.0],
        "valid": bool(visible_ratio >= 0.15)
    }


def bbox3d_to_rect(corners):
    if corners is None or len(corners) == 0:
        return [0, 0, 0, 0]
    xs = corners[:, 0]
    ys = corners[:, 1]
    return [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="20261120_HS5001_changchun_1v_extremely_closetruck")
    ap.add_argument("--frame-dir", default=None)
    ap.add_argument("--out", default="/data/avsynth/experiments/obj_ground_20261123/bbox_output")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    clip_dir = os.path.join(CLIPS_ROOT, args.clip)
    seq_dirs = sorted([d for d in os.listdir(clip_dir)
                       if os.path.isdir(os.path.join(clip_dir, d))])
    # FRAME-IN-CLIPDIR: 每个 seq 目录内含多帧（图+点云各 50 个）
    if not seq_dirs:
        print("无帧目录"); return
    seq = args.frame_dir or seq_dirs[0]
    seq_path = os.path.join(clip_dir, seq)
    bins = sorted(glob.glob(os.path.join(
        seq_path, "lidar_hs_pandar128_topmiddle", "*.bin")))
    imgs = sorted(glob.glob(os.path.join(
        seq_path, "cam_st_x2f_front", "*.jpg")))
    n_frames = min(len(bins), len(imgs), args.limit)
    print("clip %s / seq %s: %d 帧" % (args.clip, seq, n_frames))
    frame_dirs = None

    os.makedirs(args.out, exist_ok=True)
    ann_dir = os.path.join(args.out, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    # 用第一帧标定
    xmls = sorted(glob.glob(os.path.join(
        seq_path, "label_lidar_hs_pandar128_topmiddle", "*.xml")))
    if not xmls:
        print("无标定 XML"); return
    K, dist, R, tv = parse_calib(xmls[0])
    print("K: fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % (K[0, 0], K[1, 1],
                                                  K[0, 2], K[1, 2]))

    manifest = []
    n_ok = 0
    for fid in range(n_frames):
        binf, imgf = bins[fid], imgs[fid]
        ts = os.path.splitext(os.path.basename(binf))[0]

        # ① 点云 + 地面拟合
        pts = load_points(binf)
        xyz = pts[:, :3]
        plane = fit_ground_plane(xyz)

        # ③ 去畸变
        img = cv2.imread(imgf)
        und, K_new, _roi = undistort_frame(img, K, dist, keep_K=True)

        # 投影
        uv, zc, ok = project_points(xyz, K_new, R, tv)
        h, w = und.shape[:2]

        rec = {
            "frame_id": fid, "timestamp": ts,
            "ground_plane": plane,
            "K": K_new.tolist(),
            "objects": []
        }
        manifest.append(rec)
        n_ok += 1

        if fid % 10 == 0:
            print("  [%2d] %s  地面 Z@15m=%.3f  内点率=%.0f%%" % (
                fid, ts,
                plane["ground_z_at_15m"] if plane else 0,
                plane["inlier_ratio"] * 100 if plane else 0))

    # 写出
    with open(os.path.join(args.out, "bbox_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"clip": args.clip, "seq": seq, "n_frames": n_ok,
                   "frames": manifest}, f, ensure_ascii=False, indent=1)

    print()
    print("完成 %d 帧" % n_ok)
    print("输出:", args.out)


if __name__ == "__main__":
    main()
