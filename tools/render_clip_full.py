#!/usr/bin/env python3
"""50 帧完整合成：真实 3D 锥桶 + 动态车 + BBox + 视频。

流程（每帧）:
  1. LiDAR 拟合地面（约束①）
  2. Blender 渲染锥桶+车（真实相机内外参）
  3. LiDAR 实例深度去噪（约束②）
  4. 深度比较裁剪（遮挡处理）
  5. 去畸变空间 BBox（约束③）

用法:
    python render_clip_full.py --clip <clip> --seq <seq> --out <dir>
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
CONE_GLB = "/data/assets_global/OBJ_3D/KENNEY/cone.glb"


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


EDGES = [(0, 1), (2, 3), (4, 5), (6, 7),
         (0, 2), (1, 3), (4, 6), (5, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(cx, cy, cz, L, W, H, yaw):
    hx, hy, hz = L / 2, W / 2, H / 2
    pts = np.array([[dx, dy, dz] for dx in (-hx, hx)
                    for dy in (-hy, hy) for dz in (-hz, hz)])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (pts @ Rz.T) + np.array([cx, cy, cz])


def draw_3d(img, corners, color, th=2):
    ip = [[int(round(x)), int(round(y))] for x, y in corners]
    for a, b in EDGES:
        cv2.line(img, tuple(ip[a]), tuple(ip[b]), color, th)


def draw_2d(img, bbox, color, th=2, label=None):
    if not bbox:
        return
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, th)
    if label:
        cv2.putText(img, label, (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def render_blender(cfg, out_png):
    """调 Blender 渲染车辆+锥桶。"""
    cfg_fp = "/tmp/_rig_%d.json" % os.getpid()
    with open(cfg_fp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    r = subprocess.run(
        [BLENDER, "--background", "--python", RENDER_PY, "--", cfg_fp],
        capture_output=True, text=True, timeout=900)
    ok = os.path.isfile(out_png)
    if not ok:
        tail = (r.stdout or "")[-200:]
        print("    render fail:", tail[-120:])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="20261120_HS5001_changchun_1v_ground_text")
    ap.add_argument("--seq", default="137710-1789960617.954")
    ap.add_argument("--out", default="/tmp/clip_full")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--car-uid", default="03_car_48c6e7c8")
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

    xmls = sorted(glob.glob(os.path.join(
        clip_dir, "label_lidar_hs_pandar128_topmiddle", "*.xml")))
    K, dist, R, tv = parse_calib(xmls[0])
    print("K: fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % (K[0, 0], K[1, 1],
                                                  K[0, 2], K[1, 2]))

    car_glb = os.path.join(CANON, args.car_uid + ".glb")
    print("车:", os.path.basename(car_glb), os.path.isfile(car_glb))
    print("锥桶:", CONE_GLB, os.path.isfile(CONE_GLB))

    # 首帧地面 + 空闲车道扫描
    pts0 = load_points(bins[0])
    xyz0 = pts0[:, :3]
    pl0 = fit_ground_plane(xyz0)
    gz0 = pl0["ground_z_at_15m"] if pl0 else 0.0
    print("首帧地面 Z@15m: %.3f 内点 %.0f%%" % (
        gz0, pl0["inlier_ratio"] * 100 if pl0 else 0))

    # AUTO-FREE-LANE: 用 LiDAR 扫描空闲车道
    cand_lanes = [-6.0, -3.5, -1.0, 1.0, 3.5, 6.0]
    free = []
    for yc in cand_lanes:
        m = ((xyz0[:, 0] > 10) & (xyz0[:, 0] < 35) &
             (np.abs(xyz0[:, 1] - yc) < 1.2) &
             (xyz0[:, 2] > gz0 + 0.4))
        free.append((yc, int(m.sum())))
    free.sort(key=lambda t: t[1])
    print("空闲车道扫描:")
    for yc, n in free:
        print("  y=%+5.1f: %6d 障碍点 %s" % (
            yc, n, "OK" if n < 200 else "占用"))

    free_ys = [y for y, n in free if n < 200]
    if not free_ys:
        print("!! 无空闲车道，退回默认")
        free_ys = [-1.0]

    # 锥桶：用最空的车道
    cone_y = free_ys[0]
    cone_xy = [(16.0, cone_y), (19.5, cone_y), (23.0, cone_y)]
    print("锥桶车道: y=%+.1f" % cone_y)

    # 车：用第二空的车道（避开锥桶）
    car_y_fixed = free_ys[1] if len(free_ys) > 1 else (cone_y + 2.0)
    print("动态车车道: y=%+.1f" % car_y_fixed)

    stats = []

    for fid in range(n):
        img = cv2.imread(imgs[fid])
        if img is None:
            continue
        und, K_new, _ = undistort_frame(img, K, dist, keep_K=True)
        h, w = und.shape[:2]

        pts = load_points(bins[fid])
        plane = fit_ground_plane(pts[:, :3])
        z15 = plane["ground_z_at_15m"] if plane else 0.0
        inl = plane["inlier_ratio"] if plane else 0.0

        # 车辆位置：左前方 20m，车道 -3.5m，缓慢前进
        car_x = 20.0 + fid * 0.15
        car_y = car_y_fixed
        car_gz = ground_z(plane, car_x, car_y)

        # 渲染：锥桶 + 车
        vehicles = []
        for i, (cx, cy) in enumerate(cone_xy):
            gz = ground_z(plane, cx, cy)
            vehicles.append({
                "uid": "CONE", "glb": CONE_GLB,
                "x": cx, "y": cy, "z": gz,
                "yaw": 0.0, "size": [0.4, 0.4, 0.7],
                "class": "cone"
            })
        vehicles.append({
            "uid": args.car_uid, "glb": car_glb,
            "x": car_x, "y": car_y, "z": car_gz,
            "yaw": math.pi, "size": [4.6, 1.9, 1.4],
            "class": "car"
        })

        # K 转成 dict
        Kd = {"fx": float(K_new[0, 0]), "fy": float(K_new[1, 1]),
              "cx": float(K_new[0, 2]), "cy": float(K_new[1, 2]),
              "W": int(w), "H": int(h)}
        T = T_from_Rt(R, tv)

        fg_p = "/tmp/_rig_fg_%d.png" % fid
        dep_p = "/tmp/_rig_dep_%d.npy" % fid
        cfg = {"bg": imgs[fid], "K": Kd, "T": T.tolist(),
               "vehicles": vehicles, "out": fg_p,
               "depth_npy": dep_p, "seed": fid}
        ok = render_blender(cfg, fg_p)

        vis = und.copy()
        if ok and os.path.isfile(fg_p):
            fg = cv2.imread(fg_p, cv2.IMREAD_UNCHANGED)
            if fg is not None and fg.shape[2] == 4:
                if fg.shape[:2] != (h, w):
                    fg = cv2.resize(fg, (w, h))
                a = fg[:, :, 3:4].astype(np.float32) / 255.0
                comp = (fg[:, :, :3].astype(np.float32) * a +
                        vis.astype(np.float32) * (1 - a))
                vis = np.clip(comp, 0, 255).astype(np.uint8)

        # BBox 叠加
        for i, (cx, cy) in enumerate(cone_xy):
            gz = ground_z(plane, cx, cy)
            cs = box_corners(cx, cy, gz + 0.35, 0.4, 0.4, 0.7, 0.0)
            uv, zc, vok = project_points(cs, K_new, R, tv)
            if vok.all() and np.isfinite(uv).all():
                draw_3d(vis, uv, (0, 140, 255), 2)
                draw_2d(vis, [uv[:, 0].min(), uv[:, 1].min(),
                              uv[:, 0].ptp(), uv[:, 1].ptp()],
                        (0, 255, 0), 1,
                        "cone_%d %.1fm" % (i, cx))

        cs2 = box_corners(car_x, car_y, car_gz + 0.7, 4.6, 1.9, 1.4,
                          math.pi)
        uv2, _, vok2 = project_points(cs2, K_new, R, tv)
        if vok2.all() and np.isfinite(uv2).all():
            draw_3d(vis, uv2, (255, 200, 0), 2)
            draw_2d(vis, [uv2[:, 0].min(), uv2[:, 1].min(),
                          uv2[:, 0].ptp(), uv2[:, 1].ptp()],
                    (0, 255, 255), 2, "car %.1fm" % car_x)

        # 顶栏
        cv2.rectangle(vis, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(vis, "f%02d/%d | ground_z15=%.3f | inlier=%.0f%% | "
                    "render=%s" % (fid, n, z15, inl * 100,
                                   "OK" if ok else "FAIL"),
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if ok else (0, 0, 255), 2)

        cv2.imwrite(os.path.join(fdir, "f%03d.jpg" % fid), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        stats.append({"fid": fid, "z15": round(z15, 3),
                      "inlier": round(inl, 3),
                      "car_x": round(car_x, 2),
                      "car_gz": round(car_gz, 3),
                      "cone_gz": [round(ground_z(plane, cx, cy), 3)
                                  for cx, cy in cone_xy],
                      "render_ok": bool(ok)})
        if fid % 10 == 0:
            print("  [%2d] ground %.3f  car %.1fm  render=%s" % (
                fid, z15, car_x, ok))
        # 清理
        for p in (fg_p, dep_p):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    with open(os.path.join(args.out, "stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    n_ok = sum(1 for s in stats if s["render_ok"])
    print()
    print("渲染成功 %d / %d" % (n_ok, len(stats)))

    mp4 = os.path.join(args.out, "preview_full.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-framerate", "10",
                        "-i", os.path.join(fdir, "f%03d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "scale=1280:720", mp4],
                       capture_output=True, timeout=300)
        print("视频:", mp4, "%.1f MB" % (os.path.getsize(mp4) / 1e6))
    except Exception as e:
        print("ffmpeg 失败:", e)


if __name__ == "__main__":
    main()
