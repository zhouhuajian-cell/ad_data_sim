#!/usr/bin/env python3
"""阶段二：多车型批量生成 + 防碰撞约束。

约束（不满足则重采样）：
  1. 纵向间距 >= 5m 或 横向错开 >= 2.5m（对任意真实车辆）
  2. 裁剪后可见像素占比 >= 30%
  3. 必须落在可行性路面（Mask2Former road mask）内
  4. 距离范围 12~40m（太近穿模、太远看不见）

用法:
    python batch_multi_obj.py --n 10 --out /tmp/multi_out
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")

BLENDER = "/data/blender-4.2.23-linux-x64/blender"
RENDER = "/opt/ue5_synth/tools/render_align.py"
EXR2NPY = "/opt/ue5_synth/tools/exr2npy_vehicle.py"
BG = "/data/align_test/bg.jpg"
KJSON = "/data/align_test/calib_verified.json"
MASKDIR = "/data/test_data/gnd_train/02_HS5-001_20260501/1777604354.022"
MASKSTEM = "1777604355.021"

# 车型 -> (GLB, 目标 LWH)
MODELS = {
    "car":   ("/data/assets_global/OBJ_3D/KENNEY/sedan.glb",    (4.5, 1.8, 1.5)),
    "suv":   ("/data/assets_global/OBJ_3D/KENNEY/suv.glb",      (4.7, 1.9, 1.75)),
    "van":   ("/data/assets_global/OBJ_3D/KENNEY/delivery.glb", (5.2, 1.95, 2.1)),
    "truck": ("/data/assets_global/OBJ_3D/KENNEY/truck.glb",    (7.0, 2.3, 2.9)),
    "cone":  ("/data/assets_global/OBJ_3D/KENNEY/cone.glb",     (0.4, 0.4, 0.7)),
}

MIN_LONG_GAP = 5.0        # 纵向最小间距 (m)
MIN_LAT_GAP = 2.5         # 横向最小错开 (m)
MIN_VISIBLE_RATIO = 0.30  # 裁剪后可见占比
MIN_BBOX_W = 80           # 车身最小像素宽度
DIST_MIN, DIST_MAX = 12.0, 25.0


def load_K():
    return json.load(open(KJSON))


def get_real_vehicles(bg_path, K):
    """Mask2Former 真实车辆 + 接地估距。"""
    from cv_to_blender_cam import estimate_instance_depth_from_ground
    bg = cv2.imread(bg_path)
    mask_path = os.path.join(MASKDIR, "obj_mask_ori_3.1.1", MASKSTEM + ".png")
    raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError("frame-specific object mask missing")
    veh = np.any(raw != 0, axis=2) if raw.ndim == 3 else raw != 0

    Knp = np.array([[K["fx"], 0, K["cx"]],
                    [0, K["fy"], K["cy"]],
                    [0, 0, 1]], dtype=np.float64)
    Rcb = np.asarray(K["R_cam2body"], dtype=float)
    tcb = np.asarray(K["t_cam2body"], dtype=float)
    T = np.eye(4); T[:3, :3] = Rcb.T; T[:3, 3] = -Rcb.T @ tcb

    n, lab, st, _ = cv2.connectedComponentsWithStats(
        veh.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 500:
            continue
        inst = (lab == i)
        ys, xs = np.where(inst)
        y_max = int(ys.max())
        foot = np.zeros_like(inst)
        sel = (ys >= y_max - 3) & (ys <= y_max)
        foot[ys[sel], xs[sel]] = True
        d, _ = estimate_instance_depth_from_ground(foot, Knp, T)
        if d is None:
            continue
        vv = d[np.isfinite(d) & (d > 0)]
        if vv.size == 0:
            continue
        dist = float(np.median(vv))
        if not (2 < dist < 150):
            continue
        # 接地点的世界 x,y
        fy = K["fy"]; cy = K["cy"]
        cam_h = 1.5
        # 用射线反算世界坐标（简化为沿视线方向）
        Kinv = np.linalg.inv(Knp)
        R = T[:3, :3]; C = -R.T @ T[:3, 3]
        px = float(xs[sel].mean()); py = float(y_max)
        ray_c = Kinv @ np.array([px, py, 1.0])
        ray_w = R.T @ ray_c
        if abs(ray_w[2]) < 1e-9:
            continue
        t = (0.0 - C[2]) / ray_w[2]
        if t <= 0:
            continue
        P = C + t * ray_w
        out.append({"dist": dist, "wx": float(P[0]), "wy": float(P[1]),
                    "bbox": [int(st[i, cv2.CC_STAT_LEFT]),
                             int(st[i, cv2.CC_STAT_TOP]),
                             int(st[i, cv2.CC_STAT_WIDTH]),
                             int(st[i, cv2.CC_STAT_HEIGHT])]})
    return out, bg


def is_free(wx, wy, real_veh, my_dist):
    """防碰撞检查。"""
    for v in real_veh:
        d_long = abs(wx - v["wx"])
        d_lat = abs(wy - v["wy"])
        if d_long < MIN_LONG_GAP and d_lat < MIN_LAT_GAP:
            return False, v
    return True, None


def sample_pose(real_veh, rng, calib, road, blocked, lwh, yaw):
    from gnd_calib import body_to_pixel_gnd
    length, width, _ = lwh
    h, w = road.shape
    theta = math.radians(yaw)
    c, si = math.cos(theta), math.sin(theta)
    for _ in range(300):
        x, y = rng.uniform(DIST_MIN, DIST_MAX), rng.uniform(-6.0, 6.0)
        free, _ = is_free(x, y, real_veh, x)
        if not free:
            continue
        corners = np.array([[length/2,width/2],[-length/2,width/2],
                            [-length/2,-width/2],[length/2,-width/2]])
        corners = corners @ np.array([[c,si],[-si,c]]) + [x,y]
        uv, _, ok = body_to_pixel_gnd(calib, np.column_stack([corners,np.zeros(4)]))
        if not ok.all() or not np.isfinite(uv).all():
            continue
        if not ((uv[:,0] >= 0)&(uv[:,0] < w)&(uv[:,1] >= 0)&(uv[:,1] < h)).all():
            continue
        footprint = np.zeros((h,w), np.uint8)
        cv2.fillConvexPoly(footprint, np.rint(uv).astype(np.int32), 1)
        selected = footprint > 0
        if selected.sum() < 30 or float(np.mean(road[selected])) < .96:
            continue
        if float(np.mean(blocked[selected])) > .005:
            continue
        return round(x, 2), round(y, 2)
    return None


def render_one(cls, glb, lwh, pose, yaw, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        BLENDER, "--background", "--python-exit-code", "7", "--python", RENDER, "--",
        "--glb", glb, "--bg", BG, "--k", KJSON,
        "--out", out_dir,
        "--lwh", "%.2f,%.2f,%.2f" % lwh,
        "--pose", "%.2f,%.2f,1.5" % (pose[0], pose[1]),
        "--yaw", str(yaw), "--samples", "24",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    exr = os.path.join(out_dir, "depth_0001.exr")
    npy = os.path.join(out_dir, "depth.npy")
    if os.path.isfile(exr):
        subprocess.run([BLENDER, "--background", "--python", EXR2NPY, "--",
                        exr, npy], capture_output=True, text=True, timeout=120)
    return (r.returncode == 0 and os.path.isfile(npy)
            and os.path.isfile(os.path.join(out_dir, "vehicle_with_shadow.png"))
            and os.path.isfile(os.path.join(out_dir, "meta.json")))


def composite_with_occlusion(fg, depth, bg, real_veh_mask, real_depth_map,
                            tol=0.8, shadow_rgba=None):
    """深度比较裁剪 + alpha 合成。返回 (comp, vis_ratio)。"""
    h, w = bg.shape[:2]
    a = fg[:, :, 3] > 0
    sd = np.asarray(depth, np.float32)
    rd = np.asarray(real_depth_map, np.float32)
    sd_ok = np.isfinite(sd) & (sd > 0.01) & (sd < 1e5)
    rd_ok = real_veh_mask & np.isfinite(rd) & (rd > 0.01) & (rd < 200)

    occ = np.zeros((h, w), bool)
    both = a & sd_ok & rd_ok
    occ[both] = rd[both] < (sd[both] - tol)

    n_before = int(a.sum())
    n_after = int((a & ~occ).sum())
    if shadow_rgba is None or shadow_rgba.shape != fg.shape:
        raise ValueError("missing shadow render")
    out = shadow_rgba.copy()
    # Shadow-only alpha is reduced; vehicle alpha is untouched.
    shadow_only = ~a
    out[:, :, 3][shadow_only] = (out[:, :, 3][shadow_only].astype(np.float32) * .45).astype(np.uint8)
    out[occ, 3] = 0
    # Do not paint synthetic ground shadows over real foreground objects.
    out[(~a) & real_veh_mask, 3] = 0
    ratio = n_after / max(n_before, 1)

    al = out[:, :, 3:4].astype(np.float32) / 255.0
    comp = np.clip(out[:, :, :3].astype(np.float32) * al
                   + bg.astype(np.float32) * (1 - al), 0, 255).astype(np.uint8)
    return comp, ratio, int(occ.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    # OUTROOT-LOCKED
    ap.add_argument("--out",
                    default="/mnt/Data_Platform/zhj_datasim/OBJ_batch")
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    K = load_K()
    import hashlib
    if hashlib.sha256(open(BG, "rb").read()).hexdigest() != K.get("source_image_sha256"):
        raise RuntimeError("background image does not match calibration/masks")
    road_raw = cv2.imread(os.path.join(MASKDIR, "freespace_mask_ori_3.0.0", MASKSTEM + ".png"), cv2.IMREAD_UNCHANGED)
    obj_raw = cv2.imread(os.path.join(MASKDIR, "obj_mask_ori_3.1.1", MASKSTEM + ".png"), cv2.IMREAD_UNCHANGED)
    if road_raw is None or obj_raw is None:
        raise RuntimeError("frame-specific road/object masks unavailable")
    road = np.any(road_raw != 0, axis=2) if road_raw.ndim == 3 else road_raw != 0
    blocked = np.any(obj_raw != 0, axis=2) if obj_raw.ndim == 3 else obj_raw != 0
    blocked = cv2.dilate(blocked.astype(np.uint8), np.ones((9,9),np.uint8)) > 0
    if road.shape != (int(K["H"]),int(K["W"])) or blocked.shape != road.shape:
        raise RuntimeError("calibration/mask dimensions do not match")

    print("=== 1. 真实车辆检测 ===")
    real_veh, bg = get_real_vehicles(BG, K)
    print("  真实车辆: %d" % len(real_veh))
    for v in real_veh[:3]:
        print("    dist=%.1fm world=(%.1f, %.1f)" % (
            v["dist"], v["wx"], v["wy"]))

    # 真实车辆深度图（统一距离）
    veh_mask = np.any(obj_raw != 0, axis=2) if obj_raw.ndim == 3 else obj_raw != 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        veh_mask.astype(np.uint8), 8)
    real_depth_map = np.full(bg.shape[:2], np.nan, np.float32)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 500:
            continue
        inst = (lab == i)
        ys, xs = np.where(inst)
        ym = int(ys.max())
        foot = np.zeros_like(inst)
        sel = (ys >= ym - 3) & (ys <= ym)
        foot[ys[sel], xs[sel]] = True
        from cv_to_blender_cam import (build_camera_from_ground_pose,
                                       estimate_instance_depth_from_ground)
        Knp = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]],
                        [0, 0, 1]], dtype=np.float64)
        Rcb = np.asarray(K["R_cam2body"], dtype=float)
        tcb = np.asarray(K["t_cam2body"], dtype=float)
        T = np.eye(4); T[:3,:3] = Rcb.T; T[:3,3] = -Rcb.T @ tcb
        d, _ = estimate_instance_depth_from_ground(foot, Knp, T)
        if d is None:
            continue
        vv = d[np.isfinite(d) & (d > 0)]
        if vv.size == 0:
            continue
        real_depth_map[inst] = float(np.median(vv))

    print()
    print("=== 2. 批量生成 %d 帧 ===" % args.n)
    classes = ["car", "suv", "van", "truck"]
    results = []
    tried = rejected = 0

    while len(results) < args.n and tried < args.n * 6:
        tried += 1
        cls = classes[len(results) % len(classes)]
        glb, lwh = MODELS[cls]

        yaw = round(rng.uniform(-20, 20) + 180, 1)
        pose = sample_pose(real_veh, rng, K, road, blocked, lwh, yaw)
        if pose is None:
            rejected += 1
            continue
        d = os.path.join(args.out, "f%02d_%s" % (len(results), cls))
        ok = render_one(cls, glb, lwh, pose, yaw, d)
        if not ok:
            rejected += 1
            print("  [%d] %s render FAIL" % (tried, cls))
            continue

        fg = cv2.imread(os.path.join(d, "fg_rgba.png"),
                        cv2.IMREAD_UNCHANGED)
        depth = np.load(os.path.join(d, "depth.npy"))
        if fg is None:
            rejected += 1
            continue

        shadow_rgba = cv2.imread(os.path.join(d, "vehicle_with_shadow.png"), cv2.IMREAD_UNCHANGED)
        try:
            meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
            contacts = meta.get("wheel_contacts") or []
            # Models may include a spare wheel above the ground.
            contacts = [c for c in contacts if len(c.get("world") or []) == 3
                        and abs(c["world"][2]) <= .03]
            if len(contacts) < 4:
                raise ValueError("fewer than four supporting wheels")
            for contact in contacts:
                point = contact.get("pixel")
                world = contact.get("world") or []
                if point is None or len(world) != 3 or abs(world[2]) > .03:
                    raise ValueError("wheel is not on modeled ground")
                u, v = map(lambda z: int(round(z)), point)
                if not (0 <= v < road.shape[0] and 0 <= u < road.shape[1] and road[v,u] and not blocked[v,u]):
                    raise ValueError("wheel contact not on visible road")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rejected += 1
            print("  contact rejected", cls, str(exc), flush=True)
            continue
        comp, ratio, occ_px = composite_with_occlusion(
            fg, depth, bg, veh_mask, real_depth_map, shadow_rgba=shadow_rgba)

        # 可见性检查：车身像素宽度
        _a = fg[:, :, 3] > 0
        if _a.sum() == 0:
            rejected += 1
            continue
        _ys, _xs = np.where(_a)
        bbox_w = int(_xs.max() - _xs.min() + 1)
        bbox_h = int(_ys.max() - _ys.min() + 1)
        if bbox_w < MIN_BBOX_W:
            rejected += 1
            print("  [%d] %s pose=(%.1f,%.1f) 车宽仅 %dpx < %d 丢弃" % (
                tried, cls, pose[0], pose[1], bbox_w, MIN_BBOX_W))
            continue

        if ratio < MIN_VISIBLE_RATIO:
            rejected += 1
            print("  [%d] %s pose=(%.1f,%.1f) 可见率 %.0f%% < %.0f%% 丢弃" % (
                tried, cls, pose[0], pose[1], ratio * 100,
                MIN_VISIBLE_RATIO * 100))
            continue

        cv2.imwrite(os.path.join(d, "composite.jpg"), comp,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        results.append({"cls": cls, "pose": pose, "yaw": yaw,
                        "bbox": [int(_xs.min()), int(_ys.min()),
                                 bbox_w, bbox_h],
                        "visible_ratio": round(ratio, 3),
                        "occluded_px": occ_px,
                        "dir": d})
        print("  [%d] OK %-6s pose=(%5.1f,%5.1f) yaw=%6.1f bbox=%dx%d 可见 %.0f%% 遮挡 %dpx" % (
            len(results), cls, pose[0], pose[1], yaw,
            bbox_w, bbox_h, ratio * 100, occ_px))

    print()
    print("=== 3. 结果 ===")
    print("成功 %d / 尝试 %d / 丢弃 %d" % (len(results), tried, rejected))
    with open(os.path.join(args.out, "_index.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n": len(results), "training_eligible": False,
                           "warning": "2D road mask and assumed ground plane; no verified ground truth",
                           "items": results}, f,
                  ensure_ascii=False, indent=2)
    if not results:
        raise SystemExit("没有通过车轮接地和道路检查的结果")


if __name__ == "__main__":
    main()
