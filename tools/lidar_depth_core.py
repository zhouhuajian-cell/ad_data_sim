#!/usr/bin/env python3
"""LiDAR 深度图 + 实例去噪 + 可见性裁剪（3 条硬约束落地）。

用法：
    from lidar_depth_core import (undistort_frame, load_points,
                                  fit_ground_plane, points_to_depth,
                                  instance_depth_map, visible_bbox)
"""
import json
import math
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np


# ============================================================
# ① 去畸变
# ============================================================
def undistort_frame(img, K, dist, keep_K=True):
    """去畸变。

    UNDISTORT-KEEP-K: 默认保持原内参（此数据是 rational 8 参数模型，
      getOptimalNewCameraMatrix 会给出异常内参 fx!=fy，
      保持原 K 才能让 Blender 针孔相机严格对齐，有效区域 100%）。
    """
    h, w = img.shape[:2]
    if keep_K:
        und = cv2.undistort(img, K, dist, None, K)
        return und, K.copy(), (0, 0, w, h)
    K_new, roi = cv2.getOptimalNewCameraMatrix(
        K, dist, (w, h), 0, (w, h))
    und = cv2.undistort(img, K, dist, None, K_new)
    return und, K_new, roi


# ============================================================
# ② 点云加载
# ============================================================
def load_points(bin_path):
    """读 [n][x,y,z,i] float32 格式点云。"""
    raw = open(bin_path, "rb").read()
    n = int.from_bytes(raw[:4], "little")
    pts = np.frombuffer(raw[4:4 + n * 16], dtype=np.float32)
    return pts.reshape(n, 4)


# ============================================================
# ③ RANSAC 地面拟合（防悬空）
# ============================================================
def fit_ground_plane(xyz, x_range=(5, 25), y_range=(-3.5, 3.5),
                     iters=500, thr=0.15, seed=0):
    """在前方车道内拟合地面平面。

    Returns: dict(normal, offset, inlier_ratio, ground_z_at_15m)
    """
    # LOW-BAND-FIRST: 先用 Z 分位数筛出低点带，避免被货车立面淹没
    m = ((xyz[:, 0] > x_range[0]) & (xyz[:, 0] < x_range[1]) &
         (np.abs(xyz[:, 1]) < y_range[1]))
    P_all = xyz[m]
    if len(P_all) < 100:
        # 退化：扩到 3-40m
        m = ((xyz[:, 0] > 3) & (xyz[:, 0] < 40) &
             (np.abs(xyz[:, 1]) < 6))
        P_all = xyz[m]
        if len(P_all) < 100:
            return None
    # 取 Z 最低的 30% 作为地面候选
    z_lo = np.percentile(P_all[:, 2], 30)
    P = P_all[P_all[:, 2] <= z_lo]
    if len(P) < 60:
        P = P_all

    # ROBUST-LS: 低点带已经排除了绝大部分前景，此处再使用三点 RANSAC
    # 会在“许多平面同为 100% 内点”时保留随机抽到的第一个平面，导致同一
    # 帧地面高度随 seed 在几十厘米范围内跳变。改为确定性的稳健最小二乘，
    # 用 MAD 迭代剔除残余高点；seed/iters 仅为 API 兼容保留。
    keep = np.ones(len(P), dtype=bool)
    coef = None
    for _ in range(4):
        Q = P[keep]
        if len(Q) < 30:
            return None
        A = np.column_stack([Q[:, 0], Q[:, 1], np.ones(len(Q))])
        try:
            coef, *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)
        except Exception:
            return None
        pred = (P[:, 0] * coef[0] + P[:, 1] * coef[1] + coef[2])
        resid = P[:, 2] - pred
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        limit = max(0.03, min(float(thr), 3.5 * 1.4826 * mad))
        new_keep = np.abs(resid - med) <= limit
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if coef is None:
        return None
    a, b, c = [float(v) for v in coef]
    # z=a*x+b*y+c  <=>  a*x+b*y-z+c=0
    nrm = np.array([a, b, -1.0], dtype=float)
    nl = np.linalg.norm(nrm)
    nrm /= nl
    d0 = c / nl
    dist = np.abs(P @ nrm + d0)
    best_in = dist < thr

    z15 = None
    if abs(nrm[2]) > 1e-6:
        z15 = float(-(nrm[0] * 15.0 + nrm[1] * 0.0 + d0) / nrm[2])

    # 合理性：地面在 15m 处应在 [-1, 2] 米范围
    slope = float(np.hypot(a, b))
    ok = ((z15 is not None) and (-1.0 <= z15 <= 2.0) and
          slope <= 0.12 and best_in.sum() >= 30)

    return {"normal": [float(v) for v in nrm],
            "offset": float(d0),
            "inlier_ratio": float(best_in.sum() / len(P)),
            "ground_z_at_15m": z15,
            "n_points": int(len(P)),
            "fit": "robust_least_squares",
            "slope": slope,
            "sane": bool(ok)}


def ground_z(plane, x, y):
    """给定平面，求 (x,y) 处地面高度。"""
    n = plane["normal"]
    d0 = plane["offset"]
    if abs(n[2]) < 1e-6:
        return 0.0
    return float(-(n[0] * x + n[1] * y + d0) / n[2])


# ============================================================
# ④ 点云投影到图像
# ============================================================
def project_points(xyz, K, R, t):
    """LiDAR 系 -> 图像像素。返回 (uv, z_cam, valid_mask)"""
    Pc = xyz @ R.T + t
    z = Pc[:, 2]
    ok = z > 1e-3
    uv = np.full((len(xyz), 2), np.nan)
    if ok.any():
        uv[ok, 0] = K[0, 0] * Pc[ok, 0] / z[ok] + K[0, 2]
        uv[ok, 1] = K[1, 1] * Pc[ok, 1] / z[ok] + K[1, 2]
    return uv, z, ok


# ============================================================
# ⑤ 实例内深度图（含去噪）
# ============================================================
def instance_depth_map(uv, z_cam, ok, inst_mask, img_shape,
                       far_outlier_m=3.5, expand_px=2):
    """在实例掩码内生成逐像素深度图。

    约束 ②: 先按深度中位数剔除远背景穿透点。
    """
    h, w = img_shape[:2]
    depth = np.full((h, w), np.nan, np.float32)

    # 取落在掩码内且前后的点
    ui = np.round(uv[ok, 0]).astype(np.int32)
    vi = np.round(uv[ok, 1]).astype(np.int32)
    zz = z_cam[ok]
    inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, zz = ui[inb], vi[inb], zz[inb]
    inm = inst_mask[vi, ui]
    ui, vi, zz = ui[inm], vi[inm], zz[inm]
    if len(zz) < 10:
        return depth, {"n_pts": int(len(zz)), "n_valid": 0}

    # ② 去噪：剔除远背景穿透点
    med = float(np.median(zz))
    keep = zz < (med + far_outlier_m)
    n_out = int((~keep).sum())
    ui, vi, zz = ui[keep], vi[keep], zz[keep]
    if len(zz) < 10:
        return depth, {"n_pts": int(len(zz)), "n_valid": 0,
                       "n_outlier": n_out}

    # 掩码内最近邻插值
    depth[vi, ui] = zz
    # 用距离变换填充空洞
    known = ~np.isnan(depth)
    if known.sum() > 0:
        # 只在掩码内传播
        mask_in = inst_mask.copy()
        if expand_px > 0:
            k = np.ones((2 * expand_px + 1, 2 * expand_px + 1), np.uint8)
            mask_in = cv2.dilate(inst_mask.astype(np.uint8), k) > 0
        dist, idx = cv2.distanceTransformWithLabels(
            (~known).astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL)
        # 最近已知点索引
        ky, kx = np.where(known)
        idx_map = np.zeros((h, w), dtype=np.int32)
        idx_map[ky, kx] = np.arange(len(ky))
        labels = idx[idx]
        depth_fill = depth.flat[depth.flat >= 0]  # placeholder
        # 简化：直接对掩码内空洞做散点插值
        pts_known = np.column_stack([kx, ky])
        vals = depth[ky, kx]
        hole = mask_in & ~known
        if hole.any():
            hy, hx = np.where(hole)
            # 最近邻
            from scipy.spatial import cKDTree
            tree = cKDTree(pts_known)
            dmin, jmin = tree.query(np.column_stack([hx, hy]), k=1)
            ok_d = dmin <= max(6, expand_px * 4)
            depth[hy[ok_d], hx[ok_d]] = vals[jmin[ok_d]]

    n_valid = int((~np.isnan(depth)).sum())
    return depth, {"n_pts": int(len(zz)), "n_valid": n_valid,
                   "n_outlier": n_out,
                   "median_m": round(med, 2)}


# ============================================================
# ⑥ 可见性裁剪 + 2D 框（约束 ③）
# ============================================================
def visible_bbox(fg_rgba, real_depth, synth_depth, tol=0.5,
                 min_vis=0.15):
    """用真实深度裁掉被遮挡部分，返回可见 mask 与外接框。

    fg_rgba:        虚拟车 RGBA
    real_depth:     真实场景深度 (H,W)，NaN=未知
    synth_depth:    虚拟车深度 (H,W)
    """
    h, w = fg_rgba.shape[:2]
    alpha = fg_rgba[:, :, 3] > 8
    sd = np.asarray(synth_depth, np.float32)
    rd = np.asarray(real_depth, np.float32)

    sd_ok = np.isfinite(sd) & (sd > 0.01)
    rd_ok = np.isfinite(rd) & (rd > 0.01)

    # 被真实前景挡住：真实更近
    occluded = alpha & sd_ok & rd_ok & (rd < sd - tol)
    visible = alpha & ~occluded

    n_alpha = int(alpha.sum())
    n_vis = int(visible.sum())
    ratio = n_vis / max(n_alpha, 1)

    if n_vis < 10:
        return None, visible, ratio

    ys, xs = np.where(visible)
    bbox = [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    return bbox, visible, ratio


def bbox3d_to_bbox2d(corners_px, img_shape, min_area=16):
    """3D 框 8 角点 -> 紧致 2D 框（仅用于对比，不用于标注）。"""
    if corners_px is None:
        return None
    h, w = img_shape[:2]
    xs = corners_px[:, 0]
    ys = corners_px[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    x0 = max(0.0, min(x0, w - 1))
    y0 = max(0.0, min(y0, h - 1))
    x1 = max(0.0, min(x1, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    bw, bh = x1 - x0, y1 - y0
    if bw * bh < min_area:
        return None
    return [round(x0, 1), round(y0, 1), round(bw, 1), round(bh, 1)]
