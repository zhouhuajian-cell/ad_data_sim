#!/usr/bin/env python3
"""3D 点云盒体防穿模校验（替换 OBB-OBB 与 2D 像素校验）。

判据：
  ① 提取离地 0.3~3.0m 的非地面 LiDAR 点
  ② 虚拟车框扩展（长 +1.0m、宽 +0.6m）
  ③ 落入盒体内的非地面点数 < 15  -> 通过
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")


def extract_obstacle_points(xyz, ground_plane,
                            z_lo=0.3, z_hi=3.0,
                            x_range=(-5, 80), y_range=(-15, 15)):
    """提取障碍物点云（离地 0.3~3.0m）。

    Args:
        xyz: (N,3) LiDAR 系点云
        ground_plane: fit_ground_plane() 返回的平面 dict
    Returns:
        (M,3) 障碍物点
    """
    if ground_plane is None:
        return np.zeros((0, 3), np.float32)

    n = ground_plane["normal"]
    d0 = ground_plane["offset"]

    # DIST-SIGN-FIX: 法向朝下（n[2]<0），离地高度需取负
    signed = xyz @ np.array(n) + d0
    if n[2] < 0:
        height = -signed
    else:
        height = signed

    m = ((height > z_lo) & (height < z_hi) &
         (xyz[:, 0] > x_range[0]) & (xyz[:, 0] < x_range[1]) &
         (np.abs(xyz[:, 1]) < y_range[1]))
    return xyz[m]


def count_points_in_box(pts, cx, cy, cz, L, W, H, yaw,
                        expand_long=1.0, expand_lat=0.6):
    """数落入 3D OBB 盒体内的点数。

    盒体在 (cx,cy) 处、尺寸 (L+1.0, W+0.6, H)，绕 z 轴转 yaw。
    高度方向从 cz 到 cz+H。
    """
    if len(pts) == 0:
        return 0

    hL = (L + expand_long) / 2.0
    hW = (W + expand_lat) / 2.0

    # 转到盒体局部系
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    dz = pts[:, 2] - cz

    c, s = math.cos(-yaw), math.sin(-yaw)
    lx = dx * c - dy * s
    ly = dx * s + dy * c

    inside = ((np.abs(lx) <= hL) & (np.abs(ly) <= hW) &
              (dz >= -0.2) & (dz <= H + 0.2))
    return int(inside.sum())


def check_placement_cloud(pts_obst, virt, ground_plane,
                          expand_long=1.0, expand_lat=0.6,
                          max_pts=15):
    """点云盒体校验：3D 无穿模则通过。

    Args:
        pts_obst: 障碍物点云 (M,3)
        virt: dict(x, y, z, yaw, size=[L,W,H])
        ground_plane: 地面平面
    Returns:
        (ok, info)
    """
    x, y = float(virt["x"]), float(virt["y"])
    L, W, H = virt["size"][0], virt["size"][1], virt["size"][2]
    yaw = math.radians(float(virt.get("yaw", 0)))

    # 地面高度
    if ground_plane:
        n = ground_plane["normal"]
        d0 = ground_plane["offset"]
        if abs(n[2]) > 1e-6:
            gz = -(n[0] * x + n[1] * y + d0) / n[2]
        else:
            gz = 0.0
    else:
        gz = 0.0

    n_in = count_points_in_box(pts_obst, x, y, gz, L, W, H, yaw,
                               expand_long, expand_lat)
    ok = n_in < max_pts
    return ok, {"n_points_in_box": n_in,
                "threshold": max_pts,
                "reason": "ok" if ok else "cloud_collision"}


def check_road_union(footprint_mask, road_mask, veh_mask,
                     min_ratio=0.85):
    """2D 路面校验：road_mask | vehicle_mask 并集。

    允许车轮投影落在真车掩码后（被遮挡是合法的）。
    """
    union = road_mask | veh_mask
    n = int(footprint_mask.sum())
    if n < 30:
        return False, 0.0
    ratio = float((footprint_mask & union).sum()) / n
    return ratio >= min_ratio, ratio
