#!/usr/bin/env python3
"""MVE：GLB 模型按相机位姿渲染成 RGBA 贴片，贴到真实底图上做对照。

验证内容：
  1. 模型尺度是否正确（归一化到真实 L/W/H）
  2. 相机对齐（俯仰角、透视）
  3. 输出可复用为 2D 贴片（保持与现有管线兼容）

用法:
    python render_glb_mve.py --glb <path> --out <outdir>
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh


def load_normalized(glb_path, target_LWH=None):
    """加载 GLB，归一化到车体坐标：
    - 原点在底部中心（Z=0 为地面）
    - 车头朝 +X（或按模型最长轴）
    - 缩放到真实尺寸（米）
    """
    scene = trimesh.load(glb_path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)

    # 合并所有几何体
    meshes = []
    for g in scene.geometry.values():
        if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0:
            meshes.append(g)
    if not meshes:
        raise RuntimeError("no mesh")
    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    # 原始包围盒
    bb = merged.bounds
    ext = bb[1] - bb[0]
    cx, cy, cz = (bb[0] + bb[1]) / 2

    # 平移到底部中心
    merged.apply_translation([-cx, -cy, -bb[0][2]])

    # 确定长轴：最长维度当作车长方向
    ext2 = merged.bounds[1] - merged.bounds[0]
    if ext2[1] > ext2[0]:      # Y 更长 -> 旋转到 X
        merged.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 0, 1]))
    if ext2[2] > max(ext2[0], ext2[1]):   # Z 最长 -> 可能是竖着的，转到 X
        merged.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0]))

    # 缩放到真实尺寸
    if target_LWH:
        L, W, H = target_LWH
        cur = merged.bounds[1] - merged.bounds[0]
        # cur = (x_len, y_len, z_len)
        sx = L / max(cur[0], 1e-6)
        sy = W / max(cur[1], 1e-6)
        sz = H / max(cur[2], 1e-6)
        merged.apply_scale([sx, sy, sz])
    else:
        # 默认按常见轿车 4.5 x 1.8 x 1.5
        L, W, H = 4.5, 1.8, 1.5
        cur = merged.bounds[1] - merged.bounds[0]
        merged.apply_scale([L / max(cur[0], 1e-6),
                            W / max(cur[1], 1e-6),
                            H / max(cur[2], 1e-6)])

    # 重新落地
    merged.apply_translation([0, 0, -merged.bounds[0][2]])
    return merged


def render_view(mesh, yaw_deg, pitch_deg, dist_scale=2.2, res=512):
    """按 yaw/pitch 渲染一张 RGBA。"""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.55] * 3)

    tm = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(tm)

    # 目标点 = 包围盒中心
    bb = mesh.bounds
    center = (bb[0] + bb[1]) / 2
    radius = float(np.linalg.norm(bb[1] - bb[0])) / 2

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    d = radius * dist_scale

    # 相机位置（球坐标）
    cam_pos = np.array([
        center[0] + d * np.cos(pitch) * np.cos(yaw),
        center[1] + d * np.cos(pitch) * np.sin(yaw),
        center[2] + d * np.sin(pitch) + radius * 0.15,
    ])

    fwd = center - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -fwd
    pose[:3, 3] = cam_pos

    cam = pyrender.PerspectiveCamera(yfov=np.radians(35), aspectRatio=1.0)
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0),
              pose=pose)
    # 补光
    fill_pose = pose.copy()
    fill_pose[:3, 3] = cam_pos + np.array([d * 0.5, 0, d * 0.3])
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.2),
              pose=fill_pose)

    r = pyrender.OffscreenRenderer(res, res)
    color, depth = r.render(scene)
    r.delete()

    alpha = (depth > 0).astype(np.uint8) * 255
    rgba = np.dstack([cv2.cvtColor(color, cv2.COLOR_RGB2BGR), alpha])
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", default="/tmp/mve_out")
    ap.add_argument("--lwh", default="4.5,1.8,1.5",
                    help="目标尺寸 长,宽,高（米）")
    ap.add_argument("--views", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    LWH = [float(x) for x in args.lwh.split(",")]

    mesh = load_normalized(args.glb, LWH)
    bb = mesh.bounds
    print(f"归一化后尺寸: X={bb[1][0]-bb[0][0]:.2f} "
          f"Y={bb[1][1]-bb[0][1]:.2f} Z={bb[1][2]-bb[0][2]:.2f} 米")
    print(f"底部 Z: {bb[0][2]:.4f} (应为 0)")

    # 环形 yaw，中等俯仰（车载相机约 -5~-15°）
    results = []
    for i in range(args.views):
        yaw = i * (360.0 / args.views)
        for pitch in (-8.0,):
            rgba = render_view(mesh, yaw, pitch)
            fn = f"view_yaw{int(yaw):03d}_pitch{int(pitch)}.png"
            fp = os.path.join(args.out, fn)
            cv2.imwrite(fp, rgba)
            nz = (rgba[:, :, 3] > 0).sum()
            results.append({"file": fn, "yaw": yaw, "pitch": pitch,
                            "px": int(nz)})
            print(f"  {fn}  不透明像素 {nz}")

    json.dump({"glb": args.glb, "LWH": LWH, "views": results},
              open(os.path.join(args.out, "meta.json"), "w"),
              ensure_ascii=False, indent=2)
    print("完成:", args.out)


if __name__ == "__main__":
    main()
