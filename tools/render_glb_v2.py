#!/usr/bin/env python3
"""GLB → 2D RGBA 贴片渲染（轴向正确版）。

Kenney/glTF 常见轴向：
  - glTF 规范：Y 向上，-Z 向前
  - Kenney car-kit：Y 向上，车头沿 +Z 或 +X（需实测）

本脚本：
  1. 检测模型实际轴向（最长水平轴 = 车长）
  2. 统一转到 UE 风格：X 向前（车长）、Y 向左、Z 向上
  3. 底部贴地 Z=0
  4. 按目标真实尺寸（米）归一化
  5. 按 yaw/pitch 渲染 RGBA
"""
import argparse
import json
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh


def normalize_mesh(glb_path, target_LWH=(4.5, 1.8, 1.5), up_axis="Y"):
    """加载并归一化到交通目标坐标系（X=车长，Y=车宽，Z=高，底部 Z=0）。"""
    scene = trimesh.load(glb_path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)

    # 合并（保留贴图）
    geos = [g for g in scene.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    if not geos:
        raise RuntimeError("no mesh")
    # 先应用各自的 transform 再合并
    merged_list = []
    for name, g in scene.geometry.items():
        m = g.copy()
        try:
            T = scene.graph.get(name)[0]
            m.apply_transform(T)
        except Exception:
            pass
        merged_list.append(m)
    mesh = trimesh.util.concatenate(merged_list)

    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f"原始尺寸(XYZ): {ext[0]:.2f} {ext[1]:.2f} {ext[2]:.2f}")

    # ---- 轴向归一：找到"高度轴"（最小跨度之外，通常是 up）----
    if up_axis == "Y":
        # glTF: Y 向上 → 转成 Z 向上
        # rot X +90°: (x,y,z) -> (x, -z, y)
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh.apply_transform(R)
    elif up_axis == "Z":
        pass
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f"转 Z-up 后:   {ext[0]:.2f} {ext[1]:.2f} {ext[2]:.2f}")

    # ---- 车长轴：X/Y 中的较长者 → 转到 X ----
    if ext[1] > ext[0]:
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        mesh.apply_transform(R)
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(f"转 X长轴后:   {ext[0]:.2f} {ext[1]:.2f} {ext[2]:.2f}")

    # ---- 缩放（注意：先缩放再落地）----
    L, W, H = target_LWH
    cur = mesh.bounds[1] - mesh.bounds[0]
    mesh.apply_scale([L / max(cur[0], 1e-6),
                      W / max(cur[1], 1e-6),
                      H / max(cur[2], 1e-6)])

    # ---- 落地 + 居中 ----
    bb = mesh.bounds
    mesh.apply_translation([-(bb[0][0] + bb[1][0]) / 2,
                            -(bb[0][1] + bb[1][1]) / 2,
                            -bb[0][2]])
    bb = mesh.bounds
    print(f"归一化后:     {bb[1][0]-bb[0][0]:.2f} "
          f"{bb[1][1]-bb[0][1]:.2f} {bb[1][2]-bb[0][2]:.2f}  底Z={bb[0][2]:.4f}")
    return mesh


def render_view(mesh, yaw_deg, pitch_deg, res=512, fov_deg=35.0,
                dist_scale=2.0):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.7] * 3)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    bb = mesh.bounds
    center = (bb[0] + bb[1]) / 2
    radius = float(np.linalg.norm(bb[1] - bb[0])) / 2

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    d = radius * dist_scale

    cam_pos = np.array([
        center[0] + d * np.cos(pitch) * np.cos(yaw),
        center[1] + d * np.cos(pitch) * np.sin(yaw),
        center[2] + d * np.sin(pitch) + radius * 0.1,
    ])

    fwd = center - cam_pos
    fwd /= np.linalg.norm(fwd)
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

    cam = pyrender.PerspectiveCamera(yfov=np.radians(fov_deg), aspectRatio=1.0)
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.2),
              pose=pose)
    fill = pose.copy()
    fill[:3, 3] = cam_pos + np.array([d * 0.6, -d * 0.3, d * 0.4])
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.0),
              pose=fill)

    r = pyrender.OffscreenRenderer(res, res)
    color, depth = r.render(scene)
    r.delete()

    alpha = (depth > 0).astype(np.uint8) * 255
    return np.dstack([cv2.cvtColor(color, cv2.COLOR_RGB2BGR), alpha])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", default="/tmp/glb_out")
    ap.add_argument("--lwh", default="4.5,1.8,1.5")
    ap.add_argument("--up", default="Y", choices=["Y", "Z"])
    ap.add_argument("--views", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    LWH = [float(x) for x in args.lwh.split(",")]
    mesh = normalize_mesh(args.glb, LWH, args.up)

    for i in range(args.views):
        yaw = i * (360.0 / args.views)
        rgba = render_view(mesh, yaw, -8.0)
        cv2.imwrite(os.path.join(args.out, f"yaw{int(yaw):03d}.png"), rgba)
    print("完成:", args.out)


if __name__ == "__main__":
    main()
