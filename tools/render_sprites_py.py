#!/usr/bin/env python3
"""用 pyrender 离线渲染 Kenney 模型为 RGBA 精灵图。

优势：
  - 完全离线，不依赖 UE headless 渲染
  - 可精确控制视角/光照
  - 输出真正的 RGBA PNG

流程：
  1. 从 UE 导出 Kenney 模型的 OBJ（用 Python 直接读 uasset 做不到）
     → 改从 Kenney FBX/GLB 源文件读（需要先找到或下载）

  2. 用 trimesh 加载 → pyrender 渲染 → alpha 用「无背景 v.s. 有背景」差分

当前限制：如果拿不到 Kenney 源模型，就用程序化建模生成近似几何。
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender
import trimesh

OUT = "/data/obj_assets"

# 真实尺寸（长, 宽, 高）米
CLASS_SIZE = {
    "car": (4.30, 1.80, 1.50),
    "suv": (4.50, 1.90, 1.75),
    "truck": (5.20, 2.00, 2.30),
    "truck_flat": (6.50, 2.40, 3.00),
    "van": (5.00, 2.00, 2.20),
    "pedestrian": (0.50, 0.55, 1.70),
    "rider": (1.00, 0.70, 1.75),
    "cone": (0.30, 0.30, 0.70),
}

VIEWS = {
    "front": (0, 5),
    "rear": (180, 5),
    "side_left": (90, 5),
}

LIGHTS = {
    "day": (3.0, [1.0, 0.98, 0.94]),
    "night": (0.8, [0.55, 0.65, 0.85]),
}


def make_car_mesh(L, W, H):
    """程序化轿车几何（车身+车顶+4轮）。"""
    parts = []
    # 车身（下半）
    body = trimesh.creation.box(extents=[L * 0.98, W, H * 0.55])
    body.apply_translation([0, 0, H * 0.32])
    parts.append(body)
    # 车顶（上半，缩窄）
    roof = trimesh.creation.box(extents=[L * 0.55, W * 0.88, H * 0.42])
    roof.apply_translation([-L * 0.04, 0, H * 0.72])
    parts.append(roof)
    # 4 轮
    r = H * 0.20
    for dx in (L * 0.32, -L * 0.32):
        for dy in (W * 0.48, -W * 0.48):
            w = trimesh.creation.cylinder(radius=r, height=W * 0.10,
                                          sections=16)
            w.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi / 2, [1, 0, 0]))
            w.apply_translation([dx, dy, r])
            parts.append(w)
    return trimesh.util.concatenate(parts)


def make_truck_mesh(L, W, H):
    parts = []
    # 驾驶室
    cab = trimesh.creation.box(extents=[L * 0.30, W, H * 0.78])
    cab.apply_translation([L * 0.34, 0, H * 0.39])
    parts.append(cab)
    # 货箱
    box = trimesh.creation.box(extents=[L * 0.66, W, H])
    box.apply_translation([-L * 0.16, 0, H * 0.52])
    parts.append(box)
    # 轮
    r = H * 0.13
    for dx in (L * 0.36, -L * 0.08, -L * 0.38):
        for dy in (W * 0.48, -W * 0.48):
            w = trimesh.creation.cylinder(radius=r, height=W * 0.10,
                                          sections=16)
            w.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi / 2, [1, 0, 0]))
            w.apply_translation([dx, dy, r])
            parts.append(w)
    return trimesh.util.concatenate(parts)


def make_cone_mesh(L, W, H):
    """锥桶：圆锥 + 底座。"""
    cone = trimesh.creation.cone(radius=W * 0.45, height=H * 0.92,
                                 sections=24)
    cone.apply_translation([0, 0, H * 0.46])
    base = trimesh.creation.box(extents=[W, W, H * 0.08])
    base.apply_translation([0, 0, H * 0.04])
    return trimesh.util.concatenate([cone, base])


def make_pedestrian_mesh(L, W, H):
    parts = []
    # 头
    head = trimesh.creation.icosphere(radius=H * 0.075)
    head.apply_translation([0, 0, H * 0.93])
    parts.append(head)
    # 躯干
    torso = trimesh.creation.box(extents=[W * 0.9, W * 0.55, H * 0.42])
    torso.apply_translation([0, 0, H * 0.63])
    parts.append(torso)
    # 腿
    for dy in (W * 0.18, -W * 0.18):
        leg = trimesh.creation.box(extents=[W * 0.35, W * 0.30, H * 0.42])
        leg.apply_translation([0, dy, H * 0.21])
        parts.append(leg)
    # 手臂
    for dy in (W * 0.52, -W * 0.52):
        arm = trimesh.creation.box(extents=[W * 0.28, W * 0.22, H * 0.38])
        arm.apply_translation([0, dy, H * 0.65])
        parts.append(arm)
    return trimesh.util.concatenate(parts)


MESH_BUILDERS = {
    "car": make_car_mesh,
    "suv": make_car_mesh,
    "van": make_truck_mesh,
    "truck": make_truck_mesh,
    "truck_flat": make_truck_mesh,
    "pedestrian": make_pedestrian_mesh,
    "rider": make_pedestrian_mesh,
    "cone": make_cone_mesh,
}

# 材质颜色（day, night）
COLORS = {
    "car": ([0.78, 0.78, 0.80, 1.0], [0.20, 0.22, 0.28, 1.0]),
    "suv": ([0.55, 0.58, 0.62, 1.0], [0.16, 0.18, 0.22, 1.0]),
    "truck": ([0.72, 0.72, 0.74, 1.0], [0.20, 0.20, 0.24, 1.0]),
    "truck_flat": ([0.62, 0.55, 0.48, 1.0], [0.18, 0.16, 0.15, 1.0]),
    "van": ([0.85, 0.85, 0.86, 1.0], [0.22, 0.22, 0.26, 1.0]),
    "pedestrian": ([0.35, 0.42, 0.62, 1.0], [0.14, 0.16, 0.24, 1.0]),
    "rider": ([0.45, 0.35, 0.30, 1.0], [0.16, 0.13, 0.12, 1.0]),
    "cone": ([0.98, 0.42, 0.10, 1.0], [0.70, 0.28, 0.08, 1.0]),
}

RES = 512


def render_one(mesh, cls, view, light, out_path):
    """渲染单个视角 + 光照 → RGBA PNG。"""
    L, W, H = CLASS_SIZE[cls]

    # 材质
    col = COLORS[cls][0 if light == "day" else 1]
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=col,
        metallic=0.15, roughness=0.55,
    )
    pm = pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35] * 3)
    scene.add(pm)

    # 相机：按视角摆放
    az, el = VIEWS[view]
    fov = math.radians(32)
    dist = max(L, H) * 1.35 / math.tan(fov / 2)
    ar = math.radians(az)
    er = math.radians(el)
    cx = -math.cos(ar) * dist * math.cos(er)
    cy = -math.sin(ar) * dist * math.cos(er)
    cz = H * 0.5 + dist * math.sin(er)

    eye = np.array([cx, cy, cz])
    target = np.array([0, 0, H * 0.5])
    up = np.array([0, 0, 1.0])

    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye

    cam = pyrender.PerspectiveCamera(yfov=fov * 2, aspectRatio=1.0)
    scene.add(cam, pose=pose)

    inten, rgb = LIGHTS[light]
    key = pyrender.DirectionalLight(color=rgb, intensity=inten)
    kp = np.eye(4)
    kp[:3, 3] = eye
    kp[:3, :3] = pose[:3, :3]
    scene.add(key, pose=kp)

    fill = pyrender.DirectionalLight(color=[0.7, 0.75, 0.9],
                                     intensity=inten * 0.35)
    fp = np.eye(4)
    fp[:3, 3] = [-cx, -cy, cz]
    scene.add(fill, pose=fp)

    r = pyrender.OffscreenRenderer(RES, RES)
    color, depth = r.render(scene)
    r.delete()

    # alpha：depth > 0 的像素有物体
    alpha = (depth > 0).astype(np.uint8) * 255

    # 裁剪到内容边界，底部对齐
    ys, xs = np.where(alpha > 0)
    if len(ys) == 0:
        return False
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    m = 4
    y0 = max(0, y0 - m); y1 = min(RES - 1, y1 + m)
    x0 = max(0, x0 - m); x1 = min(RES - 1, x1 + m)

    rgb_c = color[y0:y1 + 1, x0:x1 + 1]
    a_c = alpha[y0:y1 + 1, x0:x1 + 1]

    rgba = np.dstack([rgb_c, a_c])
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    return True


def main():
    made = 0
    for cls in CLASS_SIZE:
        builder = MESH_BUILDERS.get(cls)
        if not builder:
            print("  %-12s 无建模函数" % cls)
            continue

        L, W, H = CLASS_SIZE[cls]
        mesh = builder(L, W, H)
        d = os.path.join(OUT, cls)
        os.makedirs(d, exist_ok=True)

        n = 0
        for view in VIEWS:
            for light in ("day", "night"):
                name = view + ("" if light == "day" else "_" + light) + ".png"
                op = os.path.join(d, name)
                if render_one(mesh, cls, view, light, op):
                    n += 1
        print("  %-12s %d 张  尺寸 %.2fx%.2fx%.2fm" % (cls, n, L, W, H))
        made += n

        meta = {
            "class": cls, "variant": "procedural",
            "size_m": {"width": W, "height": H, "length": L},
            "anchor": "bottom_center",
            "views": list(VIEWS.keys()),
            "lighting": ["day", "night"],
            "source": "procedural-pbr",
            "license": "internal",
            "_note": "\u7a0b\u5e8f\u5efa\u6a21 + PBR \u79bb\u7ebf\u6e32\u67d3",
        }
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    print("\n共 %d 张 → %s" % (made, OUT))


if __name__ == "__main__":
    main()
