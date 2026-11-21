#!/usr/bin/env python3
"""用 pyrender 重新渲染精灵图（修正 alpha 和颜色）。

问题：
  上一版 alpha 用 depth>0 二值化，边缘抗锯齿丢失；
  RGB 是黑背景，半透明处合成会发黑。

修法：
  - 渲染两次：黑背景 + 白背景
  - alpha = 1 - (black_result / white_result) 的反推
  - 或者更简单：用一个「纯色背景」渲染，再做色度抠图
"""
import json
import math
import os

import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender
import trimesh

OUT = "/data/obj_assets"
RES = 768

CLASS_SIZE = {
    "car": (4.30, 1.80, 1.50),
    "suv": (4.50, 1.90, 1.75),
    "van": (5.00, 2.00, 2.20),
    "truck": (5.20, 2.00, 2.30),
    "truck_flat": (6.50, 2.40, 3.00),
    "pedestrian": (0.50, 0.55, 1.70),
    "rider": (1.00, 0.70, 1.75),
    "cone": (0.30, 0.30, 0.70),
}

VIEWS = {"front": (0, 5), "rear": (180, 5), "side_left": (90, 5)}
LIGHTS = {"day": (1.6, [1.0, 0.98, 0.94]),
          "night": (0.5, [0.55, 0.65, 0.85])}

# 更真实的颜色（BGR 在 cv2 里，这里是 RGB）
COLORS = {
    "car": ([0.82, 0.82, 0.84], [0.24, 0.26, 0.32]),
    "suv": ([0.30, 0.34, 0.40], [0.12, 0.14, 0.18]),
    "van": ([0.92, 0.92, 0.93], [0.28, 0.28, 0.32]),
    "truck": ([0.78, 0.30, 0.24], [0.24, 0.10, 0.08]),
    "truck_flat": ([0.55, 0.48, 0.40], [0.18, 0.15, 0.13]),
    "pedestrian": ([0.28, 0.36, 0.58], [0.10, 0.13, 0.22]),
    "rider": ([0.62, 0.42, 0.28], [0.20, 0.14, 0.10]),
    "cone": ([0.98, 0.40, 0.08], [0.70, 0.26, 0.06]),
}


def make_car_mesh(L, W, H):
    parts = []
    body = trimesh.creation.box(extents=[L * 0.98, W, H * 0.55])
    body.apply_translation([0, 0, H * 0.32])
    parts.append(body)
    roof = trimesh.creation.box(extents=[L * 0.55, W * 0.88, H * 0.42])
    roof.apply_translation([-L * 0.04, 0, H * 0.72])
    parts.append(roof)
    r = H * 0.20
    for dx in (L * 0.32, -L * 0.32):
        for dy in (W * 0.48, -W * 0.48):
            w = trimesh.creation.cylinder(radius=r, height=W * 0.10,
                                          sections=20)
            w.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi / 2, [1, 0, 0]))
            w.apply_translation([dx, dy, r])
            parts.append(w)
    return trimesh.util.concatenate(parts)


def make_truck_mesh(L, W, H):
    parts = []
    cab = trimesh.creation.box(extents=[L * 0.30, W, H * 0.78])
    cab.apply_translation([L * 0.34, 0, H * 0.39])
    parts.append(cab)
    box = trimesh.creation.box(extents=[L * 0.66, W, H])
    box.apply_translation([-L * 0.16, 0, H * 0.52])
    parts.append(box)
    r = H * 0.13
    for dx in (L * 0.36, -L * 0.08, -L * 0.38):
        for dy in (W * 0.48, -W * 0.48):
            w = trimesh.creation.cylinder(radius=r, height=W * 0.10,
                                          sections=20)
            w.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi / 2, [1, 0, 0]))
            w.apply_translation([dx, dy, r])
            parts.append(w)
    return trimesh.util.concatenate(parts)


def make_cone_mesh(L, W, H):
    cone = trimesh.creation.cone(radius=W * 0.45, height=H * 0.92,
                                 sections=32)
    cone.apply_translation([0, 0, H * 0.46])
    base = trimesh.creation.box(extents=[W, W, H * 0.08])
    base.apply_translation([0, 0, H * 0.04])
    return trimesh.util.concatenate([cone, base])


def make_ped_mesh(L, W, H):
    parts = []
    head = trimesh.creation.icosphere(radius=H * 0.075, subdivisions=2)
    head.apply_translation([0, 0, H * 0.93])
    parts.append(head)
    torso = trimesh.creation.box(extents=[W * 0.9, W * 0.55, H * 0.42])
    torso.apply_translation([0, 0, H * 0.63])
    parts.append(torso)
    for dy in (W * 0.18, -W * 0.18):
        leg = trimesh.creation.box(extents=[W * 0.35, W * 0.30, H * 0.42])
        leg.apply_translation([0, dy, H * 0.21])
        parts.append(leg)
    for dy in (W * 0.52, -W * 0.52):
        arm = trimesh.creation.box(extents=[W * 0.28, W * 0.22, H * 0.38])
        arm.apply_translation([0, dy, H * 0.65])
        parts.append(arm)
    return trimesh.util.concatenate(parts)


BUILDERS = {
    "car": make_car_mesh, "suv": make_car_mesh, "van": make_truck_mesh,
    "truck": make_truck_mesh, "truck_flat": make_truck_mesh,
    "pedestrian": make_ped_mesh, "rider": make_ped_mesh,
    "cone": make_cone_mesh,
}


def render_rgba(mesh, cls, view, light):
    """渲染成 RGBA：用「黑背景 + 白背景」反推真 alpha。

    原理：结果 = fg*a + bg*(1-a)
      black = fg*a            (bg=0)
      white = fg*a + 1*(1-a)  (bg=1)
      → 1 - (white - black) = a
      → fg = black / a
    """
    L, W, H = CLASS_SIZE[cls]
    ci = 0 if light == "day" else 1
    col = [COLORS[cls][ci][0], COLORS[cls][ci][1], COLORS[cls][ci][2], 1.0]

    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=col, metallicFactor=0.10, roughnessFactor=0.65)
    pm = pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True)

    az, el = VIEWS[view]
    fov = math.radians(32)
    dist = max(L, H) * 0.5 / (0.80 * math.tan(fov / 2))
    ar, er = math.radians(az), math.radians(el)
    eye = np.array([-math.cos(ar) * dist * math.cos(er),
                    -math.sin(ar) * dist * math.cos(er),
                    H * 0.5 + dist * math.sin(er)], float)
    target = np.array([0, 0, H * 0.5], float)
    up = np.array([0, 0, 1.0], float)

    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    zc = -fwd
    xc = np.cross(up, zc)
    xc /= max(np.linalg.norm(xc), 1e-9)
    yc = np.cross(zc, xc)

    pose = np.eye(4)
    pose[:3, 0] = xc
    pose[:3, 1] = yc
    pose[:3, 2] = zc
    pose[:3, 3] = eye

    inten, rgb = LIGHTS[light]

    def one(bg):
        sc = pyrender.Scene(bg_color=bg, ambient_light=[0.28] * 3)
        sc.add(pm)
        cam = pyrender.PerspectiveCamera(yfov=fov * 2, aspectRatio=1.0)
        sc.add(cam, pose=pose)
        key = pyrender.DirectionalLight(color=rgb, intensity=inten)
        kp = np.eye(4)
        kp[:3, 3] = eye
        kp[:3, :3] = pose[:3, :3]
        sc.add(key, pose=kp)
        fill = pyrender.DirectionalLight(color=[0.72, 0.78, 0.92],
                                         intensity=inten * 0.30)
        fp = np.eye(4)
        fp[:3, 3] = [-eye[0], -eye[1], eye[2]]
        sc.add(fill, pose=fp)
        r = pyrender.OffscreenRenderer(RES, RES)
        c, d = r.render(sc)
        r.delete()
        return c.astype(np.float32), d

    blk, depth = one([0.0, 0.0, 0.0, 1.0])
    wht, _ = one([1.0, 1.0, 1.0, 1.0])

    # alpha = 1 - (white - black)
    a = 1.0 - (wht - blk)
    a = np.clip(a.mean(axis=2), 0, 1)

    # 前景色 = black / a
    fg = np.zeros_like(blk)
    nz = a > 1e-3
    fg[nz] = blk[nz] / a[nz][:, None]

    # 硬边：depth>0
    solid = depth > 0
    a = np.where(solid, np.maximum(a, 0.55), 0.0)
    a = np.clip(a, 0, 1)

    # 裁剪
    ys, xs = np.where(a > 0.02)
    if len(ys) == 0:
        return None
    y0, y1 = max(0, ys.min() - 3), min(RES - 1, ys.max() + 3)
    x0, x1 = max(0, xs.min() - 3), min(RES - 1, xs.max() + 3)

    rgb_c = np.clip(fg[y0:y1 + 1, x0:x1 + 1], 0, 255).astype(np.uint8)
    a_c = (a[y0:y1 + 1, x0:x1 + 1] * 255).astype(np.uint8)
    return np.dstack([rgb_c, a_c])


def main():
    made = 0
    for cls in CLASS_SIZE:
        b = BUILDERS.get(cls)
        if not b:
            continue
        L, W, H = CLASS_SIZE[cls]
        mesh = b(L, W, H)
        d = os.path.join(OUT, cls)
        os.makedirs(d, exist_ok=True)
        n = 0
        for view in VIEWS:
            for light in ("day", "night"):
                rgba = render_rgba(mesh, cls, view, light)
                if rgba is None:
                    print("  %s/%s/%s 空" % (cls, view, light))
                    continue
                name = view + ("" if light == "day" else "_" + light) + ".png"
                Image.fromarray(rgba, mode="RGBA").save(os.path.join(d, name))
                n += 1
        print("  %-12s %d 张" % (cls, n))
        made += n

        meta = {
            "class": cls, "variant": "procedural",
            "size_m": {"width": W, "height": H, "length": L},
            "anchor": "bottom_center",
            "views": list(VIEWS.keys()), "lighting": ["day", "night"],
            "source": "procedural-pbr", "license": "internal",
        }
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    print("\n共 %d 张" % made)


if __name__ == "__main__":
    main()
