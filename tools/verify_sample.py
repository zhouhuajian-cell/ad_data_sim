#!/usr/bin/env python3
"""验证样本素材：尺寸、朝向、接地、透明边缘。

不依赖未知的相机参数，直接读 GLB 几何做客观检查。
"""
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import trimesh

ROOT = "/data/assets_global/OBJ_SAMPLE"

# 类别 -> 典型真实尺寸（米）
TYPICAL = {
    "car": (4.50, 1.80, 1.50),
    "suv": (4.70, 1.90, 1.75),
    "truck": (7.00, 2.30, 2.90),
    "van": (5.20, 1.95, 2.10),
}


def analyze_glb(path):
    scene = trimesh.load(path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    meshes = []
    for name, g in scene.geometry.items():
        m = g.copy()
        try:
            T = scene.graph.get(name)[0]
            m.apply_transform(T)
        except Exception:
            pass
        meshes.append(m)
    if not meshes:
        return None
    mesh = trimesh.util.concatenate(meshes)
    bb = mesh.bounds
    ext = bb[1] - bb[0]
    # 长宽高排序
    srt = np.sort(ext)
    return {
        "ext_raw": [round(float(v), 4) for v in ext],
        "ext_sorted": [round(float(v), 4) for v in srt],
        "min_z": round(float(bb[0][2]), 4),
        "center": [round(float(v), 4) for v in ((bb[0] + bb[1]) / 2)],
        "n_faces": int(len(mesh.faces)),
        "aspect": round(float(srt[2] / max(srt[0], 1e-6)), 2),
    }


def analyze_view(png_path):
    im = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    a = im[:, :, 3]
    ys, xs = np.where(a > 127)
    if len(ys) == 0:
        return {"empty": True}
    bh = int(ys.max() - ys.min() + 1)
    bw = int(xs.max() - xs.min() + 1)
    # 底部 5 行内是否有 alpha（判断是否贴边裁切）
    h = a.shape[0]
    bottom_touch = bool((a[max(0, h - 3):, :] > 127).any())
    top_touch = bool((a[:3, :] > 127).any())
    return {
        "bbox_w": bw, "bbox_h": bh,
        "fill_ratio": round(float((a > 127).mean()), 3),
        "aspect_wh": round(bw / max(bh, 1), 2),
        "touch_bottom": bottom_touch,
        "touch_top": top_touch,
    }


def main():
    man = json.load(open(os.path.join(ROOT, "_manifest.json")))
    items = man["items"]

    print(f"检查 {len(items)} 个模型\n")
    issues = []

    for it in items:
        uid, cat = it["uid"], it["category"]
        d = os.path.join(ROOT, cat, uid)
        glb = os.path.join(d, "mesh.glb")

        g = analyze_glb(glb) if os.path.isfile(glb) else None
        if g is None:
            issues.append((uid, "GLB 读取失败"))
            continue

        # 长宽比检查：车应该长 > 宽
        s = g["ext_sorted"]
        long_short = s[2] / max(s[0], 1e-6)
        # 车类长宽比应在 1.5~5 之间
        if cat in ("car", "suv", "truck", "van"):
            if not (1.3 <= long_short <= 6.0):
                issues.append((uid, f"{cat} 长宽比异常 {long_short:.2f}"))

        # 视角图检查
        vstats = []
        for v in it["views"][:4]:
            vp = os.path.join(d, v)
            if os.path.isfile(vp):
                vs = analyze_view(vp)
                if vs:
                    vstats.append(vs)
                    if vs.get("touch_top"):
                        issues.append((uid, f"{v} 顶部贴边（可能被裁切）"))

        if g["n_faces"] < 100:
            issues.append((uid, f"面数过少 {g['n_faces']}"))

        typical = TYPICAL.get(cat, (4.5, 1.8, 1.5))
        print(f"[{cat:6s}] {uid[:12]}  "
              f"raw={g['ext_raw']}  "
              f"长宽比={long_short:.2f}  "
              f"面数={g['n_faces']:6d}  "
              f"视角={len(vstats)}")

    print(f"\n=== 问题汇总 ===")
    if issues:
        for uid, msg in issues[:20]:
            print(f"  {uid[:12]}: {msg}")
        print(f"  共 {len(issues)} 条")
    else:
        print("  无")


if __name__ == "__main__":
    main()
