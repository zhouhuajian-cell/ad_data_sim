#!/usr/bin/env python3
"""自动检测 GLB 的正确朝向：测试 6 种轴向假设，选"最像车"的那个。

判据（综合打分）：
  1. 底部应有轮子/支撑（alpha 分布底部宽）
  2. 长宽比：车长/车高 应在 1.2~4.5（轿车 3.0、货车 2.4）
  3. 水平对称性：车宽方向应左右对称
  4. 侧视轮廓面积占比合理

用法:
    blender --background --python detect_glb_orientation.py -- <glb路径>
输出: /tmp/glb_orient/<name>.json
"""
import bpy
import glob
import json
import math
import os
import sys
from mathutils import Vector

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
GLB = argv[0] if argv else None
OUT_DIR = "/tmp/glb_orient"
os.makedirs(OUT_DIR, exist_ok=True)

# 6 种候选旋转 (rx, ry, rz) 度
CANDS = [
    ("identity", 0, 0, 0),
    ("rx90", 90, 0, 0),
    ("rx180", 180, 0, 0),
    ("rx270", 270, 0, 0),
    ("rz90", 0, 0, 90),
    ("rz270", 0, 0, 270),
]


def bbox_of(objs):
    mn = Vector((1e9,) * 3)
    mx = Vector((-1e9,) * 3)
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    return mn, mx


def setup(res=256):
    sc = bpy.context.scene
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 8
    sc.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dt in ("OPTIX", "CUDA"):
            prefs.compute_device_type = dt
            prefs.get_devices()
            if any(d.type != "CPU" for d in prefs.devices):
                for d in prefs.devices:
                    d.use = True
                sc.cycles.device = "GPU"
                break
    except Exception:
        pass
    # 灯 + 世界
    sd = bpy.data.lights.new("S", type="SUN")
    sd.energy = 4.0
    sun = bpy.data.objects.new("S", sd)
    sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.6, 0.65, 0.75, 1)
    bg.inputs[1].default_value = 1.2
    cd = bpy.data.cameras.new("C")
    cd.lens = 50
    cam = bpy.data.objects.new("C", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def render_from(cam, mn, mx, yaw_deg, pitch_deg, path):
    sc = bpy.context.scene
    center = (mn + mx) / 2
    rad = float((mx - mn).length) / 2
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    d = rad * 2.6
    pos = Vector((
        center.x + d * math.cos(pitch) * math.cos(yaw),
        center.y + d * math.cos(pitch) * math.sin(yaw),
        center.z + d * math.sin(pitch),
    ))
    cam.location = pos
    cam.rotation_euler = (center - pos).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def analyze_alpha(png_path):
    """读 alpha 通道做统计（不依赖 PIL，用 Blender 的 image）。"""
    img = bpy.data.images.load(png_path)
    w, h = img.size
    import numpy as np
    px = np.array(list(img.pixels), dtype=np.float32).reshape(h, w, 4)
    a = np.flipud(px[:, :, 3])          # 上到下
    bpy.data.images.remove(img)
    mask = a > 0.5
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    bh = int(ys.max() - ys.min() + 1)
    bw = int(xs.max() - xs.min() + 1)
    # 底部 20% 行的宽度
    y0 = int(ys.min()); y1 = int(ys.max())
    bot_lo = int(y1 - bh * 0.20)
    bottom_w = 0
    for yy in range(bot_lo, y1 + 1):
        row = np.where(mask[yy])[0]
        if len(row):
            bottom_w = max(bottom_w, int(row.max() - row.min() + 1))
    # 顶部 20%
    top_hi = int(y0 + bh * 0.20)
    top_w = 0
    for yy in range(y0, top_hi + 1):
        row = np.where(mask[yy])[0]
        if len(row):
            top_w = max(top_w, int(row.max() - row.min() + 1))
    return {"bw": bw, "bh": bh, "fill": float(mask.mean()),
            "bottom_w": bottom_w, "top_w": top_w,
            "aspect": bw / max(bh, 1)}


def main():
    if not GLB or not os.path.isfile(GLB):
        print("[ORIENT] no glb")
        return
    name = os.path.basename(os.path.dirname(GLB))
    results = []

    for tag, rx, ry, rz in CANDS:
        try:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            cam = setup()
            bpy.ops.import_scene.gltf(filepath=GLB)
            ms = [o for o in bpy.context.scene.objects if o.type == "MESH"]
            if not ms:
                results.append({"cand": tag, "ok": False, "why": "no_mesh"})
                continue

            # 应用候选旋转
            root = bpy.data.objects.new("R", None)
            bpy.context.scene.collection.objects.link(root)
            for o in ms:
                o.parent = root
            root.rotation_euler = (math.radians(rx), math.radians(ry),
                                   math.radians(rz))
            bpy.context.view_layer.update()

            mn, mx = bbox_of(ms)
            ext = mx - mn
            if min(ext) < 1e-6:
                results.append({"cand": tag, "ok": False, "why": "degenerate"})
                continue

            # 侧视（yaw=0，pitch=-5）
            p1 = "/tmp/_o_side.png"
            render_from(cam, mn, mx, 0.0, -5.0, p1)
            side = analyze_alpha(p1)
            # 正视（yaw=90）
            p2 = "/tmp/_o_front.png"
            render_from(cam, mn, mx, 90.0, -5.0, p2)
            front = analyze_alpha(p2)

            if side is None or front is None:
                results.append({"cand": tag, "ok": False, "why": "empty"})
                continue

            # 打分：侧视长宽比、正视长宽比、底部宽度
            L_ratio = max(ext.x, ext.y) / max(ext.z, 1e-6)   # 水平/竖直
            score = 0.0
            # 侧视 aspect 期望 1.5~4.0（长/高）
            sa = side["aspect"]
            if 1.4 <= sa <= 4.5:
                score += 2.0
            elif 1.0 <= sa <= 6.0:
                score += 1.0
            # 正视 aspect 期望 0.9~2.2（宽/高）
            fa = front["aspect"]
            if 0.8 <= fa <= 2.4:
                score += 2.0
            elif 0.5 <= fa <= 3.5:
                score += 1.0
            # 底部应比顶部宽（轮子撑开）
            if side["bottom_w"] >= side["top_w"] * 1.0:
                score += 1.0
            # 水平轴应明显长于竖直轴
            if 1.2 <= L_ratio <= 5.0:
                score += 1.5
            elif 0.8 <= L_ratio <= 7.0:
                score += 0.5

            results.append({
                "cand": tag, "ok": True, "score": round(score, 2),
                "ext": [round(ext.x, 3), round(ext.y, 3), round(ext.z, 3)],
                "side": side, "front": front,
                "L_ratio": round(L_ratio, 2),
            })
        except Exception as e:
            results.append({"cand": tag, "ok": False, "why": str(e)[:60]})

    ok_list = [r for r in results if r.get("ok")]
    ok_list.sort(key=lambda r: -r["score"])
    best = ok_list[0] if ok_list else None

    out = {"glb": GLB, "name": name, "results": results,
           "best": best["cand"] if best else None,
           "best_score": best["score"] if best else None}
    fp = os.path.join(OUT_DIR, name + ".json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[ORIENT] %-40s best=%s score=%s" % (
        name, out["best"], out["best_score"]))
    for r in results:
        if r.get("ok"):
            print("    %-10s score=%.2f ext=%s L=%.2f side_a=%.2f front_a=%.2f" % (
                r["cand"], r["score"], r["ext"], r["L_ratio"],
                r["side"]["aspect"], r["front"]["aspect"]))
        else:
            print("    %-10s FAIL %s" % (r["cand"], r.get("why")))


main()
