#!/usr/bin/env python3
"""为 raw_selected 的模型渲染统一视角缩略图，并拼成看板。

相机固定：+X/-Y/+Z 方向看向原点（左前上方 3/4 视）
地面画一根红色 +X 方向箭头（指示"车头应指向"的参考方向）

用法:
    blender --background --python board_render.py -- /data/assets_global/OBJ_3D/raw_selected /tmp/board
输出:
    <out>/thumbs/<name>.png     单个缩略图
    <out>/board_1.jpg ...       看板（5x4）
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
SRC = argv[0] if argv else "/data/assets_global/OBJ_3D/raw_selected"
OUT = argv[1] if len(argv) > 1 else "/tmp/board"

CELL = 400
COLS = 5
ROWS = 4


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x = CELL
    sc.render.resolution_y = CELL
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    try:
        p = bpy.context.preferences.addons["cycles"].preferences
        p.compute_device_type = "OPTIX"
        p.get_devices()
        for d in p.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception:
        pass
    # 世界
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.62, 0.68, 0.80, 1)
    bg.inputs[1].default_value = 1.6
    return sc


def add_ground_and_arrow(radius):
    """灰色地面 + 红色 +X 箭头。"""
    bpy.ops.mesh.primitive_plane_add(size=radius * 8, location=(0, 0, 0))
    g = bpy.context.object
    g.name = "Ground"
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.45, 0.45, 0.46, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    g.data.materials.append(mat)

    # 箭头：一根长条 + 一个锥
    L = radius * 1.6
    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(L * 0.35, 0, 0.005))
    bar = bpy.context.object
    bar.scale = (L * 0.7, radius * 0.04, 0.005)
    bpy.ops.mesh.primitive_cone_add(
        vertices=4, radius1=radius * 0.14, depth=radius * 0.30,
        location=(L * 0.72, 0, radius * 0.02))
    cone = bpy.context.object
    cone.rotation_euler = (0, math.radians(90), math.radians(45))

    amat = bpy.data.materials.new("ArrowMat")
    amat.use_nodes = True
    ab = amat.node_tree.nodes["Principled BSDF"]
    ab.inputs["Base Color"].default_value = (0.85, 0.05, 0.05, 1)
    ab.inputs["Roughness"].default_value = 0.5
    for o in (bar, cone):
        o.data.materials.append(amat)
    return g, bar, cone


def add_lights(radius):
    def area(name, loc, energy, size):
        d = bpy.data.lights.new(name, type="AREA")
        d.energy = energy
        d.size = size
        o = bpy.data.objects.new(name, d)
        bpy.context.scene.collection.objects.link(o)
        o.location = loc
        o.rotation_euler = (Vector((0, 0, 0.5)) - Vector(loc)) \
            .to_track_quat("-Z", "Y").to_euler()
    r = radius
    area("key", (r * 2.0, -r * 2.2, r * 2.4), 2500 * max(r, 1), r * 3)
    area("fill", (-r * 2.2, -r * 1.4, r * 1.2), 900 * max(r, 1), r * 3)
    area("rim", (-r * 0.8, r * 2.4, r * 1.4), 1200 * max(r, 1), r * 3)


def place_camera(radius):
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("Cam")
    cd.lens = 55
    cam = bpy.data.objects.new("Cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    # 固定左前上 3/4 视角
    d = radius * 2.9
    ya = math.radians(-35)     # 从 -Y 侧看
    el = math.radians(22)
    cam.location = Vector((d * math.cos(el) * math.cos(ya),
                           d * math.cos(el) * math.sin(ya),
                           d * math.sin(el) + radius * 0.15))
    cam.rotation_euler = (Vector((0, 0, radius * 0.25)) - cam.location) \
        .to_track_quat("-Z", "Y").to_euler()
    return cam


def render_one(src_dir, out_png):
    """渲染单个模型。返回 (ok, ext, 面数)。"""
    glb = os.path.join(src_dir, "mesh.glb")
    if not os.path.isfile(glb):
        return False, None, 0
    sc = setup_scene()
    try:
        bpy.ops.import_scene.gltf(filepath=glb)
    except Exception as e:
        print("[B] import fail %s: %s" % (glb, str(e)[:60]))
        return False, None, 0

    bpy.ops.object.select_all(action="SELECT")
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True,
                                       scale=True)
    except Exception:
        pass
    for o in list(bpy.context.scene.objects):
        if o.type == "MESH" and o.parent:
            o.matrix_world = o.matrix_world.copy()
            o.parent = None
    bpy.context.view_layer.update()

    ms = [o for o in sc.objects if o.type == "MESH"]
    if not ms:
        return False, None, 0
    ntri = sum(len(o.data.polygons) for o in ms)

    mn = Vector((1e9,) * 3)
    mx = Vector((-1e9,) * 3)
    for o in ms:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    ext = mx - mn
    cen = (mn + mx) / 2.0
    maxd = max(ext.x, ext.y, ext.z)
    if maxd < 1e-6:
        return False, None, ntri

    # 归一：最长边 = 4.0，底部 z=0
    s = 4.0 / maxd
    for o in ms:
        o.location = (o.location - cen) * s
        o.scale = (o.scale.x * s, o.scale.y * s, o.scale.z * s)
    bpy.context.view_layer.update()
    mn2 = Vector((1e9,) * 3)
    mx2 = Vector((-1e9,) * 3)
    for o in ms:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mn2[i] = min(mn2[i], w[i])
                mx2[i] = max(mx2[i], w[i])
    dz = -mn2.z
    for o in ms:
        o.location.z += dz
    bpy.context.view_layer.update()

    # 重新计算中心与半径供相机使用
    mn3 = Vector((1e9,) * 3)
    mx3 = Vector((-1e9,) * 3)
    for o in ms:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mn3[i] = min(mn3[i], w[i])
                mx3[i] = max(mx3[i], w[i])
    radius = max((mx3 - mn3).x, (mx3 - mn3).y, (mx3 - mn3).z) / 2.0

    add_ground_and_arrow(radius)
    add_lights(radius)
    place_camera(radius)

    sc.render.filepath = out_png
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        print("[B] render fail: %s" % str(e)[:60])
        return False, None, ntri
    return os.path.isfile(out_png), [round(v, 3) for v in
                                     (mx3 - mn3)], ntri


def main():
    os.makedirs(OUT, exist_ok=True)
    th_dir = os.path.join(OUT, "thumbs")
    os.makedirs(th_dir, exist_ok=True)

    dirs = sorted([d for d in glob.glob(os.path.join(SRC, "*"))
                   if os.path.isdir(d)])
    print("[B] 待渲染 %d 个模型" % len(dirs))

    results = []
    for i, d in enumerate(dirs):
        name = os.path.basename(d)
        png = os.path.join(th_dir, name + ".png")
        ok, ext, ntri = render_one(d, png)
        results.append({"name": name, "ok": bool(ok),
                        "ext": ext, "tris": ntri, "png": png})
        print("[B] %02d/%02d %-24s ok=%s tris=%d ext=%s" % (
            i + 1, len(dirs), name, ok, ntri, ext))

    with open(os.path.join(OUT, "_thumbs.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("[B] done")


main()
