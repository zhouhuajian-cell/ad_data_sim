"""Render a canonical GLB thumbnail for the asset library.

用法:
    blender -b --python render_thumb.py -- <glb> <out.png> [size]

视角：左前上方 3/4 视（与 board_render.py 一致）
"""
import math
import os
import sys

import bpy
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(ARGS) < 2:
    raise SystemExit("usage: render_thumb.py <glb> <out.png> [size]")

GLB, OUT = ARGS[0], ARGS[1]
SIZE = int(ARGS[2]) if len(ARGS) > 2 else 400


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    # 导入
    bpy.ops.import_scene.gltf(filepath=GLB)
    objs = [o for o in sc.objects if o.type == "MESH"]
    if not objs:
        raise SystemExit("no mesh")

    # 统一世界变换
    for ob in objs:
        m = ob.matrix_world.copy()
        ob.parent = None
        ob.matrix_world = m
    bpy.context.view_layer.update()

    # bbox
    mn = Vector((1e18,) * 3)
    mx = Vector((-1e18,) * 3)
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    ctr = (mn + mx) / 2.0
    radius = max((mx - mn).length / 2.0, 0.5)

    # 相机：左前上 3/4（-X 方向看车头，+Z 俯视）
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    d = radius * 3.0
    cam.location = (ctr.x - d * 0.75, ctr.y - d * 0.65, ctr.z + d * 0.55)
    # 看向中心
    direction = ctr - cam.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()
    cam_data.lens = 50

    # 灯光
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))

    fill_data = bpy.data.lights.new("Fill", type="AREA")
    fill_data.energy = 200
    fill_data.size = radius * 4
    fill = bpy.data.objects.new("Fill", fill_data)
    sc.collection.objects.link(fill)
    fill.location = (ctr.x + d * 0.5, ctr.y + d * 0.5, ctr.z + d * 0.6)
    fill.rotation_euler = (math.radians(60), 0, math.radians(-135))

    # 地面
    bpy.ops.mesh.primitive_plane_add(
        size=radius * 20, location=(ctr.x, ctr.y, mn.z - 0.001))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.45, 0.45, 0.48, 1)
        bsdf.inputs["Roughness"].default_value = 0.9
    plane.data.materials.append(mat)

    # 世界背景
    world = bpy.data.worlds.new("W")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.55, 0.58, 0.62, 1)
        bg.inputs[1].default_value = 1.0

    # 渲染设置
    sc.render.engine = "BLENDER_EEVEE_NEXT" \
        if hasattr(bpy.types, "RenderEngineEeveeNext") else "CYCLES"
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        sc.render.engine = "CYCLES"
        sc.cycles.samples = 32
        try:
            sc.cycles.device = "GPU"
        except Exception:
            pass
    sc.render.resolution_x = SIZE
    sc.render.resolution_y = SIZE
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    sc.render.filepath = OUT
    bpy.ops.render.render(write_still=True)

    print("[THUMB] %s  exists=%s" % (OUT, os.path.isfile(OUT)))


if __name__ == "__main__":
    main()
