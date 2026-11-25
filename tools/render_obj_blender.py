"""Blender Headless 渲染：GLB -> RGBA + 遮罩 + 深度 + 3D 框。

用法：
    blender --background --python render_obj_blender.py -- \
        --glb /path/model.glb --out /tmp/blender_out

关键设计：
  1. Cycles GPU（OPTIX/CUDA），失败自动回落 CPU
  2. Shadow Catcher 地面接收车底软阴影
  3. film_transparent 输出透明底
  4. 同时输出 depth（用 Z pass）+ 3D 框 8 顶点投影
"""
import bpy
import bmesh
import json
import math
import os
import sys
from mathutils import Vector, Matrix

# ---------------- 参数解析 ----------------
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []


def arg(name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


GLB = arg("--glb", "/data/assets_global/OBJ_3D/KENNEY/sedan.glb")
OUT = arg("--out", "/tmp/blender_out")
TARGET_LWH = [float(x) for x in arg("--lwh", "4.5,1.8,1.5").split(",")]
RES = int(arg("--res", "640"))
SAMPLES = int(arg("--samples", "32"))

os.makedirs(OUT, exist_ok=True)
LOG = "/tmp/blender_report.txt"
report = []


def log(s):
    report.append(str(s))
    print("[BLENDER] %s" % s)


# ---------------- 清场 ----------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---------------- 导入 GLB ----------------
log("importing %s" % GLB)
bpy.ops.import_scene.gltf(filepath=GLB)
meshes = [o for o in scene.objects if o.type == "MESH"]
log("imported meshes: %d -> %s" % (len(meshes), [o.name for o in meshes][:6]))
if not meshes:
    log("FATAL: no mesh")
    open(LOG, "w", encoding="utf-8").write("\n".join(report) + "\n")
    sys.exit(1)

# ---------------- 规范化：底部 Z=0，居中，缩放到目标尺寸 ----------------
# 世界包围盒
mn = Vector((1e9, 1e9, 1e9))
mx = Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for c in o.bound_box:
        wco = o.matrix_world @ Vector(c)
        mn.x, mn.y, mn.z = min(mn.x, wco.x), min(mn.y, wco.y), min(mn.z, wco.z)
        mx.x, mx.y, mx.z = max(mx.x, wco.x), max(mx.y, wco.y), max(mx.z, wco.z)

ext = mx - mn
log("raw bounds: %.3f %.3f %.3f" % (ext.x, ext.y, ext.z))

L, W, H = TARGET_LWH
# glTF: Y up. Blender import 已转为 Z up。
# 找最长水平轴作为车长
if ext.y > ext.x:
    # 绕 Z 转 90 度让 Y 变 X
    for o in meshes:
        o.rotation_mode = "XYZ"
        o.rotation_euler.z += math.radians(90)
    bpy.context.view_layer.update()
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    for o in meshes:
        for c in o.bound_box:
            wco = o.matrix_world @ Vector(c)
            mn.x = min(mn.x, wco.x); mn.y = min(mn.y, wco.y); mn.z = min(mn.z, wco.z)
            mx.x = max(mx.x, wco.x); mx.y = max(mx.y, wco.y); mx.z = max(mx.z, wco.z)
    ext = mx - mn
    log("after rotate: %.3f %.3f %.3f" % (ext.x, ext.y, ext.z))

# 缩放
sx = L / max(ext.x, 1e-6)
sy = W / max(ext.y, 1e-6)
sz = H / max(ext.z, 1e-6)

# 空父级做统一变换
parent = bpy.data.objects.new("CarRoot", None)
scene.collection.objects.link(parent)
for o in meshes:
    o.parent = parent
parent.scale = (sx, sy, sz)
bpy.context.view_layer.update()

# 重新计算世界包围盒并落地
mn = Vector((1e9, 1e9, 1e9)); mx = Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for c in o.bound_box:
        wco = o.matrix_world @ Vector(c)
        mn.x = min(mn.x, wco.x); mn.y = min(mn.y, wco.y); mn.z = min(mn.z, wco.z)
        mx.x = max(mx.x, wco.x); mx.y = max(mx.y, wco.y); mx.z = max(mx.z, wco.z)

parent.location.x -= (mn.x + mx.x) / 2
parent.location.y -= (mn.y + mx.y) / 2
parent.location.z -= mn.z
bpy.context.view_layer.update()

# 最终包围盒（用于 3D box）
mn = Vector((1e9, 1e9, 1e9)); mx = Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for c in o.bound_box:
        wco = o.matrix_world @ Vector(c)
        mn.x = min(mn.x, wco.x); mn.y = min(mn.y, wco.y); mn.z = min(mn.z, wco.z)
        mx.x = max(mx.x, wco.x); mx.y = max(mx.y, wco.y); mx.z = max(mx.z, wco.z)
final_ext = mx - mn
log("final bounds: L=%.3f W=%.3f H=%.3f  min_z=%.4f"
    % (final_ext.x, final_ext.y, final_ext.z, mn.z))

# ---------------- 地面 Shadow Catcher ----------------
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "ShadowCatcher"
try:
    ground.is_shadow_catcher = True
    log("shadow catcher enabled")
except Exception as e:
    log("shadow catcher EXC %s" % e)

# ---------------- 灯光 ----------------
# 太阳光
sun_data = bpy.data.lights.new("Sun", type="SUN")
sun_data.energy = 3.0
sun_data.angle = math.radians(2.0)
sun = bpy.data.objects.new("Sun", sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = (math.radians(50), 0, math.radians(40))

# 环境光（世界背景，模拟天空）
world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.55, 0.65, 0.8, 1.0)   # 偏蓝天空
bg.inputs[1].default_value = 1.2
log("lights set")

# ---------------- 相机（车载视角：高 1.5m，俯角 -6°，看向车尾）----------------
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = 35
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

cam_h = 1.5
dist = 12.0
cam.location = (-dist, 0.0, cam_h)
# 看向 (0,0,0.75)
target = Vector((0, 0, 0.75))
direction = target - cam.location
cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
log("camera at (%.1f, %.1f, %.1f)" % tuple(cam.location))

# ---------------- 渲染设置 ----------------
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

scene.render.engine = "CYCLES"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True

# GPU 优先，失败回落 CPU
gpu_ok = False
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            has_gpu = any(d.type != "CPU" for d in prefs.devices)
            if has_gpu:
                for d in prefs.devices:
                    d.use = True
                scene.cycles.device = "GPU"
                gpu_ok = True
                log("GPU mode: %s" % dev_type)
                break
        except Exception:
            continue
except Exception as e:
    log("gpu setup EXC %s" % e)
if not gpu_ok:
    scene.cycles.device = "CPU"
    log("fallback CPU mode")

# 深度通道
scene.view_layers[0].use_pass_z = True
vl = scene.view_layers[0]
vl.use_pass_mist = False

# ---------------- 渲染 ----------------
rgb_path = os.path.join(OUT, "render_rgba.png")
scene.render.filepath = rgb_path
log("rendering...")
import time
t0 = time.time()
bpy.ops.render.render(write_still=True)
log("render done in %.1fs" % (time.time() - t0))

depth_stats = {}
# Depth compositor: preserve floating point values without color normalization.
try:
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    socket = rl.outputs.get("Depth") or rl.outputs.get("Z")
    if socket is None:
        raise RuntimeError("Depth pass unavailable; outputs=" + repr([x.name for x in rl.outputs]))
    output = nt.nodes.new("CompositorNodeOutputFile")
    output.base_path = OUT
    output.file_slots[0].path = "depth_"
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    output.format.color_depth = "32"
    nt.links.new(socket, output.inputs[0])
    bpy.ops.render.render(write_still=False)
    depth_path = os.path.join(OUT, "depth_%04d.exr" % scene.frame_current)
    if not os.path.isfile(depth_path):
        raise RuntimeError("depth output missing: " + depth_path)
    depth_stats = {"file": os.path.basename(depth_path), "format": "OpenEXR float32", "note": "Depth pass includes visible shadow-catcher geometry; validate camera depth convention before use."}
    log("depth written: " + depth_path)
except Exception as e:
    log("depth EXC %s" % e)
    raise

# ---------------- 3D 框投影 ----------------
try:
    from bpy_extras.object_utils import world_to_camera_view
    corners = []
    for dx in (mn.x, mx.x):
        for dy in (mn.y, mx.y):
            for dz in (mn.z, mx.z):
                corners.append(Vector((dx, dy, dz)))
    proj = []
    for c in corners:
        co = world_to_camera_view(scene, cam, c)
        proj.append({
            "world": [round(c.x, 3), round(c.y, 3), round(c.z, 3)],
            "uv": [round(co.x, 4), round(co.y, 4)],
            "pix": [round(co.x * RES, 1), round((1 - co.y) * RES, 1)],
            "depth": round(co.z, 3),
        })
    log("3D box projected, 8 corners")
except Exception as e:
    proj = []
    log("project EXC %s" % e)

# ---------------- 元数据 ----------------
meta = {
    "glb": GLB,
    "output_dir": OUT,
    "engine": scene.render.engine,
    "device": scene.cycles.device,
    "samples": SAMPLES,
    "resolution": [RES, RES],
    "film_transparent": True,
    "target_dimensions_m": {"L": L, "W": W, "H": H},
    "final_dimensions_m": {
        "L": round(final_ext.x, 4),
        "W": round(final_ext.y, 4),
        "H": round(final_ext.z, 4),
    },
    "ground_z": 0.0,
    "bbox_min": [round(mn.x, 4), round(mn.y, 4), round(mn.z, 4)],
    "bbox_max": [round(mx.x, 4), round(mx.y, 4), round(mx.z, 4)],
    "camera": {
        "location": [round(cam.location.x, 3),
                     round(cam.location.y, 3),
                     round(cam.location.z, 3)],
        "lens_mm": cam_data.lens,
        "rotation_euler": [round(math.degrees(r), 2)
                           for r in cam.rotation_euler],
    },
    "box3d_corners": proj,
    "depth_stats": depth_stats,
}
with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

open(LOG, "w", encoding="utf-8").write("\n".join(report) + "\n")
log("ALL DONE")
