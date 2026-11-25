"""真实道路场景对齐：车模按相机内外参渲染并 Alpha 叠回背景图。

关键点：
  1. OpenCV 相机系 (+Z前, +Y下) → Blender (-Z前, +Y上)：绕 X 轴 180° 翻转
     等价于旋转矩阵右乘 diag(1,-1,-1)
  2. 主点偏移用 camera.data.shift_x / shift_y
  3. Depth 存 OPEN_EXR 32-bit
  4. 输出：合成图 + 3D 框图 + 深度统计
"""
import bpy
import json
import math
import sys
import numpy as np
sys.path.insert(0, "/opt/ue5_synth/tools")
from cv_to_blender_cam import (set_blender_render_roi,
                               optimize_cycles_for_batch,
                               project_3d_box_opencv)
import os
from mathutils import Vector, Matrix

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []


def arg(name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


GLB = arg("--glb", "/data/assets_global/OBJ_3D/KENNEY/sedan.glb")
BG = arg("--bg")                     # 背景图路径
OUT = arg("--out", "/tmp/align_out")
LWH = [float(x) for x in arg("--lwh", "4.5,1.8,1.5").split(",")]
POSE = [float(x) for x in arg("--pose", "15,0,2.0").split(",")]  # x,y,height
YAW = float(arg("--yaw", "0"))
GROUND_Z = float(arg("--ground-z", "0"))
KFILE = arg("--k")                   # json: fx,fy,cx,cy,W,H
SAMPLES = int(arg("--samples", "48"))
HDRI_ENABLED = arg("--hdri", "0") == "1"
PAINT_COAT_ENABLED = arg("--paint-coat", "0") == "1"

os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "render_report.txt")
report = []


def log(s):
    report.append(str(s))
    print("[ALIGN] %s" % s)


def flush():
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")


# ---------------- 相机参数 ----------------
cam_h, pitch_deg = POSE[2], -6.0
K = {}
if KFILE and os.path.isfile(KFILE):
    K = json.load(open(KFILE))
fx = K.get("fx", 1319.71)
fy = K.get("fy", 1319.36)
cx = K.get("cx", 953.25)
cy = K.get("cy", 540.01)
W = int(K.get("W", 1920))
H = int(K.get("H", 1080))
log("K: fx=%.2f fy=%.2f cx=%.2f cy=%.2f  %dx%d" % (fx, fy, cx, cy, W, H))

# ---------------- 清场 + 导入 ----------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=GLB)
meshes = [o for o in scene.objects if o.type == "MESH"]
log("meshes: %d" % len(meshes))
if PAINT_COAT_ENABLED:
    for obj in meshes:
        for material in obj.data.materials:
            if not material or not material.use_nodes: continue
            name=(material.name or '').lower()
            if not any(k in name for k in ('carpaint','car_paint','car paint','body_paint')): continue
            node=next((n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
            if node and 'Coat Weight' in node.inputs and not node.inputs['Coat Weight'].is_linked:
                node.inputs['Coat Weight'].default_value=max(node.inputs['Coat Weight'].default_value,0.2)
                if 'Coat Roughness' in node.inputs and not node.inputs['Coat Roughness'].is_linked:
                    node.inputs['Coat Roughness'].default_value=0.18

# ---------------- 规范化 ----------------
def world_bbox(objs):
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mn.x = min(mn.x, w.x); mn.y = min(mn.y, w.y); mn.z = min(mn.z, w.z)
            mx.x = max(mx.x, w.x); mx.y = max(mx.y, w.y); mx.z = max(mx.z, w.z)
    return mn, mx


mn, mx = world_bbox(meshes)
ext = mx - mn
log("raw: %.3f %.3f %.3f" % (ext.x, ext.y, ext.z))
# Preserve every imported object's world transform while grouping it.
normalizer = bpy.data.objects.new("VehicleNormalization", None)
scene.collection.objects.link(normalizer)
for o in meshes:
    matrix_world = o.matrix_world.copy()
    o.parent = normalizer
    o.matrix_world = matrix_world
bpy.context.view_layer.update()
# Rotate the whole vehicle, not each wheel around its own origin.
if ext.y > ext.x:
    normalizer.rotation_euler.z = math.radians(90)
    bpy.context.view_layer.update()
    mn, mx = world_bbox(meshes)
    ext = mx - mn
factor = LWH[0] / max(ext.x, 1e-6)
normalizer.scale = (factor, factor, factor)
bpy.context.view_layer.update()
mn, mx = world_bbox(meshes)
normalizer.location = (-(mn.x + mx.x)/2, -(mn.y + mx.y)/2, -mn.z)
bpy.context.view_layer.update()
placement = bpy.data.objects.new("VehiclePlacement", None)
scene.collection.objects.link(placement)
normalizer.parent = placement
px, py, _ = POSE
placement.location = (px, py, GROUND_Z)
placement.rotation_euler.z = math.radians(YAW)
bpy.context.view_layer.update()
parent = placement
mn, mx = world_bbox(meshes)
fext = mx - mn
log("placed (%.2f,%.2f) yaw %.1f bounds=(%.2f,%.2f,%.2f) ground %.4f" %
    (px,py,YAW,fext.x,fext.y,fext.z,mn.z))
for o in meshes:
    if "wheel" in o.name.lower() or "tire" in o.name.lower():
        q0,q1=world_bbox([o]);log("WHEEL %s z=%.4f"%(o.name,q0.z))

# ---------------- 地面 Shadow Catcher ----------------
bpy.ops.mesh.primitive_plane_add(size=300, location=(0, 0, GROUND_Z))
ground = bpy.context.active_object
ground.is_shadow_catcher = True
# 关键：让地面不写入 Z pass，否则全屏深度都会被地面覆盖
try:
    ground.visible_camera = True
    ground.visible_diffuse = True
    ground.visible_glossy = True
    ground.visible_transmission = True
    ground.visible_volume_scatter = True
    ground.visible_shadow = True
    # Cycles：地面不参与 Z 深度写入（通过 holdout 语义已由 shadow catcher 提供）
    # 改用 layer collection 排除方式太复杂，这里用 hide_render 的替代：
    # 让地面只接收阴影，不写深度 -> 用 shadow catcher 的默认行为即可，
    # 但需要在合成时把地面深度剔除。简单做法：记录地面 mask。
except Exception as e:
    log("ground vis EXC %s" % e)
log("shadow catcher ok")

# 额外：单独渲染一次"只有车"的 Z pass（不含地面），作为可信深度
# 通过临时隐藏地面实现

# ---------------- 灯光 ----------------
sd = bpy.data.lights.new("Sun", type="SUN")
sd.energy = 3.5
sd.angle = math.radians(2.5)
sun = bpy.data.objects.new("Sun", sd)
scene.collection.objects.link(sun)
sun.rotation_euler = (math.radians(55), 0, math.radians(35))

world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.55, 0.65, 0.82, 1.0)
bg.inputs[1].default_value = 1.0
if HDRI_ENABLED:
    hdri='/data/assets_global/HDRI/urban_street_01_1k.hdr'
    if os.path.isfile(hdri):
        texture=world.node_tree.nodes.new('ShaderNodeTexEnvironment')
        texture.image=bpy.data.images.load(hdri,check_existing=True)
        world.node_tree.links.new(texture.outputs['Color'],bg.inputs['Color'])
        bg.inputs['Strength'].default_value=0.8
        log('HDRI enabled: urban_street_01')
    else: log('HDRI missing; using existing world')

# Isolated calibration test: exact body-to-camera extrinsic from source frame.
from cv_to_blender_cam import set_blender_camera
cd = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cd)
scene.collection.objects.link(cam)
scene.camera = cam
Rcb = np.asarray(K["R_cam2body"], dtype=np.float64)
tcb = np.asarray(K["t_cam2body"], dtype=np.float64)
T_w2c = np.eye(4)
T_w2c[:3,:3] = Rcb.T
T_w2c[:3,3] = -Rcb.T @ tcb
Knp = np.asarray(K["K"], dtype=np.float64)
set_blender_camera(scene, cam, Knp, T_w2c, W, H)
log("camera uses source-frame intrinsics and extrinsics")

# ---------------- 渲染设置 ----------------
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.engine = "CYCLES"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True

gpu_ok = False
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dt in ("OPTIX", "CUDA"):
        prefs.compute_device_type = dt
        prefs.get_devices()
        if any(d.type != "CPU" for d in prefs.devices):
            for d in prefs.devices:
                d.use = True
            scene.cycles.device = "GPU"
            gpu_ok = True
            log("GPU %s" % dt)
            break
except Exception as e:
    log("gpu EXC %s" % e)
if not gpu_ok:
    scene.cycles.device = "CPU"

# depth
scene.view_layers[0].use_pass_z = True
scene.use_nodes = True
nt = scene.node_tree
nt.nodes.clear()
rl = nt.nodes.new("CompositorNodeRLayers")
rl.scene = scene
z_out = rl.outputs["Depth"] if "Depth" in [o.name for o in rl.outputs] \
    else rl.outputs["Z"]
of = nt.nodes.new("CompositorNodeOutputFile")
of.base_path = OUT
of.format.file_format = "OPEN_EXR"
of.format.color_depth = "32"
of.format.color_mode = "RGBA"
slot = of.layer_slots.new("depth")
try:
    of.file_slots[0].path = "depth_"
except Exception as e:
    log("file_slot path EXC %s" % e)
nt.links.new(z_out, of.inputs["Image"])

# --- ROI-INTEGRATED: 用 3D 框投影算 ROI，开局部渲染 ---
Knp = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
from cv_to_blender_cam import build_camera_from_ground_pose as _bcgp
# T_w2c above comes from the source calibration.
uv8 = None
try:
    # 用 project_3d_box_opencv（已验证与 Blender 一致 0.07px）
    _bmn, _bmx = world_bbox(meshes)
    _cxw = (_bmn.x + _bmx.x) / 2.0
    _cyw = (_bmn.y + _bmx.y) / 2.0
    _L = _bmx.x - _bmn.x
    _Wd = _bmx.y - _bmn.y
    _Hh = _bmx.z - _bmn.z
    # 车头朝向：yaw 已在 parent.rotation_euler.z 里
    _yaw_rad = 0.0  # world_bbox already returns a world-aligned box
    uv8 = project_3d_box_opencv(Knp, T_w2c, (_cxw, _cyw, _bmn.z),
                                (_L, _Wd, _Hh), _yaw_rad)
    log("ROI bbox: (%.2f,%.2f,%.2f)~(%.2f,%.2f,%.2f)  yaw=%.3f" % (
        _bmn.x, _bmn.y, _bmn.z, _bmx.x, _bmx.y, _bmx.z, _yaw_rad))
    log("uv8: u %.1f~%.1f  v %.1f~%.1f" % (
        uv8[:,0].min(), uv8[:,0].max(), uv8[:,1].min(), uv8[:,1].max()))
    roi = set_blender_render_roi(scene, uv8, W, H)
    log("ROI: %s  area=%.1f%%" % (roi["roi_px"], roi["area_ratio"] * 100))
except Exception as e:
    import traceback
    log("ROI EXC %s" % e)
    log(traceback.format_exc()[:300])

opt = optimize_cycles_for_batch(scene, samples=SAMPLES)
log("Cycles opt: %s" % opt)

import time
# --- Pass 1: 只渲染车辆（隐藏地面），取干净 Z 深度 ---
ground.hide_render = True
bpy.context.view_layer.update()
t0 = time.time()
scene.render.filepath = os.path.join(OUT, "fg_rgba.png")
bpy.ops.render.render(write_still=True)
log("render(vehicle only) %.1fs" % (time.time() - t0))
ground.hide_render = False
bpy.context.view_layer.update()

# 恢复普通输出并重渲一次（用于 RGB）
nt.nodes.remove(of)
comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
# Second render explicitly includes the shadow-catching ground.
scene.render.filepath = os.path.join(OUT, "vehicle_with_shadow.png")
bpy.ops.render.render(write_still=True)
log("vehicle+ground shadow rendered")

# ---------------- 深度统计 ----------------
dst = {}
try:
    import glob as _g
    import numpy as np
    fs = sorted(_g.glob(os.path.join(OUT, "*.exr")))
    log("depth files: %s" % [os.path.basename(f) for f in fs])
    for f in fs:
        if f.endswith(".exr"):
            img = bpy.data.images.load(f)
            _h, _w = img.size[1], img.size[0]
            _px = np.array(list(img.pixels), dtype=np.float32).reshape(_h, _w, 4)
            arr = np.flipud(_px[:, :, 0]).copy()
            arr[arr <= 0.0] = np.inf          # ROI 外 0 值 -> inf
            nz = arr[np.isfinite(arr) & (arr < 1e5)]
            if nz.size:
                dst = {"min": float(nz.min()), "max": float(nz.max()),
                       "mean": float(nz.mean()), "valid_px": int(nz.size)}
                log("DEPTH min=%.2f max=%.2f mean=%.2f  (expect ~%.1f)"
                    % (nz.min(), nz.max(), nz.mean(), px))
            bpy.data.images.remove(img)
            break
except Exception as e:
    log("depth read EXC %s" % e)

# ---------------- 3D 框投影 ----------------
import sys as _sys
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
        "uv": [round(co.x, 5), round(co.y, 5)],
        "pix": [round(co.x * W, 1), round((1 - co.y) * H, 1)],
        "dist": round(co.z, 3),
    })
log("3D box projected")

# Geometric contact points are computed from actual wheel vertices in world space.
wheel_contacts = []
for obj in meshes:
    if "wheel" not in obj.name.lower() and "tire" not in obj.name.lower():
        continue
    coords = np.asarray([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices], dtype=float)
    if coords.size == 0:
        continue
    bottom = float(coords[:, 2].min())
    low = coords[coords[:, 2] <= bottom + 0.02]
    point = np.array([float(np.mean(low[:,0])), float(np.mean(low[:,1])), bottom, 1.0])
    camera_point = T_w2c @ point
    if camera_point[2] > 0:
        img_point = Knp @ camera_point[:3]
        pixel = (img_point[:2] / img_point[2]).tolist()
    else:
        pixel = None
    wheel_contacts.append({"name": obj.name, "world": point[:3].tolist(), "pixel": pixel})

meta = {
    "wheel_contacts": wheel_contacts,
    "training_eligible": False, "calibration_note": "source intrinsics and extrinsics; lens distortion not modeled",
    "glb": GLB, "bg": BG,
    "K": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "W": W, "H": H},
    "camera": {
        "calibration": KFILE, "extrinsic": "source_camera_to_body",
        "lens_mm": round(cd.lens, 3),
        "shift": [round(cd.shift_x, 5), round(cd.shift_y, 5)],
    },
    "pose": {"x": px, "y": py, "z": GROUND_Z, "yaw_deg": YAW},
    "dimensions_m": {"L": round(fext.x, 3), "W": round(fext.y, 3),
                     "H": round(fext.z, 3)},
    "bbox_min": [round(mn.x, 3), round(mn.y, 3), round(mn.z, 3)],
    "bbox_max": [round(mx.x, 3), round(mx.y, 3), round(mx.z, 3)],
    "box3d_corners": proj,
    "depth_stats": dst,
    "resolution": [W, H],
    "samples": SAMPLES,
}
with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

log("ALL DONE")
flush()
