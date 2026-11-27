#!/usr/bin/env python3
"""用 canonical 3D 资产做单帧合成渲染（Blender Cycles + 真实相机）。

流程：
  1. 读帧的相机标定（XML 或 yaml）
  2. 按参数随机采样 N 辆车的位置/朝向
  3. Blender 渲染（车辆 + Shadow Catcher 阴影 + 深度 EXR）
  4. 与背景图合成
  5. 产出 <stem>_obj.jpg

用法（在 Blender 里跑）:
    blender --background --python render_canonical_one.py -- <config.json>
"""
import bpy
import glob
import json
import math
import os
import random
import sys
from mathutils import Vector, Matrix
# LIGHT-SYSPATH: Blender 自带 Python，需显式加 tools 目录
import sys as _sys
_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in _sys.path:
    _sys.path.insert(0, _TOOLS)
try:
    from vehicle_lights import apply_lights, config_from_task
except Exception as _ie:
    print("[LIGHTS] import 失败:", _ie)
    apply_lights = None
    config_from_task = None

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
CFG = json.load(open(argv[0], encoding="utf-8"))

CANON_DIR = "/data/assets_global/OBJ_3D/canonical"
INDEX = "/data/assets_global/OBJ_3D/asset_index_obj3d.json"
OUT = CFG["out"]
BG = CFG["bg"]
K = CFG["K"]          # {fx,fy,cx,cy,W,H}
T_W2C = CFG["T"]      # 4x4 list
VEHICLES = CFG["vehicles"]   # [{uid, x, y, yaw, class}]
YEARS = CFG.get("seed", 0)


def log(s):
    print("[CANON-R] %s" % s)


def load_index():
    with open(INDEX, encoding="utf-8") as f:
        return {a["uid"]: a for a in json.load(f)["assets"]}


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x = K["W"]
    sc.render.resolution_y = K["H"]
    sc.render.resolution_percentage = 50
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.cycles.samples = 8
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    try:
        p = bpy.context.preferences.addons["cycles"].preferences
        p.compute_device_type = "OPTIX"
        p.get_devices()
        for d in p.devices:
            d.use = (d.type != "CPU")
        sc.cycles.device = "GPU"
        log("GPU OPTIX")
    except Exception:
        sc.cycles.device = "CPU"
    # 世界光照（阴天均匀）
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    bgn = w.node_tree.nodes["Background"]
    bgn.inputs[0].default_value = (0.62, 0.66, 0.74, 1)
    bgn.inputs[1].default_value = 2.2   # SHADOW-SOFT: 环境光补足
    if CFG.get('hdri_enabled', False):
        path='/data/assets_global/HDRI/urban_street_01_1k.hdr'
        if os.path.isfile(path):
            env=w.node_tree.nodes.new('ShaderNodeTexEnvironment')
            env.image=bpy.data.images.load(path,check_existing=True)
            w.node_tree.links.new(env.outputs['Color'],bgn.inputs['Color'])
            bgn.inputs['Strength'].default_value=0.8
            log('HDRI enabled: urban_street_01')
        else:
            log('HDRI not available, using existing world')
    return sc


def make_camera():
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("Cam")
    cd.sensor_width = 36.0
    cd.sensor_fit = "HORIZONTAL"
    W, H = K["W"], K["H"]
    cd.lens = K["fx"] * 36.0 / W
    # shift: cx_render = W/2 - shift_x*W  =>  shift_x = (W/2 - cx)/W
    cd.shift_x = (W / 2.0 - K["cx"]) / W
    cd.shift_y = (H / 2.0 - K["cy"]) / H
    cam = bpy.data.objects.new("Cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    # OpenCV -> Blender
    T = Matrix(T_W2C)
    M = T.inverted()
    # diag(1,-1,-1)
    flip = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0),
                   (0, 0, 0, 1)))
    cam.matrix_world = M @ flip
    return cam


def import_canonical(uid, idx):
    a = idx.get(uid)
    if not a:
        return None, None
    fp = a["path"]
    if not os.path.isfile(fp):
        return None, None
    before = set(bpy.data.objects.keys())
    try:
        bpy.ops.import_scene.gltf(filepath=fp)
    except Exception as e:
        log("import fail %s: %s" % (uid, str(e)[:50]))
        return None, None
    new = [bpy.data.objects[k] for k in bpy.data.objects.keys()
           if k not in before and bpy.data.objects[k].type == "MESH"]
    # 应用变换 + 解父
    bpy.ops.object.select_all(action="DESELECT")
    for o in new:
        o.select_set(True)
    if new:
        bpy.context.view_layer.objects.active = new[0]
        try:
            bpy.ops.object.transform_apply(location=True, rotation=True,
                                           scale=True)
        except Exception:
            pass
    for o in new:
        if o.parent:
            o.matrix_world = o.matrix_world.copy()
            o.parent = None
    bpy.context.view_layer.update()
    return new, a


def place(objs, x, y, yaw, ground_z=0.0):
    """平移到位置 + 绕 Z 旋转。canonical 模型已中心在(0,0)、底面 z=0。"""
    for o in objs:
        o.rotation_mode = "XYZ"
        # 先平移再用旋转矩阵（绕世界 Z）
        M = Matrix.Translation((x, y, ground_z)) @ \
            Matrix.Rotation(yaw, 4, 'Z')
        o.matrix_world = M @ o.matrix_world
    bpy.context.view_layer.update()


def setup_shadow_catcher():
    # NO-GHOST-FIX: 平面只需覆盖视野近处，太大会污染深度/alpha
    bpy.ops.mesh.primitive_plane_add(size=120, location=(60, 0, float(VEHICLES[0].get("ground_z", 0.0)) if VEHICLES else 0.0))
    g = bpy.context.object
    g.name = "ShadowCatcher"
    g.is_shadow_catcher = True
    return g


def add_sun():
    d = bpy.data.lights.new("Sun", type="SUN")
    d.energy = 1.2                 # SHADOW-SOFT: 弱化直射
    d.angle = math.radians(25)     # SHADOW-SOFT: 大角度 = 软阴影
    o = bpy.data.objects.new("Sun", d)
    bpy.context.scene.collection.objects.link(o)
    o.rotation_euler = (math.radians(52), 0, math.radians(145))


def add_contact_light():
    """近似天空顶光，补出轮胎正下方的软接触阴影。

    单一低角度太阳会把阴影整体甩到车外，几何虽然接地，视觉上仍像
    悬浮。大面积弱顶光只增强接触区，不制造硬黑椭圆。
    """
    d = bpy.data.lights.new("ContactSky", type="AREA")
    d.energy = 420.0
    d.shape = "DISK"
    d.size = 7.0
    o = bpy.data.objects.new("ContactSky", d)
    bpy.context.scene.collection.objects.link(o)
    o.location = (18.0, 0.0, 14.0)
    # Area light 默认沿局部 -Z 发光，零旋转即垂直照向路面。
    o.rotation_euler = (0.0, 0.0, 0.0)


def main():
    idx = load_index()
    sc = setup_scene()
    make_camera()
    add_sun()
    add_contact_light()

    placed = 0
    _PLACED = []
    for v in VEHICLES:
        uid = v["uid"]
        # GLB-PATH-DIRECT: 优先用 vehicles 里给的具体路径
        direct_glb = v.get("glb")
        if direct_glb and os.path.isfile(direct_glb):
            before = set(bpy.data.objects.keys())
            try:
                bpy.ops.import_scene.gltf(filepath=direct_glb)
            except Exception as e:
                log("direct import fail %s: %s" % (uid, str(e)[:60]))
                continue
            new = [bpy.data.objects[k] for k in bpy.data.objects.keys()
                   if k not in before and bpy.data.objects[k].type == "MESH"]
            if not new:
                log("no mesh %s" % uid)
                continue
            bpy.ops.object.select_all(action="DESELECT")
            for o in new:
                o.select_set(True)
            bpy.context.view_layer.objects.active = new[0]
            try:
                bpy.ops.object.transform_apply(location=True,
                                               rotation=True, scale=True)
            except Exception:
                pass
            for o in new:
                if o.parent:
                    o.matrix_world = o.matrix_world.copy()
                    o.parent = None
            bpy.context.view_layer.update()
            # 归一化到给定尺寸
            sz = v.get("size") or [1, 1, 1]
            mn0 = Vector((1e18,)*3); mx0 = Vector((-1e18,)*3)
            for o in new:
                for c in o.bound_box:
                    w = o.matrix_world @ Vector(c)
                    for i in range(3):
                        mn0[i] = min(mn0[i], w[i]); mx0[i] = max(mx0[i], w[i])
            ext0 = mx0 - mn0
            # SCALE-UNIFORM-FIX: canonical 模型已是标准尺寸，只做等比缩放
            #   用最长边对齐（避免 min(scale3) 把车压扁）
            _le = max(ext0.x, ext0.y, ext0.z)
            _lt = max(sz[0], sz[1], sz[2])
            # 若尺寸接近（±5%），完全不动
            if abs(_le - _lt) / max(_lt, 1e-6) > 0.05:
                s_uniform = _lt / max(_le, 1e-6)
            else:
                s_uniform = 1.0
            if abs(s_uniform - 1.0) > 1e-3:
                for o in new:
                    o.scale = (o.scale.x * s_uniform,
                               o.scale.y * s_uniform,
                               o.scale.z * s_uniform)
            bpy.context.view_layer.update()
            # 重新算 bbox 并居中+贴地
            mn1 = Vector((1e18,)*3); mx1 = Vector((-1e18,)*3)
            for o in new:
                for c in o.bound_box:
                    w = o.matrix_world @ Vector(c)
                    for i in range(3):
                        mn1[i] = min(mn1[i], w[i]); mx1[i] = max(mx1[i], w[i])
            cx0 = (mn1.x+mx1.x)/2; cy0 = (mn1.y+mx1.y)/2
            transform = (Matrix.Translation((float(v['x']), float(v['y']),
                                              float(v.get('z', 0.0)))) @
                         Matrix.Rotation(math.radians(float(v.get('yaw', 0.0))), 4, 'Z') @
                         Matrix.Translation((-cx0, -cy0, -mn1.z)))
            for o in new:
                o.matrix_world = transform @ o.matrix_world
            bpy.context.view_layer.update()
            log("direct placed %-24s at (%.1f, %.1f, %.2f) size=%s" % (
                os.path.basename(direct_glb), v["x"], v["y"],
                v.get("z", 0.0), sz))
            _PLACED.append((new, uid))
            placed += 1
            continue

        objs, a = import_canonical(uid, idx)
        _PLACED.append((objs, uid))
        if not objs:
            log("SKIP %s" % uid)
            continue
        place(objs, float(v["x"]), float(v["y"]),
              math.radians(float(v.get("yaw", 0.0))),
              ground_z=float(v.get("z", 0.0)))
        # NO-GHOST-FIX: 强制不透明（除玻璃外），避免半透明重影
        for o in objs:
            for m in (o.data.materials or []):
                if m is None:
                    continue
                nm = (m.name or "").lower()
                # Only explicitly identified paint: never overwrite glass, tire,
                # metal/roughness maps or unclassified generated materials.
                if CFG.get('paint_coat_enabled',False) and any(k in nm for k in ('carpaint','car_paint','car paint','body_paint')) and m.use_nodes:
                    bsdf=next((n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
                    if bsdf and 'Coat Weight' in bsdf.inputs and not bsdf.inputs['Coat Weight'].is_linked:
                        bsdf.inputs['Coat Weight'].default_value=max(bsdf.inputs['Coat Weight'].default_value,0.2)
                        if 'Coat Roughness' in bsdf.inputs and not bsdf.inputs['Coat Roughness'].is_linked:
                            bsdf.inputs['Coat Roughness'].default_value=0.18
                if any(k in nm for k in ("plate", "license", "badge",
                                         "emblem", "decal")):
                    m.blend_method = "BLEND"
                else:
                    m.blend_method = "OPAQUE"
        log("placed %-22s at (%.1f, %.1f) yaw=%.0f cls=%s lwh=%s" % (
            uid, v["x"], v["y"], v.get("yaw", 0), a.get("true_class"),
            a.get("lwh")))
        placed += 1

    # VEHICLE-LIGHTS-HOOK: 驱动所有车的车灯
    try:
        if apply_lights and _PLACED:
            _params = CFG.get("params") or {}
            _lights = config_from_task(_params)
            if _lights.get("enabled"):
                _fidx = int(CFG.get("frame_idx", 0))
                _fps = float(CFG.get("fps", 10.0))
                for _objs, _uid in _PLACED:
                    apply_lights(_objs, _uid, _fidx, _fps, _lights, log=log)
    except Exception as _le:
        log("  lights 失败: %s" % _le)

    if placed != len(VEHICLES):
        raise RuntimeError("vehicle placement failed: %d/%d" % (placed, len(VEHICLES)))

    ground = setup_shadow_catcher()

    # DEPTH-EXPORT: 先渲染深度（隐藏地面避免污染 Z pass）
    dep_npy = CFG.get("depth_npy")
    if dep_npy:
        ground.hide_render = True
        sc.render.film_transparent = True
        sc.view_layers[0].use_pass_z = True
        sc.use_nodes = True
        nt = sc.node_tree
        nt.nodes.clear()
        rl = nt.nodes.new("CompositorNodeRLayers")
        rl.scene = sc
        of = nt.nodes.new("CompositorNodeOutputFile")
        of.base_path = dep_npy + ".exr_dir"
        of.format.file_format = "OPEN_EXR"
        of.format.color_depth = "32"
        of.format.color_mode = "RGBA"
        of.layer_slots.new("z")
        os.makedirs(of.base_path, exist_ok=True)
        nt.links.new(rl.outputs["Depth"], of.inputs["Image"])
        sc.render.filepath = of.base_path + "/_d"
        bpy.ops.render.render(write_still=False)
        zs = sorted(glob.glob(of.base_path + "/*.exr"))
        if zs:
            img = bpy.data.images.load(zs[0])
            W2, H2 = img.size
            import numpy as np
            px = np.array(list(img.pixels), dtype=np.float32)
            px = px.reshape(H2, W2, 4)
            z = np.flipud(px[:, :, 0]).copy()
            z[z <= 0] = np.inf
            np.save(dep_npy, z)
            log("depth saved %s  范围 %.2f~%.2f" % (
                dep_npy, float(np.nanmin(z[np.isfinite(z)])),
                float(np.nanmax(z[np.isfinite(z)]))))
            bpy.data.images.remove(img)
        # RESTORE-COMPOSITE: 清空节点后必须重建 Composite 才能正常出图
        nt.nodes.clear()
        _rl = nt.nodes.new("CompositorNodeRLayers")
        _rl.scene = sc
        _cp = nt.nodes.new("CompositorNodeComposite")
        nt.links.new(_rl.outputs["Image"], _cp.inputs["Image"])
        ground.hide_render = False

    # 渲染前景（含阴影）
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sc.render.filepath = OUT
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    bpy.ops.render.render(write_still=True)
    # SYNC-WRITE: 确保落盘（Blender 有时延迟写）
    import time as _time
    for _ in range(30):
        if os.path.isfile(OUT) and os.path.getsize(OUT) > 1000:
            break
        _time.sleep(0.2)
    log("rendered %s  (%d vehicles)  exists=%s size=%s" % (
        OUT, placed, os.path.isfile(OUT),
        os.path.getsize(OUT) if os.path.isfile(OUT) else 0))


main()
