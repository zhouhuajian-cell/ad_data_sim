#!/usr/bin/env python3
"""把候选 GLB 素材批量渲染成 12 个 yaw 视角的 RGBA 贴片，写入 OBJ_PATCH。

用法:
    blender --background --python render_glb_patches.py -- [--only cls1,cls2]
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


def arg(name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


SRC_ROOT = "/data/assets_global/OBJ_3D"
OUT_ROOT = "/data/assets_global/OBJ_PATCH"
RES = 512
YAWS = list(range(0, 360, 30))       # 12 个角度
PITCH = -8.0

# 语义类别 -> (目标 LWH 米, GLB 相对路径列表)
# 只收录质量检查通过的（有贴图、面数合理、语义清晰）
PLAN = {
    "car": [
        ("GLOBAL/candidate_car/car-6695d36a/mesh.glb", (4.5, 1.8, 1.5)),
        ("GLOBAL/candidate_car/car-e2a25f0d/mesh.glb", (4.5, 1.8, 1.5)),
    ],
    "truck": [
        ("GLOBAL/candidate_truck/fire-truck-e0fadd42/mesh.glb", (7.5, 2.4, 3.0)),
        ("GLOBAL/candidate_truck/gaz-66-offroad-truck-1cc3374e/mesh.glb", (6.0, 2.3, 2.6)),
        ("GLOBAL/candidate_truck/japanese-mini-truck-d5d93879/mesh.glb", (4.5, 1.7, 2.0)),
        ("GLOBAL/candidate_truck/truck-0b0fa5fd/mesh.glb", (8.0, 2.5, 3.2)),
    ],
    "van": [
        ("GLOBAL/candidate_van/french-delivery-van-a350a422/mesh.glb", (5.2, 2.0, 2.2)),
    ],
    "bus": [
        ("GLOBAL/candidate_bus/generic-town-bus-14fe03d7/mesh.glb", (10.0, 2.5, 3.0)),
    ],
    "cone": [
        ("GLOBAL/candidate_cone/traffic-cone-4867e2a7/mesh.glb", (0.40, 0.40, 0.70)),
        ("GLOBAL/candidate_cone/traffic-cone-game-ready-8cdac192/mesh.glb", (0.40, 0.40, 0.70)),
    ],
    "barrier": [
        ("GLOBAL/candidate_barrier/road-barrier-f0dfd802/mesh.glb", (2.0, 0.5, 0.9)),
        ("GLOBAL/candidate_barrier/roadworks-barrier-074f4faa/mesh.glb", (2.0, 0.5, 0.9)),
    ],
    "motorcycle": [
        ("GLOBAL/candidate_motorcycle/classic-motorcycle-f246ff1a/mesh.glb", (2.2, 0.8, 1.3)),
        ("GLOBAL/candidate_motorcycle/moped-a8f0d743/mesh.glb", (1.8, 0.7, 1.1)),
    ],
    "bicycle": [
        ("GLOBAL/candidate_bicycle/bicycle-467c6873/mesh.glb", (1.8, 0.6, 1.2)),
    ],
    "scooter": [
        ("GLOBAL/candidate_scooter/tier-scooter-9479a6d3/mesh.glb", (1.2, 0.5, 1.2)),
    ],
}

ORIENT_DIR = "/tmp/glb_orient"
_CAND_ROT = {
    "identity": (0, 0, 0),
    "rx90": (90, 0, 0),
    "rx180": (180, 0, 0),
    "rx270": (270, 0, 0),
    "rz90": (0, 0, 90),
    "rz270": (0, 0, 270),
}


def load_orient(name):
    """ORIENT-AUTO: read detected orientation. Returns (rx,ry,rz) deg or None."""
    fp = os.path.join(ORIENT_DIR, name + ".json")
    if not os.path.isfile(fp):
        return None
    try:
        d = json.load(open(fp, encoding="utf-8"))
        b = d.get("best")
        if b in _CAND_ROT:
            return _CAND_ROT[b]
    except Exception:
        pass
    return None



# RESTORE-9: 之前因朝向问题被删，现已用 detect_glb_orientation 修正
PLAN_EXTRA = {
    "barrier": [
        ("GLOBAL/candidate_barrier/road-barrier-f0dfd802/mesh.glb",
         (2.0, 0.5, 0.9)),
        ("GLOBAL/candidate_barrier/roadworks-barrier-074f4faa/mesh.glb",
         (2.0, 0.5, 0.9)),
    ],
    "bicycle": [
        ("GLOBAL/candidate_bicycle/yellow-bicycle-4cc719f3/mesh.glb",
         (1.8, 0.6, 1.2)),
    ],
    "bus": [
        ("GLOBAL/candidate_bus/bus-xyz-coursework-330fdc30/mesh.glb",
         (10.0, 2.5, 3.0)),
    ],
    "car": [
        ("GLOBAL/candidate_car/car-6695d36a/mesh.glb", (4.5, 1.8, 1.5)),
    ],
    "motorcycle": [
        ("GLOBAL/candidate_motorcycle/moped-a8f0d743/mesh.glb",
         (1.8, 0.7, 1.1)),
        ("GLOBAL/candidate_motorcycle/motorcycle-0d0aee26/mesh.glb",
         (2.2, 0.8, 1.3)),
    ],
    "scooter": [
        ("GLOBAL/candidate_scooter/tier-scooter-9479a6d3/mesh.glb",
         (1.2, 0.5, 1.2)),
    ],
    "truck": [
        ("GLOBAL/candidate_truck/fire-truck-e0fadd42/mesh.glb",
         (7.5, 2.4, 3.0)),
        ("GLOBAL/candidate_truck/japanese-mini-truck-d5d93879/mesh.glb",
         (4.5, 1.7, 2.0)),
        ("GLOBAL/candidate_truck/truck-0b0fa5fd/mesh.glb",
         (8.0, 2.5, 3.2)),
    ],
    "van": [
        ("GLOBAL/candidate_van/french-delivery-van-a350a422/mesh.glb",
         (5.2, 2.0, 2.2)),
    ],
}
for _k, _v in PLAN_EXTRA.items():
    PLAN.setdefault(_k, []).extend(_v)

LOG = "/tmp/glb_patch_report.txt"
rep = []


def log(s):
    rep.append(str(s))
    print("[PATCH] %s" % s)


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


def normalize(objs, lwh):
    """车长→X，车宽→Y，高→Z，底部 Z=0，居中到原点。"""
    mn, mx = bbox_of(objs)
    ext = mx - mn

    # 最长水平轴 -> X
    if ext.y > ext.x:
        for o in objs:
            o.rotation_mode = "XYZ"
            o.rotation_euler.z += math.radians(90)
        bpy.context.view_layer.update()
        mn, mx = bbox_of(objs)
        ext = mx - mn
    # 若 Z 是最长轴（可能模型是躺着的），绕 X 转 90 度立起来
    if ext.z > max(ext.x, ext.y) * 1.3:
        for o in objs:
            o.rotation_mode = "XYZ"
            o.rotation_euler.x += math.radians(90)
        bpy.context.view_layer.update()
        mn, mx = bbox_of(objs)
        ext = mx - mn
        # 再检查水平轴
        if ext.y > ext.x:
            for o in objs:
                o.rotation_mode = "XYZ"
                o.rotation_euler.z += math.radians(90)
            bpy.context.view_layer.update()
            mn, mx = bbox_of(objs)
            ext = mx - mn

    sx = lwh[0] / max(ext.x, 1e-6)
    sy = lwh[1] / max(ext.y, 1e-6)
    sz = lwh[2] / max(ext.z, 1e-6)

    parent = bpy.data.objects.new("Root", None)
    bpy.context.scene.collection.objects.link(parent)
    for o in objs:
        o.parent = parent
    parent.scale = (sx, sy, sz)
    bpy.context.view_layer.update()

    mn, mx = bbox_of(objs)
    parent.location.x -= (mn.x + mx.x) / 2
    parent.location.y -= (mn.y + mx.y) / 2
    parent.location.z -= mn.z
    bpy.context.view_layer.update()

    mn, mx = bbox_of(objs)
    return mn, mx


def setup_scene():
    sc = bpy.context.scene
    sc.render.resolution_x = RES
    sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.05
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 2
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dt in ("OPTIX", "CUDA"):
            prefs.compute_device_type = dt
            prefs.get_devices()
            if any(d.type != "CPU" for d in prefs.devices):
                for d in prefs.devices:
                    d.use = True
                sc.cycles.device = "GPU"
                return "GPU"
    except Exception:
        pass
    sc.cycles.device = "CPU"
    return "CPU"


def make_camera():
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("Cam")
    cd.lens = 50
    cam = bpy.data.objects.new("Cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def make_lights():
    sc = bpy.context.scene
    sd = bpy.data.lights.new("Sun", type="SUN")
    sd.energy = 3.5
    sd.angle = math.radians(3.0)
    sun = bpy.data.objects.new("Sun", sd)
    sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(55), 0, math.radians(35))

    world = bpy.data.worlds.new("W")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.55, 0.65, 0.82, 1.0)
    bg.inputs[1].default_value = 1.1


def render_yaw(cam, mn, mx, yaw_deg):
    sc = bpy.context.scene
    center = (mn + mx) / 2
    radius = float(np_norm(mx - mn)) / 2

    yaw = math.radians(yaw_deg)
    pitch = math.radians(PITCH)
    d = radius * 2.4

    cam_pos = Vector((
        center.x + d * math.cos(pitch) * math.cos(yaw),
        center.y + d * math.cos(pitch) * math.sin(yaw),
        center.z + d * math.sin(pitch) + radius * 0.10,
    ))
    fwd = center - cam_pos
    cam.location = cam_pos
    cam.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()


def np_norm(v):
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def main():
    only = arg("--only", "")
    only_set = set(only.split(",")) if only else None

    dev = setup_scene()
    log("device: %s" % dev)
    os.makedirs(OUT_ROOT, exist_ok=True)

    total = 0
    for cls, items in PLAN.items():
        if only_set and cls not in only_set:
            continue
        out_dir = os.path.join(OUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)
        for rel, lwh in items:
            src = os.path.join(SRC_ROOT, rel)
            if not os.path.isfile(src):
                log("MISS %s" % rel)
                continue
            name = rel.split("/")[-2]
            try:
                bpy.ops.wm.read_factory_settings(use_empty=True)
                setup_scene()
                bpy.ops.import_scene.gltf(filepath=src)
                ms = [o for o in bpy.context.scene.objects if o.type == "MESH"]
                if not ms:
                    log("NO_MESH %s" % name)
                    continue
                # ORIENT-AUTO: apply detected axis correction
                rot = load_orient(name)
                if rot is not None:
                    root0 = bpy.data.objects.new("PreRot", None)
                    bpy.context.scene.collection.objects.link(root0)
                    for o in ms:
                        o.parent = root0
                    root0.rotation_euler = (math.radians(rot[0]),
                                            math.radians(rot[1]),
                                            math.radians(rot[2]))
                    bpy.context.view_layer.update()
                    log("  orient %s -> %s" % (name, list(rot)))
                else:
                    log("  orient %s -> NOT_DETECTED" % name)

                mn, mx = normalize(ms, lwh)
                make_lights()
                cam = make_camera()
                for yaw in YAWS:
                    render_yaw(cam, mn, mx, yaw)
                    bpy.context.scene.render.filepath = os.path.join(
                        out_dir, "%s_yaw%03d.png" % (name, yaw))
                    bpy.ops.render.render(write_still=True)
                meta = {"model": name, "category": cls,
                        "source": rel, "real_size_m":
                        {"L": lwh[0], "W": lwh[1], "H": lwh[2]},
                        "yaws": YAWS, "res": RES}
                with open(os.path.join(out_dir, "%s_meta.json" % name), "w",
                          encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                total += 1
                log("OK %-6s %-45s %d 视角" % (cls, name, len(YAWS)))
            except Exception as e:
                log("ERR %s: %s" % (name, str(e)[:80]))

    log("=== 完成 %d 个模型 ===" % total)
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")


main()
