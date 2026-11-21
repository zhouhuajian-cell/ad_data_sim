#!/usr/bin/env python3
"""渲染 Kenney CC0 模型为精灵图（按 demo_obj_v4 的成功配置）。

关键点（来自 demo_obj_v4）：
  - RT 格式 RTF_RGBA16F
  - capture_source = SCS_FINAL_COLOR_LDR
  - lock_exposure() 锁定曝光（不锁会全黑）
  - capture_every_frame = False
  - 先 spawn 物体再摆相机
"""
import json
import math
import os
import time
import traceback

import sys
sys.path.insert(0, "/data/avsynth/Content/Python")

OUT_ROOT = "/data/obj_assets"
MAP = "/Game/Maps/codex_validation"

CLASS_DEF = {
    "car": ("sedan", 4.30, 1.80, 1.50),
    "suv": ("suv", 4.50, 1.90, 1.75),
    "truck": ("delivery", 5.20, 2.00, 2.30),
    "truck_flat": ("truck_flat", 6.50, 2.40, 3.00),
    "taxi": ("taxi", 4.30, 1.80, 1.50),
    "hatchback": ("hatchback_sports", 3.90, 1.75, 1.45),
    "police": ("police", 4.30, 1.80, 1.50),
    "ambulance": ("ambulance", 5.00, 2.00, 2.40),
    "cone": ("cone", 0.28, 0.28, 0.70),
}

VIEWS = {
    "front": (0, 0),
    "rear": (180, 0),
    "side_left": (90, 0),
}

report = {"steps": []}
WORLD = None


def rec(n, ok, v=""):
    report["steps"].append({"step": n, "ok": bool(ok), "value": str(v)[:300]})
    print("[sp] %s %s %s" % ("OK " if ok else "ERR", n, v))


def _editor_world():
    import unreal
    sub = getattr(unreal, "UnrealEditorSubsystem", None)
    getter = getattr(unreal, "get_editor_subsystem", None)
    if sub and getter:
        try:
            return getter(sub).get_editor_world()
        except Exception:
            pass
    lib = getattr(unreal, "EditorLevelLibrary", None)
    if lib:
        try:
            return lib.get_editor_world()
        except Exception:
            pass
    return None


def lock_exposure(comp):
    try:
        pp = comp.get_editor_property("post_process_settings")
        for a, v in (
            ("override_auto_exposure_min_brightness", True),
            ("override_auto_exposure_max_brightness", True),
            ("auto_exposure_min_brightness", 1.0),
            ("auto_exposure_max_brightness", 1.0),
            ("auto_exposure_bias", 0.0),
            ("override_bloom_intensity", True), ("bloom_intensity", 0.0),
            ("override_vignette_intensity", True), ("vignette_intensity", 0.0),
            ("override_motion_blur_amount", True), ("motion_blur_amount", 0.0),
            ("override_ambient_occlusion_intensity", True),
            ("ambient_occlusion_intensity", 0.0),
        ):
            try:
                pp.set_editor_property(a, v)
            except Exception:
                pass
        comp.set_editor_property("post_process_settings", pp)
        return True
    except Exception:
        return False


def build_actor(unreal, eas, cls, model_name, L, Wd, Ht):
    """生成物体并按真实尺寸缩放。"""
    lib = unreal.EditorAssetLibrary
    parts = []

    body_path = "/Game/Validation/Models/SM_" + model_name + "_body"
    if not lib.does_asset_exist(body_path):
        body_path = "/Game/Validation/Models/SM_" + model_name
    if not lib.does_asset_exist(body_path):
        return None

    bm = lib.load_asset(body_path)
    a = eas.spawn_actor_from_object(bm, unreal.Vector(0, 0, 0),
                                    unreal.Rotator(0, 0, 0))
    if a is None:
        return None
    a.set_actor_label("SP_" + cls)
    parts.append(a)

    # 按长度缩放
    try:
        bnd = a.get_actor_bounds(False)
        ext = bnd[1]
        cur_len = ext.x * 2.0
        if cur_len > 1:
            s = (L * 100.0) / cur_len
            a.set_actor_scale3d(unreal.Vector(s, s, s))
    except Exception:
        pass

    return parts


def main():
    import unreal

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    rl = unreal.RenderingLibrary

    les.load_level(MAP)
    rec("load", True, MAP)

    global WORLD
    WORLD = _editor_world()
    rec("world", WORLD is not None)

    for a in list(eas.get_all_level_actors()):
        try:
            eas.destroy_actor(a)
        except Exception:
            pass
    rec("clear", True)

    # 光照（照 demo_obj_v4）
    sun = eas.spawn_actor_from_class(unreal.DirectionalLight,
                                     unreal.Vector(0, 0, 900),
                                     unreal.Rotator(-45, 30, 0))
    try:
        c = sun.get_component_by_class(unreal.DirectionalLightComponent)
        c.set_editor_property("intensity", 8.0)
    except Exception:
        pass
    sky = eas.spawn_actor_from_class(unreal.SkyLight,
                                     unreal.Vector(0, 0, 500))
    try:
        sc = sky.get_component_by_class(unreal.SkyLightComponent)
        sc.set_editor_property("intensity", 1.2)
        sc.set_editor_property("real_time_capture", True)
    except Exception:
        pass
    rec("lights", True)

    # 相机（照 demo_obj_v4）
    W, H = 640, 640
    cam = eas.spawn_actor_from_class(unreal.SceneCapture2D,
                                     unreal.Vector(0, 0, 100))
    cam.set_actor_label("SP_Cam")
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    src = getattr(unreal.SceneCaptureSource, "SCS_FINAL_COLOR_LDR", None)
    rr = rl.create_render_target2d(
        WORLD, W, H, unreal.TextureRenderTargetFormat.RTF_RGBA16F)
    rec("rt", rr is not None, str(rr)[:50] if rr else "None")
    if rr is None:
        rec("abort", False, "RT 创建失败")
        return

    comp.set_editor_property("texture_target", rr)
    if src is not None:
        comp.set_editor_property("capture_source", src)
    comp.set_editor_property("fov_angle", 35.0)
    comp.set_editor_property("always_persist_rendering_state", True)
    comp.set_editor_property("capture_every_frame", False)
    rec("lock_exposure", lock_exposure(comp))
    rec("cam_setup", True)

    lib = unreal.EditorAssetLibrary
    made = 0

    for cls, (model, L, Wd, Ht) in CLASS_DEF.items():
        class_dir = os.path.join(OUT_ROOT, cls)
        os.makedirs(class_dir, exist_ok=True)

        parts = build_actor(unreal, eas, cls, model, L, Wd, Ht)
        if not parts:
            rec("build_" + cls, False, model)
            continue
        rec("build_" + cls, True, model)

        for view, (az, el) in VIEWS.items():
            for light in ("day", "night"):
                # 光照调整
                try:
                    lc = sun.get_component_by_class(
                        unreal.DirectionalLightComponent)
                    lc.set_editor_property(
                        "intensity", 8.0 if light == "day" else 1.2)
                except Exception:
                    pass

                # 相机位置
                dist = max(L, Ht) * 100.0 * 2.6
                ar = math.radians(az)
                er = math.radians(el)
                cx = -math.cos(ar) * dist * math.cos(er)
                cy = -math.sin(ar) * dist * math.cos(er)
                cz = (Ht * 100.0) / 2.0 + dist * math.sin(er)

                cam.set_actor_location(unreal.Vector(cx, cy, cz), False, False)
                yaw = math.degrees(math.atan2(cy, cx))
                pitch = -math.degrees(
                    math.atan2(cz - (Ht * 100.0) / 2.0,
                               math.hypot(cx, cy)))
                cam.set_actor_rotation(unreal.Rotator(pitch, yaw, 0), False)

                for _ in range(4):
                    comp.capture_scene()
                time.sleep(0.12)

                name = view + ("" if light == "day" else "_" + light) + ".png"
                rl.export_render_target(WORLD, rr, class_dir, name)
                time.sleep(0.25)

                p = os.path.join(class_dir, name)
                ok = os.path.isfile(p) and os.path.getsize(p) > 500
                if ok:
                    made += 1
                rec("render_%s_%s_%s" % (cls, view, light), ok,
                    "%d B" % (os.path.getsize(p) if os.path.isfile(p) else 0))

        # meta
        meta = {
            "class": cls, "variant": "kenney",
            "size_m": {"width": Wd, "height": Ht, "length": L},
            "anchor": "bottom_center",
            "views": list(VIEWS.keys()),
            "lighting": ["day", "night"],
            "source": "kenney-car-kit-cc0",
            "license": "CC0",
        }
        with open(os.path.join(class_dir, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

        for a in parts:
            try:
                eas.destroy_actor(a)
            except Exception:
                pass

    rec("total", True, "%d 张" % made)


try:
    main()
except Exception:
    report["fatal"] = traceback.format_exc()[:1500]

with open("/tmp/sp_report.json", "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("DONE")
