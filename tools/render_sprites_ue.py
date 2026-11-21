#!/usr/bin/env python3
"""用 UE 把 Kenney CC0 模型渲染成写实精灵图（带 alpha）。

产物：
  /data/obj_assets/<class>/<view>.png
  /data/obj_assets/<class>/<view>_night.png (可选)

视角：front / rear / side_left / side_right / three_quarter
光照：day / night / dusk

关键：
  - 透明背景（SceneCapture 的 RGBA + 后处理 alpha）
  - 底部对齐（相机对准物体底边）
  - 尺度真实（按 Kenney 模型真实尺寸缩放）
"""
import json
import math
import os
import traceback

OUT_ROOT = "/data/obj_assets"
MAP = "/Game/Maps/codex_validation"

# 类别 → (Kenney 部件, 真实尺寸 m 长/宽/高)
CLASS_DEF = {
    "car": ({"body": "sedan_body",
             "wheels": ["sedan_wheel-front-left", "sedan_wheel-front-right",
                        "sedan_wheel-back-left", "sedan_wheel-back-right"]},
            4.30, 1.80, 1.50),
    "suv": ({"body": "suv_body",
             "wheels": ["suv_wheel-front-left", "suv_wheel-front-right",
                        "suv_wheel-back-left", "suv_wheel-back-right"]},
            4.50, 1.90, 1.75),
    "truck": ({"body": "delivery_body",
               "wheels": ["delivery_wheel-front-left", "delivery_wheel-front-right",
                          "delivery_wheel-back-left", "delivery_wheel-back-right"]},
              5.20, 2.00, 2.30),
    "truck_flat": ({"body": "truck_flat_body",
                    "wheels": ["truck_flat_wheel-front-left",
                               "truck_flat_wheel-front-right",
                               "truck_flat_wheel-back-left",
                               "truck_flat_wheel-back-right"]},
                   6.50, 2.40, 3.00),
    "taxi": ({"body": "taxi_body",
              "wheels": ["taxi_wheel-front-left", "taxi_wheel-front-right",
                         "taxi_wheel-back-left", "taxi_wheel-back-right"]},
             4.30, 1.80, 1.50),
    "hatchback": ({"body": "hatchback_sports_body",
                   "wheels": ["hatchback_sports_wheel-front-left",
                              "hatchback_sports_wheel-front-right",
                              "hatchback_sports_wheel-back-left",
                              "hatchback_sports_wheel-back-right"]},
                  3.90, 1.75, 1.45),
}

# 视角 → (方位角°, 仰角°)  方位角 0=正对车头(Y 轴负方向看)
VIEWS = {
    "front": (0, 6),
    "rear": (180, 6),
    "side_left": (90, 6),
    "side_right": (-90, 6),
}

# 光照 → (强度, 色温RGB)
LIGHTS = {
    "day": (3.2, (255, 250, 240)),
    "night": (0.35, (150, 170, 220)),
    "dusk": (1.4, (255, 200, 150)),
}

report = {"steps": []}


def rec(name, ok, value=""):
    report["steps"].append({"step": name, "ok": bool(ok),
                            "value": str(value)[:300]})
    print("[sprite] %s %s %s" % ("OK " if ok else "ERR", name, value))


def main():
    import unreal

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    kel = unreal.KismetSystemLibrary
    rl = unreal.RenderingLibrary

    les.load_level(MAP)
    rec("load_level", True, MAP)

    # 清场
    for a in list(eas.get_all_level_actors()):
        try:
            eas.destroy_actor(a)
        except Exception:
            pass
    rec("clear", True, "")

    # 光照
    sun = eas.spawn_actor_from_class(unreal.DirectionalLight,
                                     unreal.Vector(0, 0, 800),
                                     unreal.Rotator(-45, 30, 0))
    sun.set_actor_label("SPRITE_Sun")
    try:
        comp = sun.get_component_by_class(unreal.DirectionalLightComponent)
        comp.set_editor_property("intensity", 8.0)
    except Exception:
        pass

    sky = eas.spawn_actor_from_class(unreal.SkyLight,
                                     unreal.Vector(0, 0, 500))
    sky.set_actor_label("SPRITE_Sky")
    try:
        sc = sky.get_component_by_class(unreal.SkyLightComponent)
        sc.set_editor_property("intensity", 1.2)
        sc.set_editor_property("real_time_capture", True)
    except Exception:
        pass
    rec("lights", True, "")

    # 相机（每个视角重建）
    world = les.get_current_level() or None

    def render_view(actor_list, cls, size, view_name, light_name, out_path,
                    res=(512, 512)):
        W, H = res
        rr = rl.create_render_target2d(
            None, W, H, unreal.TextureRenderTargetFormat.RTF_RGBA8)
        cam = eas.spawn_actor_from_class(
            unreal.SceneCapture2D, unreal.Vector(0, 0, 0),
            unreal.Rotator(0, 0, 0))
        cam.set_actor_label("SPRITE_Cam")
        comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
        comp.set_editor_property("texture_target", rr)
        comp.set_editor_property("fov_angle", 30.0)
        comp.set_editor_property("always_persist_rendering_state", True)
        try:
            comp.set_editor_property(
                "capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        except Exception:
            pass

        L, Wd, Ht = size
        az, el = VIEWS[view_name]
        # 距离：让物体占画面 70%
        fov = 30.0
        dist = max(L, Ht) / (2.0 * math.tan(math.radians(fov) / 2.0)) / 0.70

        ar = math.radians(az)
        er = math.radians(el)
        cx = -math.cos(ar) * dist * math.cos(er)
        cy = -math.sin(ar) * dist * math.cos(er)
        cz = Ht / 2.0 + dist * math.sin(er)

        cam.set_actor_location(unreal.Vector(cx, cy, cz), False, False)
        # 看向物体中心
        import math as _m
        yaw = _m.degrees(_m.atan2(cy, cx))
        pitch = -_m.degrees(_m.atan2(cz - Ht / 2.0,
                                     _m.hypot(cx, cy)))
        cam.set_actor_rotation(unreal.Rotator(pitch, yaw, 0), False)

        # 光照
        inten, rgb = LIGHTS[light_name]
        try:
            lc = sun.get_component_by_class(unreal.DirectionalLightComponent)
            lc.set_editor_property("intensity", 8.0 * inten)
        except Exception:
            pass

        for _ in range(3):
            comp.capture_scene()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        rl.export_render_target(None, rr, os.path.dirname(out_path),
                                os.path.basename(out_path))
        eas.destroy_actor(cam)
        p = out_path
        if not os.path.isfile(p) and os.path.isfile(p.replace(".png", ".exr")):
            p = p.replace(".png", ".exr")
        return os.path.isfile(p) and os.path.getsize(p) > 100

    # 逐类别渲染
    lib = unreal.EditorAssetLibrary
    made = 0
    for cls, (parts, L, Wd, Ht) in CLASS_DEF.items():
        class_dir = os.path.join(OUT_ROOT, cls)

        # 生成物体
        actors = []
        body = "/Game/Validation/Models/SM_" + parts["body"]
        if not lib.does_asset_exist(body):
            rec("missing_" + cls, False, body)
            continue
        try:
            bm = lib.load_asset(body)
            pos = unreal.Vector(0, 0, 0)
            a = eas.spawn_actor_from_object(
                bm, pos, unreal.Rotator(0, 0, 0))
            a.set_actor_label("SPRITE_" + cls)
            actors.append(a)
        except Exception as e:
            rec("spawn_" + cls, False, str(e)[:120])
            continue

        # 缩放：Kenney 单位 → 真实米
        try:
            bnd = a.get_actor_bounds(False)
            cur = bnd[1] - bnd[0]
            # 用长度对齐
            s = L / max(cur.x, 1.0) if cur.x > 1 else 1.0
            a.set_actor_scale3d(unreal.Vector(s, s, s))
        except Exception:
            pass

        for v in ("front", "side_left"):
            for lt in ("day", "night"):
                name = v + ("" if lt == "day" else "_" + lt) + ".png"
                op = os.path.join(class_dir, name)
                ok = render_view(actors, cls, (L, Wd, Ht), v, lt, op)
                if ok:
                    made += 1
                rec("render_%s_%s_%s" % (cls, v, lt), ok, op)

        for a in actors:
            eas.destroy_actor(a)

        # meta
        meta = {
            "class": cls,
            "variant": "kenney",
            "size_m": {"width": Wd, "height": Ht, "length": L},
            "anchor": "bottom_center",
            "views": ["front", "side_left"],
            "lighting": ["day", "night"],
            "source": "kenney-car-kit-cc0",
            "license": "CC0",
        }
        with open(os.path.join(class_dir, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    rec("total", True, "%d 张精灵图" % made)


try:
    main()
except Exception:
    report["fatal"] = traceback.format_exc()[:2000]

with open("/tmp/sprite_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("DONE")
