# render_semantic_v2.py — 语义通道（编码版）端到端：换材质→捕获→由 Python 侧解码验证
#
# 与 render_semantic_depth.py 的区别：用 Semantic3 下按编码表生成的材质（灰度 = ENCODE_INPUT/255），
#   输出的每像素亮度落在 32/64/96/128/160/192/224 档位上，解码时按最近档位判定类别。
#
# 用法：
#   xvfb-run -a UnrealEditor-Cmd <uproject> -ExecutePythonScript=<本文件> -Vulkan -unattended
#   AD_SEM_RES=1920x1080  AD_SEM_OUT=/data/render/semantic
import json
import os
import sys
import time
import traceback

sys.path.insert(0, "/data/avsynth/Content/Python")
import unreal
import semantic_encode as se

MAT_DIR = "/Game/Validation/Semantic3"
MAP = os.environ.get("AD_SEM_MAP", "/Game/Maps/codex_validation")
OUT = os.environ.get("AD_SEM_OUT", "/data/render/semantic")
RES = os.environ.get("AD_SEM_RES", "1920x1080")
CAM = os.environ.get("AD_SEM_CAM", "-900,0,180,0,0")

report = {"steps": []}


def rec(name, ok, value=""):
    report["steps"].append({"step": name, "ok": ok, "value": str(value)[:300]})
    print("[sem] %s %s %s" % ("OK " if ok else "ERR", name, value))


try:
    w, h = [int(x) for x in RES.lower().split("x")]
    cx, cy, cz, cyaw, cpitch = [float(x) for x in CAM.split(",")]
    os.makedirs(OUT, exist_ok=True)

    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    les.load_level(MAP)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    rec("load_level", True, MAP)

    # 补 MOVABLE 光（阈值判定不该依赖光照，但 Unlit 材质也需要场景被初始化）
    sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1000), unreal.Rotator(-40, -30, 0))
    c = sun.get_component_by_class(unreal.DirectionalLightComponent)
    c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    c.set_editor_property("intensity", 10.0)

    rt = unreal.RenderingLibrary.create_render_target2d(
        world, w, h, unreal.TextureRenderTargetFormat.RTF_RGBA16F)
    cam = eas.spawn_actor_from_class(
        unreal.SceneCapture2D, unreal.Vector(cx, cy, cz), unreal.Rotator(cpitch, cyaw, 0))
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", rt)
    comp.set_editor_property("fov_angle", 90.0)
    comp.set_editor_property("always_persist_rendering_state", True)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR)

    # 固定曝光：自动曝光会给画面加增益，直接毁掉亮度档位（实测输出被抬到 48~50）
    pps = comp.get_editor_property("post_process_settings")
    pps.set_editor_property("override_auto_exposure_method", True)
    pps.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    pps.set_editor_property("override_auto_exposure_bias", True)
    pps.set_editor_property("auto_exposure_bias", 0.0)
    comp.set_editor_property("post_process_settings", pps)
    rec("fixed_exposure", True, "AEM_MANUAL")

    # 按 stencil 换材质
    mats = {}
    for cls in se.SEMANTIC_CLASSES:
        p = "%s/M_Sem3_%02d_%s" % (MAT_DIR, cls, se.SEMANTIC_CLASSES[cls])
        m = unreal.load_asset(p)
        if m is not None:
            mats[cls] = m
    rec("load_materials", len(mats), "%d 个" % len(mats))

    saved, n = [], 0
    for a in eas.get_all_level_actors():
        comp2 = a.get_component_by_class(unreal.StaticMeshComponent)
        if comp2 is None or comp2.get_editor_property("static_mesh") is None:
            continue
        st = int(comp2.get_editor_property("custom_depth_stencil_value"))
        if st in mats:
            slots = comp2.get_num_materials()
            for slot in range(max(1, slots)):
                orig = comp2.get_material(slot) if slots else None
                saved.append((comp2, slot, orig))
                comp2.set_material(slot, mats[st])
            n += 1
    rec("swap", n > 0, "替换 %d 个物体" % n)

    for _ in range(4):
        comp.capture_scene()
    time.sleep(0.3)
    unreal.RenderingLibrary.export_render_target(None, rt, OUT, "semantic_enc.exr")
    for cand in ("semantic_enc.exr", "semantic_enc"):
        p = os.path.join(OUT, cand)
        if os.path.exists(p) and os.path.getsize(p) > 100:
            final = os.path.join(OUT, "semantic_enc.exr")
            if p != final:
                os.replace(p, final)
            rec("export", True, final)
            break

    # 还原
    for comp2, slot, orig in saved:
        try:
            comp2.set_material(slot, orig)
        except Exception:
            pass
    rec("restore", True, "%d 槽" % len(saved))

    report["encode"] = {"target_output": se.TARGET_OUTPUT,
                        "encode_input": se.ENCODE_INPUT}
    report["map"] = MAP
    report["resolution"] = [w, h]

except Exception:
    report["fatal"] = traceback.format_exc()[:1500]

with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("DONE")
