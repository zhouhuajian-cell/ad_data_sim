# render_semantic_depth.py — 单独渲染语义图与深度图（独立于 batch_capture）
#
# 为什么单独成脚本：语义通道要在**同一个 SceneCapture 上先换材质、再捕获**，
#   而 batch_capture 的通道流程是先配相机再捕获，集成时容易互相干扰。
#   本脚本把"换材质→捕获→还原"整段封装，一次进程只干这一件事，最稳。
#
# 用法（服务器）：
#   xvfb-run -a UnrealEditor-Cmd <uproject> -ExecutePythonScript=<本文件> -Vulkan -unattended
#  环境变量：
#     AD_SEM_MAP      关卡路径（默认 /Game/Maps/codex_validation）
#     AD_SEM_OUT      输出目录（默认 /data/render/semantic）
#     AD_SEM_RES      分辨率 WxH（默认 1920x1080）
#     AD_SEM_CAM      相机 "x,y,z,yaw,pitch"（默认 -900,0,180,0,0）
# 产物：
#   <out>/semantic.png   每像素 = class_id/255（背景 0）
#   <out>/depth.png      R 通道：线性深度 / 200m，裁到 0~1
#   <out>/rgb.png        参考图（同一相机同一帧）
#   <out>/meta.json      类别表与统计
import os
import sys
import json
import time
import traceback

sys.path.insert(0, "/data/avsynth/Content/Python")
import unreal

SEMANTIC_STENCIL = {
    1: "road", 2: "vehicle", 3: "pedestrian", 4: "traffic_sign",
    5: "cone", 6: "vegetation", 7: "landline",
}
MAT_DIR = "/Game/Validation/Semantic"

MAP = os.environ.get("AD_SEM_MAP", "/Game/Maps/codex_validation")
OUT = os.environ.get("AD_SEM_OUT", "/data/render/semantic")
RES = os.environ.get("AD_SEM_RES", "1920x1080")
CAM = os.environ.get("AD_SEM_CAM", "-900,0,180,0,0")

report = {"steps": []}


def rec(name, ok, value=""):
    report["steps"].append({"step": name, "ok": ok, "value": str(value)[:400]})
    print("[sem] %s %s %s" % ("OK " if ok else "ERR", name, value))


def ensure_materials():
    """为每个 stencil 建一个 Unlit 灰度材质（Emissive = class_id/255）。"""
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    out = {}
    for stencil, name in SEMANTIC_STENCIL.items():
        gray = stencil / 255.0
        asset = "M_Sem_%02d_%s" % (stencil, name)
        path = MAT_DIR + "/" + asset
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            out[stencil] = unreal.load_asset(path)
            continue
        mat = tools.create_asset(asset, MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
        mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
        mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
        lib = unreal.MaterialEditingLibrary
        v = lib.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -300, 0)
        v.set_editor_property("constant", unreal.LinearColor(gray, gray, gray, 1.0))
        lib.connect_material_property(v, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        lib.recompile_material(mat)
        unreal.EditorAssetLibrary.save_loaded_asset(mat)
        out[stencil] = mat
    return out


def swap_materials(mats):
    """把带 stencil 的网格换成灰度材质；返回 [(comp, slot, orig)] 以便还原。"""
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    saved = []
    n = 0
    for actor in eas.get_all_level_actors():
        try:
            comp = actor.get_component_by_class(unreal.StaticMeshComponent)
            if comp is None or comp.get_editor_property("static_mesh") is None:
                continue
            st = int(comp.get_editor_property("custom_depth_stencil_value"))
            if st not in mats:
                continue
            slots = comp.get_num_materials()
            for slot in range(max(1, slots)):
                orig = comp.get_material(slot) if slots else None
                saved.append((comp, slot, orig))
                comp.set_material(slot, mats[st])
            n += 1
        except Exception as e:
            rec("swap_warn", False, "%s: %s" % (actor.get_actor_label(), e))
    return saved, n


def restore(saved):
    for comp, slot, orig in saved:
        try:
            comp.set_material(slot, orig)
        except Exception:
            pass


try:
    w, h = [int(x) for x in RES.lower().split("x")]
    cx, cy, cz, cyaw, cpitch = [float(x) for x in CAM.split(",")]
    os.makedirs(OUT, exist_ok=True)

    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    les.load_level(MAP)
    rec("load_level", True, MAP)

    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    # 补 MOVABLE 光：无头环境下关卡里保存的静态光源可能不发光
    sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1000), unreal.Rotator(-40, -30, 0))
    c = sun.get_component_by_class(unreal.DirectionalLightComponent)
    c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    c.set_editor_property("intensity", 10.0)
    sky = eas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 1000))
    sc = sky.get_component_by_class(unreal.SkyLightComponent)
    sc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    sc.set_editor_property("real_time_capture", True)
    rec("lights", True)

    rt = unreal.RenderingLibrary.create_render_target2d(
        world, w, h, unreal.TextureRenderTargetFormat.RTF_RGBA16F)
    cam = eas.spawn_actor_from_class(
        unreal.SceneCapture2D, unreal.Vector(cx, cy, cz), unreal.Rotator(cpitch, cyaw, 0))
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", rt)
    comp.set_editor_property("fov_angle", 90.0)
    comp.set_editor_property("always_persist_rendering_state", True)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR)
    rec("scene_capture", True, "%dx%d" % (w, h))

    def grab(name):
        for _ in range(4):
            comp.capture_scene()
        time.sleep(0.3)
        # export_render_target 对 16F 的 RT 会写出无扩展名或无 ".exr" 的文件，
        # 这里按实际产物重命名成 .exr，下游 OpenCV 才能正确解码。
        unreal.RenderingLibrary.export_render_target(None, rt, OUT, name + ".exr")
        for cand in (name + ".exr", name, name + ".png", name + ".hdr"):
            p = os.path.join(OUT, cand)
            if os.path.exists(p) and os.path.getsize(p) > 100:
                final = os.path.join(OUT, name + ".exr")
                if p != final:
                    os.replace(p, final)
                rec("export_" + name, True, "%s (%d B)" % (final, os.path.getsize(final)))
                return final
        rec("export_" + name, False, "no file")
        return None

    # 1) RGB 参考
    grab("rgb")

    # 2) 语义：换材质 → 捕获 → 还原
    mats = ensure_materials()
    rec("materials", True, "%d 个类别材质" % len(mats))
    saved, n = swap_materials(mats)
    rec("swap", n > 0, "替换 %d 个物体" % n)
    grab("semantic")
    restore(saved)
    rec("restore", True, "%d 槽" % len(saved))

    # 3) 深度：直接用 SceneDepth 捕获源（非线性 NDC），由 Python 侧线性化
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_DEPTH)
    grab("depth_raw")

    report["map"] = MAP
    report["resolution"] = [w, h]
    report["classes"] = SEMANTIC_STENCIL

except Exception:
    report["fatal"] = traceback.format_exc()[:1500]

with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("DONE")
