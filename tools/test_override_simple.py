# test_override_simple.py — 用「材质替换」做语义通道的最小验证
#
# 背景：实测两条后处理路径（SceneCapture.PostProcessSettings / PostProcessVolume）
#   在无头 -ExecutePythonScript 下完全不生效。改试材质替换：
#   把 Road 整体换成 Unlit 纯色材质，整条路面应该变成该颜色——用图像统计判断，避免逐像素采样太慢。
#
# 关键做法：不用 read_render_target_pixel 逐点读（在大分辨率下会卡死/超时），
#   而是导出图片后用 OpenCV 统计（Python 侧完成）。
import unreal, json, time, traceback, os

r = {"steps": []}


def rec(name, ok, value=""):
    r["steps"].append({"step": name, "ok": ok, "value": str(value)[:200]})


try:
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    les.load_level("/Game/Maps/codex_validation")
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1000), unreal.Rotator(-40, -30, 0))
    c = sun.get_component_by_class(unreal.DirectionalLightComponent)
    c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    c.set_editor_property("intensity", 10.0)
    rec("lights", True)

    mat = unreal.load_asset("/Game/Validation/M_SemDebug")
    rec("load_mat", mat is not None)

    road = None
    for a in eas.get_all_level_actors():
        if a.get_actor_label() == "Road":
            road = a
            break
    rec("find_road", road is not None)

    rt = unreal.RenderingLibrary.create_render_target2d(
        world, 320, 180, unreal.TextureRenderTargetFormat.RTF_RGBA16F)
    cam = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(-900, 0, 180), unreal.Rotator(0, 0, 0))
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", rt)
    comp.set_editor_property("fov_angle", 90.0)
    comp.set_editor_property("always_persist_rendering_state", True)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR)

    os.makedirs("/tmp/ovr", exist_ok=True)

    for _ in range(5):
        comp.capture_scene()
    time.sleep(0.8)
    unreal.RenderingLibrary.export_render_target(None, rt, "/tmp/ovr", "before.exr")
    rec("export_before", True)

    if road is not None:
        mesh = road.get_component_by_class(unreal.StaticMeshComponent)
        try:
            mesh.set_material(0, mat)
            rec("override", True)
        except Exception as e:
            rec("override", False, str(e)[:120])

        for _ in range(6):
            comp.capture_scene()
        time.sleep(1.0)
        unreal.RenderingLibrary.export_render_target(None, rt, "/tmp/ovr", "after.exr")
        rec("export_after", True)

except Exception:
    r["fatal"] = traceback.format_exc()[:1000]

with open("/tmp/ovr/result.json", "w") as f:
    json.dump(r, f, indent=1, ensure_ascii=False)
print("WROTE")
