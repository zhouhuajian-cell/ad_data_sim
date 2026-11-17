# test_pp_material.py — 判定 SceneCapture2D 是否真的应用 PostProcessSettings 里的后处理材质
#
# 干什么：造一个"输出纯红"的后处理材质，挂到 SceneCapture 上捕获一帧，多点采样中心像素。
#   - 读到 R 明显变大、G/B 变小 → 后处理链生效，此前深度/语义材质不生效是材质本身的问题
#   - 与不挂材质完全一致        → SceneCapture 不应用后处理，要换方案
#
# 怎么跑（服务器）：
#   xvfb-run -a UnrealEditor-Cmd <uproject> -ExecutePythonScript=<本文件> -Vulkan -unattended
# 结果写到 /tmp/pp_test.json
import unreal, json, time, traceback

r = {"steps": []}


def rec(name, ok, value=""):
    r["steps"].append({"step": name, "ok": ok, "value": str(value)[:200]})


try:
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    les.load_level("/Game/Maps/codex_validation")
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    rec("load_level", True)

    # 补 MOVABLE 光（无头环境下关卡里的静态光源不发光）
    sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1000), unreal.Rotator(-40, -30, 0))
    c = sun.get_component_by_class(unreal.DirectionalLightComponent)
    c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    c.set_editor_property("intensity", 10.0)
    sky = eas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 1000))
    sc = sky.get_component_by_class(unreal.SkyLightComponent)
    sc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    sc.set_editor_property("real_time_capture", True)
    rec("spawn_lights", True)

    # 造纯红后处理材质（BL_AFTER_TONEMAPPING 是合法值）
    PATH = "/Game/Validation/M_TestRed"
    if unreal.EditorAssetLibrary.does_asset_exist(PATH):
        unreal.EditorAssetLibrary.delete_asset(PATH)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset("M_TestRed", "/Game/Validation", unreal.Material, unreal.MaterialFactoryNew())
    mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    mat.set_editor_property("blendable_location", unreal.BlendableLocation.BL_AFTER_TONEMAPPING)
    lib = unreal.MaterialEditingLibrary
    const = lib.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -300, 0)
    const.set_editor_property("constant", unreal.LinearColor(1.0, 0.0, 0.0, 1.0))
    connected = lib.connect_material_property(const, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    lib.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    red = unreal.load_asset(PATH)
    rec("make_red_material", red is not None, "connected=%s" % connected)

    # SceneCapture
    rt = unreal.RenderingLibrary.create_render_target2d(
        world, 320, 180, unreal.TextureRenderTargetFormat.RTF_RGBA16F)
    cam = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(-900, 0, 180), unreal.Rotator(0, 0, 0))
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", rt)
    comp.set_editor_property("fov_angle", 90.0)
    comp.set_editor_property("always_persist_rendering_state", True)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR)

    def sample(tag):
        pts = [(80, 45), (160, 90), (160, 150), (240, 120), (60, 160)]
        vals = []
        for (x, y) in pts:
            px = unreal.RenderingLibrary.read_render_target_pixel(None, rt, x, y)
            vals.append((round(px.r, 3), round(px.g, 3), round(px.b, 3)))
        rec("sample_" + tag, True, vals)
        return vals

    for _ in range(5):
        comp.capture_scene()
    time.sleep(0.8)
    base = sample("no_material")

    # 挂纯红材质
    s = comp.get_editor_property("post_process_settings")
    wb = s.get_editor_property("weighted_blendables")
    arr = wb.get_editor_property("array")
    e = unreal.WeightedBlendable()
    e.set_editor_property("weight", 1.0)
    e.set_editor_property("object", red)
    arr.append(e)
    wb.set_editor_property("array", arr)
    s.set_editor_property("weighted_blendables", wb)
    comp.set_editor_property("post_process_settings", s)
    rec("attach_red", True, "len=%d" % len(arr))

    for _ in range(5):
        comp.capture_scene()
    time.sleep(0.8)
    after = sample("with_red_material")

    # 判定
    changed = sum(1 for a, b in zip(base, after) if a != b)
    rec("verdict", True,
        "变化的采样点 %d/%d → %s" % (changed, len(base),
                                   "后处理生效" if changed > 0 else "后处理未生效"))

except Exception:
    r["fatal"] = traceback.format_exc()[:1200]

with open("/tmp/pp_test.json", "w") as f:
    json.dump(r, f, indent=1, ensure_ascii=False)
print("WROTE")
