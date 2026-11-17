# test_semantic_override.py — 语义通道改用「材质替换」而非后处理
#
# 背景：实测 SceneCapture.PostProcessSettings 和 PostProcessVolume 两条后处理路径
#   在无头 -ExecutePythonScript 下都**完全不生效**（采样值与基线逐位相同）。
#   所以深度/语义不能靠后处理材质。
#
# 本方案（UE 做语义分割的常规做法）：
#   1) 为每个类别建一个 Unlit 材质，Emissive = 类别编码色（灰度值 = class_id/255）；
#   2) 渲染语义通道前，遍历场景物体的 StaticMeshComponent，
#      用 set_material(0, 类别材质) 整体替换（原材质记录下来，渲染后还原）；
#   3) 捕获 → 得到的就是纯语义图，像素值即类别。
#   4) 深度通道同理：用一个把深度线性化后写成灰度的材质替换（用 LinearDepth 节点）。
#
# 本脚本验证第 1、2、3 步：把 traffic_sign（stencil=4）替换成纯灰 4/255≈0.0157，
#   看捕获结果该物体区域是否变成对应灰度。
import unreal, json, time, traceback

r = {"steps": []}


def rec(name, ok, value=""):
    r["steps"].append({"step": name, "ok": ok, "value": str(value)[:300]})


def make_unlit_gray(name, gray):
    """建一个 Unlit 材质，Emissive = (gray, gray, gray)。"""
    path = "/Game/Validation/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset(name, "/Game/Validation", unreal.Material, unreal.MaterialFactoryNew())
    mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    lib = unreal.MaterialEditingLibrary
    v = lib.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -300, 0)
    v.set_editor_property("constant", unreal.LinearColor(gray, gray, gray, 1.0))
    lib.connect_material_property(v, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    lib.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return unreal.load_asset(path)


try:
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    les.load_level("/Game/Maps/codex_validation")
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1000), unreal.Rotator(-40, -30, 0))
    c = sun.get_component_by_class(unreal.DirectionalLightComponent)
    c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    c.set_editor_property("intensity", 10.0)
    rec("setup_lights", True)

    # 语义色：traffic_sign(4) → 4/255 = 0.0157；这里用更容易分辨的 0.8 灰便于肉眼/数值确认
    GRAY = 0.8
    sem_mat = make_unlit_gray("M_SemDebug", GRAY)
    rec("make_sem_material", sem_mat is not None, "gray=%.3f" % GRAY)

    rt = unreal.RenderingLibrary.create_render_target2d(
        world, 320, 180, unreal.TextureRenderTargetFormat.RTF_RGBA16F)
    cam = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(-900, 0, 180), unreal.Rotator(0, 0, 0))
    comp = cam.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", rt)
    comp.set_editor_property("fov_angle", 90.0)
    comp.set_editor_property("always_persist_rendering_state", True)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR)

    def stats(tag):
        # 统计整帧的均值/最大值，判断是否整体变亮
        px_pts = [(x, y) for x in range(40, 300, 40) for y in range(20, 170, 30)]
        vals = []
        for (x, y) in px_pts:
            px = unreal.RenderingLibrary.read_render_target_pixel(None, rt, x, y)
            vals.append(px.r)
        rec("stats_" + tag, True, "mean=%.4f max=%.4f min=%.4f n=%d"
            % (sum(vals) / len(vals), max(vals), min(vals), len(vals)))
        return vals

    for _ in range(5):
        comp.capture_scene()
    time.sleep(0.8)
    base = stats("baseline")

    # 找到 traffic_sign 那个 Actor 并替换材质
    target = None
    for a in eas.get_all_level_actors():
        if "TrafficSign" in a.get_actor_label():
            target = a
            break
    rec("find_traffic_sign", target is not None, target.get_actor_label() if target else "not found")

    if target:
        mesh = target.get_component_by_class(unreal.StaticMeshComponent)
        orig = mesh.get_editor_property("override_materials") if hasattr(mesh, "override_materials") else None
        try:
            mesh.set_material(0, sem_mat)
            rec("override_material", True)
        except Exception as e:
            rec("override_material", False, str(e)[:150])

        for _ in range(6):
            comp.capture_scene()
        time.sleep(1.0)
        after = stats("after_override")

        changed = sum(1 for a, b in zip(base, after) if abs(a - b) > 0.01)
        rec("verdict", True,
            "亮度变化采样点 %d/%d → %s" % (changed, len(base),
                                       "材质替换生效" if changed > 0
                                       else "材质替换也没变化（可能该物体不在画面内）"))

except Exception:
    r["fatal"] = traceback.format_exc()[:1500]

with open("/tmp/sem_override.json", "w") as f:
    json.dump(r, f, indent=1, ensure_ascii=False)
print("WROTE")
