# test_pp_volume.py — 用 PostProcessVolume 替代 SceneCapture.PostProcessSettings
#
# 背景（实测）：把后处理材质塞进 SceneCapture2D 的 post_process_settings.weighted_blendables，
#   5 个采样点与不挂材质时**完全一致**，说明该路径在无头 -ExecutePythonScript 下不生效。
#   这是 UE 的已知限制，不是材质本身的问题（材质节点、blendable_location 均已验证正确）。
#
# 本脚本验证另一条路：在关卡里放一个 PostProcessVolume（无限范围），
#   把材质加到它的 Settings.WeightedBlendables，看 SceneCapture 是否吃这个。
#   - 生效 → 深度/语义通道改用"渲染前临时建体积、渲染后删除"
#   - 不生效 → 只能改用材质替换（MaterialOverride）或分通道后处理材质
import unreal, json, time, traceback

r = {"steps": []}


def rec(name, ok, value=""):
    r["steps"].append({"step": name, "ok": ok, "value": str(value)[:300]})


try:
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    les.load_level("/Game/Maps/codex_validation")
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1000), unreal.Rotator(-40, -30, 0))
    c = sun.get_component_by_class(unreal.DirectionalLightComponent)
    c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    c.set_editor_property("intensity", 10.0)
    sky = eas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 1000))
    sc = sky.get_component_by_class(unreal.SkyLightComponent)
    sc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    sc.set_editor_property("real_time_capture", True)
    rec("setup_lights", True)

    red = unreal.load_asset("/Game/Validation/M_TestRed")
    rec("load_red_material", red is not None)

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
            vals.append((round(px.r, 2), round(px.g, 2), round(px.b, 2)))
        rec("sample_" + tag, True, vals)
        return vals

    for _ in range(5):
        comp.capture_scene()
    time.sleep(0.8)
    base = sample("baseline")

    # 建 PostProcessVolume（无限范围 = unbound）
    vol = eas.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    try:
        vol.set_editor_property("unbound", True)
    except Exception:
        try:
            vol.set_actor_label("PPV_Global")
        except Exception:
            pass
    rec("spawn_ppv", True, vol.get_actor_label())

    # 把材质加到体积的 WeightedBlendables
    ok_attach = False
    try:
        settings = vol.get_editor_property("settings")
        wb = settings.get_editor_property("weighted_blendables")
        arr = wb.get_editor_property("array")
        e = unreal.WeightedBlendable()
        e.set_editor_property("weight", 1.0)
        e.set_editor_property("object", red)
        arr.append(e)
        wb.set_editor_property("array", arr)
        settings.set_editor_property("weighted_blendables", wb)
        vol.set_editor_property("settings", settings)
        ok_attach = len(arr) > 0
    except Exception as ex:
        rec("attach_to_ppv_err", False, str(ex)[:200])
    rec("attach_to_ppv", ok_attach)

    for _ in range(6):
        comp.capture_scene()
    time.sleep(1.0)
    after = sample("with_ppv")

    changed = sum(1 for a, b in zip(base, after) if a != b)
    rec("verdict", True,
        "变化 %d/%d → %s" % (changed, len(base),
                            "PostProcessVolume 生效" if changed > 0 else "PostProcessVolume 也不生效"))

except Exception:
    r["fatal"] = traceback.format_exc()[:1200]

with open("/tmp/ppv_test.json", "w") as f:
    json.dump(r, f, indent=1, ensure_ascii=False)
print("WROTE")
