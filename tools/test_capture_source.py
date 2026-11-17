import unreal, json, time

r = {}
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
les.load_level("/Game/Maps/codex_validation")
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

mat = unreal.load_asset("/Game/Validation/M_SemanticMask")
if mat is None:
    r["mat"] = "NOT FOUND"

def make_capture():
    rt = unreal.RenderingLibrary.create_render_target2d(
        world, 160, 90, unreal.TextureRenderTargetFormat.RTF_RGBA8)
    sc = eas.spawn_actor_from_class(
        unreal.SceneCapture2D, unreal.Vector(-900, 0, 180), unreal.Rotator(0, 0, 0))
    comp = sc.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", rt)
    comp.set_editor_property("fov_angle", 90.0)
    comp.set_editor_property("capture_every_frame", False)
    return sc, comp, rt

def sample(rt):
    px = unreal.RenderingLibrary.read_render_target_pixel(None, rt, 80, 80)
    return [px.r, px.g, px.b]

def attach(comp, m):
    s = comp.get_editor_property("post_process_settings")
    wb = s.get_editor_property("weighted_blendables")
    arr = wb.get_editor_property("array")
    e = unreal.WeightedBlendable()
    e.set_editor_property("weight", 1.0)
    e.set_editor_property("object", m)
    arr.append(e)
    wb.set_editor_property("array", arr)
    s.set_editor_property("weighted_blendables", wb)
    comp.set_editor_property("post_process_settings", s)

# 测试不同 CaptureSource 下材质是否生效
for src_name in ["SCS_FINAL_COLOR_LDR", "SCS_FINAL_COLOR_HDR", "SCS_SCENE_COLOR_HDR"]:
    src = getattr(unreal.SceneCaptureSource, src_name, None)
    if src is None:
        r[src_name] = "enum missing"
        continue
    key = src_name
    try:
        # A: 不挂材质
        sc1, c1, rt1 = make_capture()
        c1.set_editor_property("capture_source", src)
        c1.capture_scene()
        time.sleep(0.6)
        r[key + "_noMat"] = sample(rt1)
        eas.destroy_actor(sc1)

        # B: 挂材质
        sc2, c2, rt2 = make_capture()
        c2.set_editor_property("capture_source", src)
        attach(c2, mat)
        c2.capture_scene()
        time.sleep(0.6)
        r[key + "_withMat"] = sample(rt2)
        eas.destroy_actor(sc2)
    except Exception as e:
        r[key] = "ERR " + str(e)[:100]

with open("/tmp/csrc.json", "w") as f:
    json.dump(r, f, indent=1, ensure_ascii=False)
print("DONE")
