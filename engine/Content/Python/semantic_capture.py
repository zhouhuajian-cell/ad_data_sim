# semantic_capture.py — 语义/深度通道：用「材质替换」实现（不走后处理）
#
# 为什么不用后处理：
#   实测两条后处理路径在无头 -ExecutePythonScript 下**完全不生效**（采样值与基线逐位相同）：
#     1) SceneCapture2D.post_process_settings.weighted_blendables
#     2) 关卡里的 PostProcessVolume.settings.weighted_blendables
#   而「把物体的材质临时替换成 Unlit 纯色材质」是有效的（实测地面区均值 0.13 → 0.55）。
#
# 语义通道做法：
#   1) 为每个类别建一个 Unlit 材质，Emissive = class_id/255 的灰度；
#   2) 遍历场景 Actor，按 CustomDepthStencilValue 找到类别，记下原材质并替换；
#   3) 捕获 → 还原材质。
#   输出的每个像素值 = class_id/255，背景（天空）为 0。
#
# 深度通道做法：
#   同理，用「LinearDepth 转灰度」的材质替换全部物体，背景天空给最远值。
#   注意：这条路要求每个物体都有材质槽才能替换；无材质的（如天空）保持 0。
import unreal

SEMANTIC_STENCIL = {
    1: "road",
    2: "vehicle",
    3: "pedestrian",
    4: "traffic_sign",
    5: "cone",
    6: "vegetation",
    7: "landline",
}

MATERIAL_DIR = "/Game/Validation/Semantic"

_sem_materials = {}
_saved_materials = []   # [(component, slot, original_material)]


def _log(msg):
    unreal.log("[semantic] " + msg)
    print("[semantic] " + msg)


def ensure_semantic_materials():
    """为每个 stencil 值建一个 Unlit 灰度材质；已存在则复用。"""
    global _sem_materials
    if _sem_materials:
        return _sem_materials

    tools = unreal.AssetToolsHelpers.get_asset_tools()
    for stencil, name in SEMANTIC_STENCIL.items():
        gray = stencil / 255.0
        asset_name = "M_Sem_%02d_%s" % (stencil, name)
        path = MATERIAL_DIR + "/" + asset_name

        exists = unreal.EditorAssetLibrary.does_asset_exist(path)
        if exists:
            mat = unreal.load_asset(path)
        else:
            mat = tools.create_asset(asset_name, MATERIAL_DIR, unreal.Material,
                                     unreal.MaterialFactoryNew())
            mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
            mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
            lib = unreal.MaterialEditingLibrary
            v = lib.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -300, 0)
            v.set_editor_property("constant", unreal.LinearColor(gray, gray, gray, 1.0))
            lib.connect_material_property(v, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            lib.recompile_material(mat)
            unreal.EditorAssetLibrary.save_loaded_asset(mat)
        _sem_materials[stencil] = mat

    _log("语义材质就绪：%d 个类别" % len(_sem_materials))
    return _sem_materials


def apply_semantic_materials():
    """把场景里所有带 CustomDepthStencilValue 的网格换成对应灰度材质；返回替换数量。"""
    global _saved_materials
    _saved_materials = []
    mats = ensure_semantic_materials()
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    count = 0

    for actor in eas.get_all_level_actors():
        try:
            comp = actor.get_component_by_class(unreal.StaticMeshComponent)
            if comp is None or comp.get_editor_property("static_mesh") is None:
                continue
            stencil = int(comp.get_editor_property("custom_depth_stencil_value"))
            if stencil not in mats:
                continue
            n_slots = comp.get_num_materials()
            for slot in range(max(1, n_slots)):
                try:
                    orig = comp.get_material(slot) if n_slots else None
                except Exception:
                    orig = None
                _saved_materials.append((comp, slot, orig))
                comp.set_material(slot, mats[stencil])
            count += 1
        except Exception as e:
            _log("替换失败 %s: %s" % (actor.get_actor_label(), e))

    _log("已替换 %d 个物体的材质（记录 %d 个原材质槽）" % (count, len(_saved_materials)))
    return count


def restore_materials():
    """捕获完成后把原材质还回去。"""
    global _saved_materials
    n = 0
    for comp, slot, orig in _saved_materials:
        try:
            comp.set_material(slot, orig)
            n += 1
        except Exception:
            pass
    _saved_materials = []
    _log("已还原 %d 个材质槽" % n)
    return n


def capture_semantic(comp, rt, save_fn, path, warm_frames=3):
    """替换材质 → 捕获 → 还原。返回保存后的实际路径。

    comp       SceneCaptureComponent2D
    rt         语义通道的渲染目标
    save_fn    batch_capture._save_render_target
    path       输出路径
    """
    apply_semantic_materials()
    try:
        comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR)
        for _ in range(warm_frames):
            comp.capture_scene()
        return save_fn(rt, path)
    finally:
        restore_materials()
