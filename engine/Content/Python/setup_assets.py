# -*- coding: utf-8 -*-
"""Create isolated validation assets and import staged images in UE Editor.

Invoked by the dashboard via UnrealEditor-Cmd. Existing user assets are never replaced.
"""
import os
import unreal

ROOT = "/Game/Validation"
MAP = "/Game/Maps/codex_validation"

def log(msg):
    unreal.log("[setup_assets] " + msg)
    print("[setup_assets] " + msg)

def material(name, scene_id, divisor):
    path = ROOT + "/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        log("已有材质，保留: " + path)
        return
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset(name, ROOT, unreal.Material, unreal.MaterialFactoryNew())
    if not mat:
        raise RuntimeError("创建材质失败: " + path)
    mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    lib = unreal.MaterialEditingLibrary
    sample = lib.create_material_expression(mat, unreal.MaterialExpressionSceneTexture, -400, 0)
    sample.set_editor_property("scene_texture_id", scene_id)
    scale = lib.create_material_expression(mat, unreal.MaterialExpressionMultiply, -150, 0)
    scale.set_editor_property("const_b", 1.0 / divisor)
    if not lib.connect_material_expressions(sample, "", scale, "A"):
        raise RuntimeError("材质节点连线失败: " + path)
    if not lib.connect_material_property(scale, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR):
        raise RuntimeError("材质输出连线失败: " + path)
    lib.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    log("已创建材质: " + path)

def spawn_cube(mesh, name, location, scale, stencil):
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(*location), unreal.Rotator(0, 0, 0))
    if not actor:
        raise RuntimeError("生成 Actor 失败: " + name)
    actor.set_actor_label(name)
    actor.set_actor_scale3d(unreal.Vector(*scale))
    comp = actor.get_editor_property("static_mesh_component")
    comp.set_static_mesh(mesh)
    comp.set_editor_property("render_custom_depth", True)
    comp.set_editor_property("custom_depth_stencil_value", stencil)

def create_map():
    if unreal.EditorAssetLibrary.does_asset_exist(MAP):
        log("已有验证关卡，保留: " + MAP)
        return
    if not unreal.EditorLevelLibrary.new_level(MAP):
        raise RuntimeError("创建关卡失败: " + MAP)
    cube = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    if not cube:
        raise RuntimeError("引擎基础 Cube 资产不存在")
    spawn_cube(cube, "Road", (0, 0, -55), (40, 8, 1), 1)
    spawn_cube(cube, "LaneMarkLeft", (0, -200, 0), (40, 0.08, 0.02), 1)
    spawn_cube(cube, "LaneMarkRight", (0, 200, 0), (40, 0.08, 0.02), 1)
    spawn_cube(cube, "TrafficSignProxy", (800, -250, 130), (0.1, 0.8, 0.8), 4)
    spawn_cube(cube, "VehicleProxy", (1400, 100, 80), (2, 1, 0.8), 2)
    spawn_cube(cube, "ConeProxy", (650, 230, 50), (0.5, 0.5, 1), 5)
    if not unreal.EditorLevelLibrary.save_current_level():
        raise RuntimeError("保存关卡失败: " + MAP)
    log("已创建验证关卡: " + MAP)

def import_images():
    project = unreal.Paths.project_dir()
    inbox = os.path.join(project, "AssetInbox")
    if not os.path.isdir(inbox):
        return
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    for name in sorted(os.listdir(inbox)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        asset_name = os.path.splitext(name)[0]
        target = ROOT + "/Textures/" + asset_name
        if unreal.EditorAssetLibrary.does_asset_exist(target):
            log("已有纹理，保留: " + target)
            continue
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", os.path.join(inbox, name))
        task.set_editor_property("destination_path", ROOT + "/Textures")
        task.set_editor_property("automated", True)
        task.set_editor_property("save", True)
        task.set_editor_property("replace_existing", False)
        tools.import_asset_tasks([task])
        if not unreal.EditorAssetLibrary.does_asset_exist(target):
            raise RuntimeError("图片导入失败: " + name)
        log("已导入纹理: " + target)

try:
    material("M_LinearDepth", unreal.SceneTextureId.PPI_SCENE_DEPTH, 20000.0)
    material("M_SemanticMask", unreal.SceneTextureId.PPI_CUSTOM_STENCIL, 255.0)
    create_map()
    import_images()
    log("完成。材质尚需接入捕获通道并做像素验收。")
except Exception as exc:
    unreal.log_error("[setup_assets] " + str(exc))
    raise
