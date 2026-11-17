import os
import sys
import unreal

sys.path.insert(0, os.path.dirname(__file__))
import batch_capture as bc

if not unreal.EditorLevelLibrary.load_level('/Game/Maps/codex_validation'):
    raise RuntimeError('无法加载验证关卡')
world = unreal.EditorLevelLibrary.get_editor_world()
light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 500), unreal.Rotator(-45, -30, 0))
sky = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 500))
light_comp = light.get_component_by_class(unreal.DirectionalLightComponent)
light_comp.set_editor_property('mobility', unreal.ComponentMobility.MOVABLE)
light_comp.set_editor_property('intensity', 15.0)
sky_comp = sky.get_component_by_class(unreal.SkyLightComponent)
sky_comp.set_editor_property('mobility', unreal.ComponentMobility.MOVABLE)
sky_comp.set_editor_property('intensity', 2.0)
cam = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(-1100, 0, 300), unreal.Rotator(-8, 0, 0))
comp = bc._scene_capture_component(cam)
rt = bc._create_render_target('preview', 640, 360, 'RTF_RGBA8')
comp.set_editor_property('texture_target', rt)
comp.set_editor_property('capture_source', unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property('fov_angle', 90.0)
comp.set_editor_property('always_persist_rendering_state', True)
comp.capture_scene()
comp.capture_scene()
path = os.path.join(unreal.Paths.project_saved_dir(), 'web_preview.png')
actual = bc._save_render_target(rt, path)
if not os.path.isfile(actual) or os.path.getsize(actual) < 100:
    raise RuntimeError('预览图片生成失败: ' + str(actual))
print('PREVIEW_IMAGE=' + actual)
