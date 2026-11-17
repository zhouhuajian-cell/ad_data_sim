import os
import sys
import unreal

sys.path.insert(0, os.path.dirname(__file__))
import batch_capture as bc

if not unreal.EditorLevelLibrary.load_level('/Game/Maps/codex_validation'):
    raise RuntimeError('无法加载验证关卡')

# 光照：每次都补一套 MOVABLE 光源。
# ⚠️ 不能"关卡里已有太阳光就跳过"：保存进关卡的光源在无头 -ExecutePythonScript 环境下
#   mobility 可能是 Static，没有经过构建光照就不发光，画面会暗到 max≈0.005（实测）。
#   这里新增的 MOVABLE 光会立即生效；多个预览进程互不影响（进程结束即销毁）。
# 说明：这些参数（角度 -40/-30、强度 10 与天光 1）与 batch_capture 出图时手动补的那套对齐，
#   用同样的光才能让预览亮度和成品图一致（实测强度 10 + 天光 2 时 max 只有 0.30，成品图是 0.79）。
light = unreal.EditorLevelLibrary.spawn_actor_from_class(
    unreal.DirectionalLight, unreal.Vector(0, 0, 1000), unreal.Rotator(-40, -30, 0))
light_comp = light.get_component_by_class(unreal.DirectionalLightComponent)
light_comp.set_editor_property('mobility', unreal.ComponentMobility.MOVABLE)
light_comp.set_editor_property('intensity', 10.0)

sky = unreal.EditorLevelLibrary.spawn_actor_from_class(
    unreal.SkyLight, unreal.Vector(0, 0, 1000))
sky_comp = sky.get_component_by_class(unreal.SkyLightComponent)
sky_comp.set_editor_property('mobility', unreal.ComponentMobility.MOVABLE)
sky_comp.set_editor_property('intensity', 1.0)
sky_comp.set_editor_property('real_time_capture', True)

cam = unreal.EditorLevelLibrary.spawn_actor_from_class(
    unreal.SceneCapture2D, unreal.Vector(-900, 0, 180), unreal.Rotator(0, 0, 0))
comp = bc._scene_capture_component(cam)
# 与 batch_capture 的 RGB 通道保持一致：16F 线性 + SCS_SCENE_COLOR_HDR。
# 用 RTF_RGBA8 + SCS_FINAL_COLOR_LDR 会走色调映射，画面明显偏暗（实测 max 0.09 vs 0.79）。
rt = bc._create_render_target('preview', 640, 360, 'RTF_RGBA16F')
comp.set_editor_property('texture_target', rt)
src, _src_name = bc._resolve_capture_source('rgb')
comp.set_editor_property('capture_source', src)
comp.set_editor_property('fov_angle', 90.0)
comp.set_editor_property('always_persist_rendering_state', True)

# ⚠️ 不要在这里设 AEM_MANUAL + 固定曝光：实测会让画面变成纯黑（max≈0.0003）。
# 自动曝光在无头渲染下确实偏暗，但改它风险更大；亮度后续用增益或光照强度调。

# 预热多帧：首帧伴随着色器编译，自动曝光也需要若干帧才收敛。
# 实测只捕获 2 帧时画面明显偏暗（max≈0.09，正常约 0.79）；连捕 8 帧后亮度接近正常。
for _ in range(8):
    comp.capture_scene()

path = os.path.join(unreal.Paths.project_saved_dir(), 'web_preview.png')
actual = bc._save_render_target(rt, path)
if not os.path.isfile(actual) or os.path.getsize(actual) < 100:
    raise RuntimeError('预览图片生成失败: ' + str(actual))
print('PREVIEW_IMAGE=' + actual)
