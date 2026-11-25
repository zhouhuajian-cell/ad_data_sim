# UE 5.3 GLB 导入流程笔记（已跑通）

日期: 2026-11-23

## 环境
- 引擎: /data/UnrealEngine (UE 5.3.2, 源码编译)
- 必须以普通用户运行: `sudo -u algo`
- root 运行会被拒绝: "Refusing to run with the root privileges"
- 需要虚拟显示: `xvfb-run -a -s "-screen 0 1920x1080x24"`

## 项目配置 (uproject)
插件名必须是真实存在的：
```json
"Plugins": [
  { "Name": "PythonScriptPlugin", "Enabled": true },
  { "Name": "EditorScriptingUtilities", "Enabled": true },
  { "Name": "Interchange", "Enabled": true },
  { "Name": "InterchangeEditor", "Enabled": true }
]
```
注意: `InterchangeOpenGLTF` 不存在（会 fatal error）。glTF 支持在 `Interchange` 插件里
（Modules: GLTFCore, InterchangeImport, ...）。

## GLB 导入（已验证成功）
```python
import unreal
t = unreal.AssetImportTask()
t.set_editor_property("filename", "/path/to/model.glb")
t.set_editor_property("destination_path", "/Game/ImportedTest")
t.set_editor_property("automated", True)
t.set_editor_property("replace_existing", True)
t.set_editor_property("save", True)
unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([t])
```

导入结果 (Kenney sedan.glb):
```
body.body                    [StaticMesh]  150 x 255 x 115 cm
colormap1.colormap1          [Material]    ← 自动生成材质
wheel-front-left/right       [StaticMesh]  35 x 60 x 60 cm
wheel-back-left/right        [StaticMesh]
```
关键: GLB 导入会自动生成 **Material**（不是 Texture），这是它优于现有 SM_* 资产的原因。

## 已知坑
1. 现有 `/Game/Validation/Models/SM_*.uasset` 依赖名为 `colormap` 的 **Material**，
   但磁盘上只有 `colormap.png`（纹理）。导入成 Texture2D 后类型不匹配 →
   `Resolved import ... with a different class` → 严重时 Signal 11 崩溃。
   **解决: 不要用这批 SM_* 资产，改成从 GLB 重新导入。**

2. UE Python 是 3.9，文件写入默认 ascii 编码。写中文日志需显式
   `open(path, "w", encoding="utf-8")`。

3. `-nullrhi` 下资产导入不生效（Texture 生成需要 RHI）。

4. `create_render_target2d(world_obj, ...)` 会报
   "No world was found for object"，需改用 `AssetTools.create_asset`。

5. `SceneComponent.set_world_rotation()` 在 UE5.3 Python 里需要 3 个参数
   (rotation, sweep, teleport)；用 `actor.set_actor_rotation(rot, False)` 更稳。

## 未打通的部分（渲染读回）
- `-run=pythonscript` (Commandlet): 无渲染循环，RT 读出纯黑
- `-ExecutePythonScript`: 脚本执行时机与渲染循环难对齐；
  用 `time.sleep` 阻塞会占住主线程导致 slate tick 不触发
- `export_render_target` 在部分模式下返回 None

结论: UE 的 headless 渲染读回需要进一步研究（可能要用 MovieRenderPipeline
或 -game 模式）。资产导入链路已可稳定使用。
