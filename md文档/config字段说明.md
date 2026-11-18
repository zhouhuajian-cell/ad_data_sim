# config/ — 四维任务生成样例（queue_gen.py 的输入）

四个文件是「场景 × 光照工况 × 目标组合 × 相机位姿」四维笛卡尔展开的输入，
`queue_gen.py` 按 3×5×6×4 展开出 360 条任务（样例规模，正式批次按需增删）。
字段名同时被 `engine/Content/Python/batch_capture.py`（执行）与
`post/postprocess.py`（按工况取退化参数）、`post/pack_dataset.py`（类别）消费，
改字段要三处一起看。

## scenes.json（场景/关卡）

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | 字符串 | 便于人读的场景名，不参与执行 |
| `map` | 字符串 | UE 资产路径，交给 `unreal.EditorLoadingAndSavingUtils.load_map` |
| `note` | 字符串 | 备注，不参与执行 |

## lights.json（光照工况）

| 字段 | 类型 | 说明 |
|---|---|---|
| `condition` | 字符串 | 工况键：`front_light`/`back_light`/`dusk`/`night`/`rain`；postprocess 按它取曝光与噪声 |
| `label` | 字符串 | 中文名，只用于日志 |
| `sun_elevation_deg` | 浮点 | 太阳高度角（度），负值表示日落之后 |
| `sun_azimuth_deg` | 浮点 | 太阳方位角（度） |
| `sun_intensity` | 浮点 | DirectionalLight 强度（lux 量级，按工程实际再标定） |
| `skylight_intensity` | 浮点 | SkyLight 强度倍率 |
| `weather` | 字符串 | `clear`/`rain`，天气标签进真值 JSON |
| `street_lights` | 布尔 | 夜间/黄昏是否点亮路灯（编辑器内开关，脚本只记录） |

## targets.json（目标组合）

每个元素是一个「组合」，组合内是若干目标；同帧一起生成。

| 字段 | 类型 | 说明 |
|---|---|---|
| `class` | 字符串 | 类别名，必须是 `road`/`vehicle`/`pedestrian`/`traffic_sign`/`cone`/`vegetation` 之一（与语义模板值、pack_dataset 类别表同源） |
| `actor_class` | 字符串 | 要 spawn 的蓝图类路径 |
| `location` | [x,y,z] | 世界坐标（米），建议相对相机锚点摆放 |
| `rotation` | [pitch,yaw,roll] | 目标朝向（度） |
| `jitter` | [dx,dy,dz,dyaw] | 每帧确定性随机抖动的最大绝对值；由任务的 `seed` 固定，可复现 |

## cameras.json（相机位姿与内参）

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | 字符串 | 便于人读，不参与执行 |
| `location` | [x,y,z] | CineCameraActor 世界坐标（米） |
| `rotation` | [pitch,yaw,roll] | 相机朝向（度）；roll 目前保持 0 |
| `focal_length` | 浮点 | 焦距（毫米），样例覆盖 8/12/25 三档 |
| `film_back_width`/`film_back_height` | 浮点 | 胶片背板（毫米），默认 36×24 |
| `aperture` | 浮点 | 光圈 f 值；夜间工况在引擎内按方案表 6 覆盖为 1.8 |
| `shutter_deg` | 浮点 | 快门角度，默认 180 |
| `speed_mps` | 浮点 | 自车速度（米/秒），用于后处理运动模糊核长换算 |

## 生成与执行

```bash
python3 common/queue_gen.py \
  --scenes config/scenes.json --lights config/lights.json \
  --targets config/targets.json --cameras config/cameras.json \
  -o /data/queue/tasks.jsonl --seed 20261113
```

`queue_gen.py` 会用固定序号生成 `id`/`frame_id`（`av_000000` 起），同参数重跑输出逐字节一致。
dashboard 的「任务生成」卡片默认读 `queue_cfg_dir`（服务器上是 `/opt/ue5_synth/config`），
把本目录同步过去即可直接点按钮。
