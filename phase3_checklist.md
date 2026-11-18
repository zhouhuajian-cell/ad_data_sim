# 阶段三 T0 核对清单（编辑器内，一次过完）

> 目的：把 `engine/Content/Python/batch_capture.py` 里 **25 处 `TODO(阶段三)`** 在真实工程上一次性核对完。
> 每条都要留证据（命令输出 / 帧号 / 截图）；**没过不要进入批量生产**，否则问题会以成批废片的形式出现（方案表 11）。
> 本清单只管"脚本与数据的正确性"；反光膜材质、相机标定属文档 3.2/3.3 的手工活，不在此表。

## 0. 前置

- [ ] `bash linux/env_check.sh` → `RESULT: ALL PASS`（20.04 已按 2026-11-13 决定放行试编译，见脚本注释）
- [ ] 工程可加载；`python common/case_check.py /data/avsynth` → `ALL PASS`
- [ ] `python tools/dryrun_check.py` → `自检通过：PASS 19 / FAIL 0`（引擎外链路）
- [ ] 训练 / 抽帧 / VLM 判定全停；`bash linux/gpu_watch.sh` 已在跑（显存、温度有记录）
- [ ] 先只跑 **3~5 帧**（把 `tasks.jsonl` 截前几条），核对完再放全量

## T0-1 16bit RGB 是否真的落盘（硬卡点）

- 为什么：`SceneCapture2D → RTF_RGBA16F → PNG` 这条链是否保 16bit、是否偷偷做了 sRGB，**只能实测**（见 `batch_capture.py` 文件头第 1 条）。
- 怎么验：

```bash
# 引擎跑完 1 帧后（系统 python3）
python3 - <<'PY'
import struct, glob
for p in sorted(glob.glob('/data/render/raw/*_rgb.png'))[:3]:
    h = open(p, 'rb').read(26)
    print(p, 'bit_depth=', h[24], 'color_type=', h[25])
PY
```

- 判定：`bit_depth=16, color_type=2`。若为 `8` → 16bit 链路没通，**停下**改 `_save_render_target()` 的落盘路径（见代码 TODO），不要继续。
- 旁证：`meta/<frame>.json` 的 `image_files.bit_depth` 应与 PNG 头一致（脚本按实际位深记录，不写死）。

## T0-2 线性 / 色调映射

- 为什么：后处理按线性值做曝光；若引擎已把 sRGB/色调曲线烘进像素，整批图像分布会错一层。
- 怎么验：同一帧里含太阳/高光的区域，用 OpenCV 读 16bit 原图，`max()` 应明显 >255 且不是大面积 65535 饱和：

```bash
python3 - <<'PY'
import cv2, glob
p = sorted(glob.glob('/data/render/raw/*_rgb.png'))[0]
a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
print('dtype', a.dtype, 'max', a.max(), 'sat_ratio', (a == 65535).mean())
PY
```

- 判定：`dtype=uint16`、`max` 在数千到数万量级、`sat_ratio` 很小。若 `max≈255` 或大量饱和 → 改自定义后处理材质直接取 SceneColor 并关 tone curve。

## T0-3 语义通道（CustomDepth 模板值）

- 为什么：模板值表必须与 `post/pack_dataset.py` 的类别表同源；天空 255、每类一个值。
- 前置：项目设置 `r.CustomDepth=3`；`apply_semantic_stencil()` 里属性名核对通过（日志无 ⚠️）。
- 怎么验：

```bash
python3 - <<'PY'
import cv2, glob, numpy as np
p = sorted(glob.glob('/data/render/raw/*_sem.png'))[0]
a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
print('unique', sorted(np.unique(a).tolist())[:20])
PY
```

- 判定：唯一值 ⊆ `{1,2,3,4,5,6,255}`，且同一帧里 RGB 有标志/车辆/行人/锥桶时对应模板值像素数 >0；天空区域为 255。
- 若出现渐变/大量中间值 → 语义通道还在走最终颜色，未取 stencil（见 `capture_channels()` 第 3 段 TODO）。

## T0-4 深度 0~200m 线性化

- 为什么：`SCS_SCENE_DEPTH` 是非线性 NDC；不挂线性化材质，深度图不能当米用。
- 怎么验：放一个已知距离的目标（例如标志 10m），读其 bbox 中心像素的 16bit 值：

```bash
python3 - <<'PY'
import cv2, json, glob
p = sorted(glob.glob('/data/render/raw/*_depth.png'))[0]
a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
print('dtype', a.dtype, 'min/max', int(a.min()), int(a.max()))
PY
```

- 判定：`normalize_depth(10) = 3277`（±2%）；距离越远值越大；天空接近 65535。对不上 → 深度材质参数（DepthFarMeters=200）没接。

## T0-5 二维框与像素对齐

- 怎么验（20 帧，表 11 抽检项）：

```bash
python3 tools/overlay_boxes.py --meta /data/render/meta --images /data/render/final \
    --out /data/render/overlay --limit 20
```

- 判定：打开 `overlay/contact_sheet.jpg`，框与目标边缘偏差 **≤2px（1080p）**；无整体平移/镜像/缩放；贴边目标标 `truncated=true`。
- 记录：抽检帧号 + 结论（人工签字）。

## T0-6 断点续跑

- 步骤：跑 50 帧后 `tmux kill-session -t ue5_render`（或 Ctrl-C）→ 原命令重跑。
- 判定：日志 `skip≈已完成帧数`、`/data/queue/done` 标记齐全、**不重复出帧**、补齐剩余帧。

## T0-7 清场（目标不串帧）

- 步骤：连续两帧用不同 `targets`（第一帧有 vehicle，第二帧只有 sign）。
- 判定：第二帧语义图里没有第一帧的模板值（如值 2），RGB 里也没有残留 actor。

## T0-8 主循环容错

- 步骤：把一条任务的 `actor_class` 故意改坏（或指向不存在的资产）。
- 判定：该帧 `fail` 写进 `capture_errors.log`，进程继续跑完后退出码 **1**；修好后重跑只补该帧。
- 说明：有失败帧时退出码为 1 是**预期**（批次已跑完，不是整批失败）。

## T0-9 相机 / 光照参数是否真的生效

- 判定：`batch.log` 里没有 `⚠️ 设置属性失败`；否则按日志逐条修属性名。
- 旁证：8mm 与 25mm 两帧视场明显不同；night 与 dusk 两帧亮度不同；夜间 `aperture=1.8`（见 `parse_camera_spec()`）。
- **原则**：`_try_set_editor_property()` 失败只告警不中断，是为了无人值守；所以"跑完了"不等于"参数生效了"，这条必须人工确认。

## T0-10 资源与产能

- `gpu_watch.csv`：显存峰值 <10.5G、无 OOM、温度不触发降频。
- 100 帧标定单帧耗时 → 按方案 6.1（有效占比 0.85）折算日产；记录到批次台账。
- 显存超限处置顺序：降阴影分辨率 → 减同屏目标 → **不要先降分辨率**（改变小目标像素占位）。

## T0-11 已知缺口（登记即可，不在 T0 修）

- `rain` / `street_lights` 目前只记录进真值，不驱动渲染（需工程内粒子/路灯蓝图）。
- 材质随机化只抽了档位号，未接资产库（`material_variant`）。
- 快门角→秒按 24fps 折算，项目实际 24/30 待定。
- 语义/深度两条通道的材质属工程内资产，不在本仓。

## 验收记录

| 日期 | 执行人 | 项 | 结果 | 证据（帧号/日志/截图） |
|---|---|---|---|---|
| | | T0-1 | | |
| | | T0-2 | | |
| | | T0-3 | | |
| | | T0-4 | | |
| | | T0-5 | | |
| | | T0-6 | | |
| | | T0-7 | | |
| | | T0-8 | | |
| | | T0-9 | | |
| | | T0-10 | | |

**出口条件**：T0-1 ~ T0-10 全 PASS，或未过项已明确登记豁免与影响，才进入全量生产。
