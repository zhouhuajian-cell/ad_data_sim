# PerceptionSim 感知仿真平台

> 文档入口：[文档目录](docs/README.md)。本文件包含历史配置与操作，实际部署和训练交付请先核对当前服务器版本、任务标注和质量门禁。

> 面向自动驾驶感知训练数据生产的合成平台。支持 **TSR**（交通标志替换）、**OBJ**（3D 目标植入）、**GND**（地面标线重绘）三大模块，输出带标注的合成图像。

- **访问地址**：`http://10.2.248.34:8010/`
- **服务器**：`root@10.2.248.34`
- **项目根**：`/opt/ue5_synth`
- **Python**：`/opt/ad_mining/venv/bin/python`（ML，Py3.8 + torch 2.4.1+cu121）／`/usr/bin/python3`（服务端，无 cv2）
- **GPU**：RTX 4070 Ti 12GB
- **前端**：`/opt/ue5_synth/dashboard/index.html`（单文件，约 488K 字符）
- **后端**：`/opt/ue5_synth/dashboard/server.py`（约 190KB）

---

## 目录

1. [快速开始](#1-快速开始)
2. [系统架构](#2-系统架构)
3. [模块一：TSR 标志替换](#3-模块一tsr-标志替换)
4. [模块二：OBJ 目标植入](#4-模块二obj-目标植入)
5. [模块三：GND 地面标线](#5-模块三gnd-地面标线)
6. [数据目录](#6-数据目录)
7. [API 接口](#7-api-接口)
8. [运维命令](#8-运维命令)
9. [关键设计决策](#9-关键设计决策)
10. [已知问题](#10-已知问题)
11. [变更历史](#11-变更历史)

---

## 1. 快速开始

### 1.1 启动服务

```bash
pkill -9 -f "server.py 8010"
cd /opt/ue5_synth/dashboard
setsid nohup env HF_ENDPOINT=https://hf-mirror.com \
  python3 server.py 8010 > /tmp/srv.log 2>&1 < /dev/null &
```

浏览器打开 `http://10.2.248.34:8010/`，顶部切换 `TSR / OBJ / GND`。

### 1.2 平台操作

| 步骤 | TSR | OBJ | GND |
|---|---|---|---|
| 选数据 | 顶部 `国家` 下拉 | 顶部 `OBJ` 下拉选 batch | 顶部 `GND` 下拉选 batch |
| 调参数 | 右侧 `仿真参数` | 右侧 `OBJ 参数` | 右侧 `GND 参数` |
| 应用 | 点「开始仿真」 | 点「应用参数并重绘」 | 点「应用参数并重绘」 |
| 看结果 | 缩略图 + 全屏双栏 | 左右对比（左原图右仿真） | 左右对比 |

### 1.3 全屏查看

点右上角全屏按钮，左右双栏对比：

- **左栏**：原图（绿色 BBox）
- **右栏**：仿真图（橙色 BBox）
- **按钮**：`1x/2x/3x/4x/6x/8x` 缩放、`重置`、`◀▶` 翻帧、`播放`、`BBox` 开关、`✕` 关闭
- **缩放时 BBox 字号自动反向补偿**，视觉大小恒定

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│  前端 dashboard/index.html（单文件，无构建）              │
│  ├─ 三模块切换（TSR / OBJ / GND）                        │
│  ├─ 双栏对比（左原图 + 右仿真）                          │
│  ├─ 全屏查看（缩放/播放/BBox）                           │
│  └─ Canvas BBox 叠加（cpCv / cpCvSim）                   │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP JSON
┌────────────────────▼────────────────────────────────────┐
│  后端 dashboard/server.py（http.server，单进程多线程）    │
│  ├─ 静态文件 / 图片服务                                  │
│  ├─ 数据源目录查找（多候选回退）                          │
│  └─ 渲染任务队列（_RJ_Q + 后台 worker）                   │
└────────────────────┬────────────────────────────────────┘
                     │ subprocess
┌────────────────────▼────────────────────────────────────┐
│  algorithms tools/*.py（GPU 渲染，独立进程）              │
│  ├─ TSR: batch_replace.py → pipeline6.py                │
│  ├─ OBJ: render_obj_params.py → pipeline_sim.py         │
│  └─ GND: gnd_render_one.py → gnd_clean_pipeline.py      │
└─────────────────────────────────────────────────────────┘
```

### 2.1 渲染任务机制

```python
# server.py
_RJ_Q = queue.Queue()          # 任务队列
_render_worker()               # 单线程消费者
  └─ subprocess.run(algo_python, script, --config payload)
```

- 提交：`POST /api/real/render_params`（OBJ/TSR）或 `/api/gnd/render`（GND）
- 返回：`{"ok": true, "job": "obj_1795338270_8f6387"}`
- 轮询：`GET /api/render/job?id=<job>`
- 状态：`queued` → `running` → `done` / `failed`

---

## 3. 模块一：TSR 标志替换

### 3.1 流程

```
场景图 → YOLO 检测 → 【三层质检漏斗】 → 形状+底色匹配采样
  → pipeline6 六步渲染 → 标注导出
```

### 3.2 三层质检漏斗（`tools/tsr_filter.py`）

> **核心原则**：仿真**怕误检**（宁可漏 10 个真牌，不可换错 1 个三轮车）。

| 层 | 规则 | 参数 |
|---|---|---|
| **1. 硬规则** | 置信度 | `conf >= 0.35`（小目标 `>= 0.30`） |
| | 尺寸 | `>= 12px`（12~24 走小目标档） |
| | 长宽比 | `0.38 <= w/h <= 2.60`（覆盖竖长方形牌） |
| **2. 语义互斥** | 车辆占比 | 常规 `<= 0.30`，小目标 `<= 0.10` |
| | Mask2Former 交叉验证 | `sign_ratio >= 0.25` **或** 软概率 `>= 0.10` |
| **3. 单帧直通** | 高置信 | `conf>=0.50 + veh<=0.10 + (sign>=0.30 或 soft>=0.15)` |

**类别 ID（Mapillary Vistas）**

```
Traffic Sign: 46 Traffic Sign Frame / 49 Traffic Sign (Back) / 50 Traffic Sign (Front)
Vehicle:      22 Rider / 52 Bicycle / 54 Bus / 55 Car / 56 Caravan
              57 Motorcycle / 59 Other Vehicle / 60 Trailer
```

**实测效果**

| 场景 | 无过滤 | 三层漏斗 |
|---|---|---|
| 三轮车头盔误检 | 2 张被贴 ❌ | **0 张** ✅ |
| 行人 | 7 张被贴 ❌ | **0 张** ✅ |
| 卡车 | 3 张被贴 ❌ | **0 张** ✅ |
| 真标牌 | 20/20 | **15~16/20**（差的是 YOLO 本身未检出） |

### 3.3 形状判定（`tools/tsr_shape_cover.py`）

**禁止用 BBox 长宽比** —— 圆/方/等边三角形的 BBox 都是 1:1。

```python
detect_shape_foreground(image, bbox, cls_hint)
  1. 类别映射优先：prohibition/mandatory→circle, warning→triangle, guide→rect
  2. 提取彩色前景（红/蓝/黄，**排除白色背板**）
  3. 与标准圆/三角/矩形模板算 IoU
  4. 颜色投票纠正：蓝底/红底 → circle，黄底 → triangle
  5. 最高 IoU >= 0.65 才放行，否则 unknown
```

**防露底门控**

```python
cover_check(image, bbox, asset_alpha)
  # 原牌彩色前景中，落在素材 alpha 外的比例
  # > 12% → 拒绝（如三角形盖不住圆形牌）
```

实测：圆牌素材 `leak=2.8%` 通过；三角素材 `leak=56.5%` 拒绝。

### 3.4 素材索引（`tools/build_asset_index.py`）

```bash
/opt/ad_mining/venv/bin/python tools/build_asset_index.py \
  --asset-dir /data/tsr_library_v2 \
  --output tools/asset_index.json \
  --workers 16
# 33716 张，约 46 秒
```

**产出分布**

```
circle_blue         2186    circle_red_white    5026
circle_yellow       2575    triangle_red_white  2356
triangle_yellow     1155    rect_blue           6219
rect_other          7518    rect_red_white      3426
```

### 3.5 采样策略（`AssetIndexSampler`）

```
同国家 + 同形状 + 同底色  →  exact
同国家 + 同形状           →  country_shape
全局同形状 + 同底色       →  exact
全局同形状                →  shape_only
都没有                    →  miss（跳过该帧）
```

**开关**：`--match-color`（默认开）／`--no-match-color`（只按形状）

**实测**：40 张产出中 **39 张 exact（97.5%）**，1 张 fallback。

### 3.6 渲染管线（`tools/pipeline6.py`）

| Step | 内容 |
|---|---|
| 1 | 形状强校验（`src_shape == dst_shape`） |
| 2 | 红圈精定位 + 切 `bg_roi` + resize 素材（`pad_ratio=1.04`） |
| 3 | **三锚点颜色对齐**（白底/红圈/黑字分别对齐） |
| 3.5 | 光学模糊 + 噪点（scene_optics 估计的 `blur_sigma`） |
| 4 | 轻微光学退化 |
| 5 | 窄羽化融合（`ksize=3`，不腐蚀） |
| 6 | JPEG 压缩（质量 88 + 红色饱和度补偿） |
| **终态** | 红圈终态校准（补偿 Step4/5/6 累积偏差） |

**红圈修复关键**（曾严重发粉）

```
ORIG      : 红S=101  红V=74
修复前    : 红S= 67  红V=159   ← 发粉
修复后    : 红S= 93  红V= 75   ← 与原图一致 ✅
```

### 3.7 命令行用法

```bash
/opt/ad_mining/venv/bin/python tools/batch_replace.py \
  --images <场景目录> \
  --signs-dir /data/tsr_library_v2 \
  --country VN \
  --all-signs --vary-class --v6 \
  --out /data/render/tsr_v3_10

# 可选参数
--temporal          启用轨迹回溯（默认按帧处理）
--no-gate           关闭质检漏斗（不推荐）
--no-semantic       只做硬规则（快）
--match-color       按形状+底色匹配（默认）
--roi-match         ROI 动态范围对齐（pipeline6 已处理后不需要）
```

---

## 4. 模块二：OBJ 目标植入

### 4.1 流程

```
场景图 + LiDAR → 3D 位置规划 → 精灵渲染 → 投影贴合 → 阴影/遮挡 → 合成
```

### 4.2 参数（前端 `OBJ 参数` 面板）

| 参数 | 说明 | 默认 |
|---|---|---|
| `编辑模式` | 仅新增 / 替换 / 擦除 | 仅新增 |
| `物体类别` | sedan/suv/truck/van/bus/cone/rider/pedestrian | car |
| `指定素材` | 从 216 张素材里选；或程序生成 | 关 |
| `指定位置` | 手动指定前向/横向米数 | 关 |
| `数量` | 每帧植入个数 | 3 |
| `最近/最远距离` | 米 | 8 / 28 |
| `横向范围` | ±米 | 3.0 |
| `尺寸倍率` | | 1.0 |
| `阴影强度` | | 35% |
| `遮挡裁剪` | 被前车遮挡时裁掉 | 开 |
| `光照` | auto / day / night | auto |

### 4.3 关键文件

| 文件 | 功能 |
|---|---|
| `tools/render_obj_params.py` | 前端参数入口，**渲染整个 clip** |
| `tools/pipeline_sim.py` | 主流程：`process_clip()` |
| `tools/obj_sprite.py` | 精灵提取/渲染 |
| `tools/pipeline_sim.py:find_sprite()` | 素材查找（`RANDOM-SPRITES` 支持列表轮询） |

### 4.4 输出

```
/data/render/obj_params/
  ├─ <stem>_obj.jpg       每帧合成图
  └─ annotations/<stem>.json
```

**命令**

```bash
/opt/ad_mining/venv/bin/python tools/render_obj_params.py --config cfg.json
# cfg: {"batch":..., "clip":..., "params":{...}, "out":...}
```

---

## 5. 模块三：GND 地面标线

### 5.1 四步流程（`tools/gnd_clean_pipeline.py`）

```
① 提取旧标线（4A: Mask2Former ∪ v3 Top-Hat）
② LaMa 清零（只跑一次，结果只读缓存）
③ 净面重绘（素材贴图 + 磨损/污渍/反光）
④ 车辆回贴（用原始 raw 的 vehicle_mask）
```

### 5.2 标线提取演进

| 版本 | 方法 | 问题 |
|---|---|---|
| v2 | 亮度比值 + 形态学 | 文字/箭头漏检 |
| v3 | 双边对比度 + 连通域面积上限 8% | 中远景细线仍漏 |
| **4A** | **Mask2Former 软概率 ∪ v3** | ✅ 当前方案 |

**4A 关键改进**：不用 `argmax`（会吞掉远景 1~3px 细线），改用 **Softmax 概率 > 0.12**。

```python
# tools/gnd_mask_4a.py
class_probs = F.softmax(outputs.class_queries_logits, dim=-1)[..., :-1]
mask_probs = outputs.masks_queries_logits.sigmoid()
sem_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
marking_prob = sem_logits[0, self.marking_ids].sum(dim=0)
mask = (marking_prob > 0.12) * 255
```

**效果对比（同一帧）**

```
v3 擦除区:   2.41%
4A 擦除区:   9.33%   ← 多抓 7%，中远景细线全清
```

### 5.3 只读缓存架构（`tools/gnd_session.py`）

```python
class GNDRenderSession:
    prepare_frame(frame_id, raw, drivable, vehicle, erase_fn)
        └─ clean_bg.setflags(write=False)   # ① 只读
    render_asset(frame_id, cfg, render_fn, blend_fn)
        ├─ canvas = data["clean"].copy()    # ② 深拷贝，无状态
        └─ np.where(v_mask, data["raw"], merged)  # ③ 车辆回贴
```

**三条铁律**（缺一不可）

1. 同一帧只跑一次 LaMa，`clean_bg` 设只读
2. 每次换素材从 `clean.copy()` 全新画布，禁止叠加
3. 最后用原始 `raw_img` 的 `vehicle_mask` 回贴

**磁盘缓存**：`/data/render/_gnd_clean_cache/*.npz`（跨进程复用）

**实测**

```
首次（含 LaMa）: 6.79s
后续切换素材:    0.49s / 0.59s / 0.50s    ← 13 倍提速
净面只读:       writeable = False   ✅
净面无残留:     max_diff  = 0       ✅
车辆回贴:       车体与raw差异 = 0    ✅
```

### 5.4 素材渲染（`tools/gnd_asset_render.py`）

13 类标线素材，`tex.png` / `tex_wear25.png` / `tex_wear55.png` 三档磨损。

```python
render_items(calib, items, shape)
  items = [{"asset": "lane_dashed_white", "x": 5, "y": -1.75,
            "len_m": 45, "width_m": 0.15}]
```

### 5.5 命令

```bash
/opt/ad_mining/venv/bin/python tools/gnd_render_one.py --config cfg.json
# cfg: {"batch_mode": true, "batch":..., "clip":..., "frames": 0,
#       "params": {...}, "out": "/data/render/gnd_clean_v3"}
```

`frames: 0` = 整个 clip（默认）。

---

## 6. 数据目录

### 6.1 输入数据

| 模块 | 路径 | 规模 |
|---|---|---|
| **TSR 场景** | `/data/test_data/real_clips` + `real_clips_extra` | 69 clip / 3280 帧 |
| **TSR 素材** | `/data/tsr_library_v2` | 86 国 / 33,716 张 |
| **OBJ 场景** | 同 TSR（带 LiDAR 标注） | — |
| **OBJ 素材** | `/data/obj_assets_real` | 11 类 / 216 张 |
| **GND 场景** | `/data/test_data/gnd_train` | 60 clip / 1200 帧 |
| **GND 素材** | `/data/gnd_assets` | 13 类 × 3 磨损档 |

### 6.2 输出数据

```
/data/render/
├── tsr_v3_10/           TSR 10 clip（40 张）
│   ├── <stem>_composed.jpg
│   └── annotations/<stem>.json
├── obj_v3_10/           OBJ 10 clip（200 帧）
│   ├── frames/<stem>_sim.jpg
│   └── annotations/
├── gnd_v3_10/           GND 10 clip（200 帧 + 200 净面）
│   ├── frames/<stem>_sim.jpg
│   ├── clean/<stem>_clean.jpg
│   └── annotations/
├── gnd_clean_v3/        GND 全量 4A（1200 帧，补齐中）
├── obj_params/          前端实时渲染（OBJ）
├── gnd_params/          前端实时渲染（GND）
└── _gnd_clean_cache/    净面磁盘缓存（.npz）
```

### 6.3 标注格式

```json
{
  "image": "1788047166.917_composed.jpg",
  "source_image": "1788047166.917.jpg",
  "source_path": "/data/test_data/real_clips_extra/.../1788047166.917.jpg",
  "width": 1920,
  "height": 1080,
  "objects": [{
    "class": "mandatory_turn_right_v2",   // 替换后的素材类别
    "detected_as": "mandatory",           // 原标牌的 YOLO 检测类别
    "category": "mandatory",
    "label": "指令 turn_right_v2",
    "bbox": [1067.0, 381.0, 1097.0, 411.0],
    "scene_shape": "circle",              // 原标牌形状
    "det_conf": 0.7827
  }]
}
```

**字段说明**

| 字段 | 含义 | 用途 |
|---|---|---|
| `class` | 替换后素材类别 | 右栏橙色 BBox 标签 |
| `detected_as` | 原标牌检测类别 | 左栏绿色 BBox 标签 |
| `scene_shape` | 原标牌形状（前景模板 IoU 判定） | 形状匹配依据 |

---

## 7. API 接口

### 7.1 数据列表

| 接口 | 说明 |
|---|---|
| `GET /api/real/clips?module=obj` | 列出 OBJ 批次 |
| `GET /api/real/frames?batch=&clip=&module=` | 列出帧 |
| `GET /api/gnd/clips` | 列出 GND 批次 |
| `GET /api/gnd/frames?batch=&clip=` | 列出 GND 帧（含 `sim`/`clean`/`seg`/`gt`） |
| `GET /api/tsr/gallery?dir=&limit=` | TSR 结果列表 |

### 7.2 图片

| 接口 | 说明 |
|---|---|
| `GET /api/real/image?batch=&clip=&stem=` | 原图 |
| `GET /api/gnd/image?batch=&clip=&name=` | GND 原图 |
| `GET /api/render_job_image?job=&name=&sub=` | 渲染产出图 |
| `GET /api/tsr/src?dir=&stem=` | TSR 原图 |
| `GET /api/tsr/result?dir=&file=` | TSR 合成图 |

**`render_job_image` 查找顺序**

```
1. /data/render/<job>/<sub>/<name>
2. /data/render/<job>/<name>
3. /data/render/<job>/frames/<name>
```

### 7.3 渲染

| 接口 | 说明 |
|---|---|
| `POST /api/real/render_params` | OBJ/TSR 渲染，body：`{module,batch,clip,params}` |
| `POST /api/gnd/render` | GND 渲染 |
| `GET /api/render/job?id=` | 查询任务状态 |

**任务状态**

```json
{"state": "done", "kind": "obj", "n": 32, "t0": ..., "t1": ..., "id": "..."}
```

### 7.4 素材

| 接口 | 说明 |
|---|---|
| `GET /api/obj/assets` | OBJ 素材列表 |
| `GET /api/obj/asset_thumb?path=` | OBJ 缩略图（PIL 实现，服务端无 cv2） |
| `GET /api/gnd/assets` | GND 素材列表 |
| `GET /api/gnd/asset_thumb?key=&wear=` | GND 缩略图 |

---

## 8. 运维命令

### 8.1 服务管理

```bash
# 重启
pkill -9 -f "server.py 8010"
cd /opt/ue5_synth/dashboard
setsid nohup env HF_ENDPOINT=https://hf-mirror.com \
  python3 server.py 8010 > /tmp/srv.log 2>&1 < /dev/null &

# 看日志
tail -50 /tmp/srv.log

# 健康检查
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8010/
```

### 8.2 前端修改检查

**每次改完 `index.html` 必须跑**：

```bash
/usr/bin/python3 /opt/ue5_synth/tools/check_blocks.py
```

输出 `块 1: XXX 字符  OK` 才算正常。**出现 `!! dp=-1 db=-2` 说明括号不平衡，整个 488K 脚本会静默失效。**

### 8.3 常见故障排查

| 现象 | 排查 |
|---|---|
| **页面白屏** | `check_blocks.py` 看括号；F12 Console 看报错 |
| **图片 404** | 检查 `source_path` 是否存在；看 `/api/render_job_image` 三级回退 |
| **仿真栏黑屏** | 检查 `/api/gnd/frames` 的 `sim` 字段是否为 `null` |
| **渲染 0 帧** | 看 `/api/render/job` 的 `n` 和 `err`；检查输出目录是否存在 |
| **素材索引失效** | 重跑 `build_asset_index.py` |

### 8.4 数据清理

```bash
# 清理临时缓存
rm -rf /data/render/_gnd_clean_cache/*

# 清理渲染任务中间产物
rm -rf /data/render/obj_params/*  /data/render/gnd_params/*
```

**⚠️ 删除前必须 `grep -r` 确认代码没引用该路径。** 历史上删 `final_obj3`、`gnd_fusion`、`gnd_model` 都导致过服务崩溃。

---

## 9. 关键设计决策

### 9.1 为什么 TSR 按帧而不是按 clip

**早期用轨迹回溯（Track-back）**：近景确证 → 向前放行远景小目标。

**实测问题**：YOLO 框在帧间抖动大，IoU 跟踪串不起来，反而误杀真标牌。

**最终方案**：按帧独立判断 + 单帧高置信直通。三轮车/行人/卡车全拦，真标牌召回 75~80%。

### 9.2 为什么形状判定不能用 BBox 长宽比

圆形、正方形、等边三角形的 BBox 都是 1:1，光看框分不出。

**必须提取前景轮廓**（红/蓝/黄像素，排除白背板），与标准模板算 IoU。

### 9.3 为什么 GND 要设净面只读

```python
clean_bg.setflags(write=False)
```

**踩过的坑**：直接在干净底图上反复贴图，第二次贴图会叠加到第一次结果上，越切越脏。

**修法**：每次 `clean.copy()` 全新画布 + `setflags(write=False)` 防止意外原地修改。

### 9.4 为什么 4A 用软概率而不是 argmax

中远景标线在模型输出里只有 1~3px 宽，`argmax` 会把它们判成 Road 丢掉。

**实测**：`argmax` 擦除 2.5%，软概率 `>0.12` 擦除 4.89%（多抓近一倍）。

### 9.5 为什么前端 BBox 有两个绘制函数

**历史遗留**：`drawBBox()`（旧，只画 `cpCv`）和 `draw3dBbox()`（新，画双栏）并存。

**问题**：`drawBBox` 会先把 `cpCv` 设成显示尺寸 414×233，然后因 `P.ann` 为空提前 return，把 `draw3dBbox` 画好的内容覆盖掉。

**修法**：`drawBBox` 直接转发给 `draw3dBbox`，不再自己设尺寸。

---

## 10. 已知问题

| # | 问题 | 状态 |
|---|---|---|
| 1 | TSR 5 个 clip 中 `highway` 场景标牌只出现 1 帧且 `conf=0.48`，被过滤门槛拦住 | 数据限制，非 bug |
| 2 | GND `gnd_clean_v3` 只有 36/60 clip 有 4A 产出 | **补齐中（1176/1200）** |
| 3 | 旧版 `gnd_clean_v2`（1200 帧）无 4A，擦除痕迹明显 | 补齐后应删除或降级为回退 |
| 4 | TSR `--strict-shape` 未启用，形状判不准时放行 | 设计如此（宁漏勿误 vs 宁误勿漏的取舍） |
| 5 | 服务端无 cv2，缩略图用 PIL 实现，功能受限 | 已知限制 |
| 6 | `draw3dBbox` 的 `3D-FILTER` 会丢弃 `b[0] > W` 的框 | 极远端目标可能不显示 |

---

## 11. 变更历史

### 2026-11-22

**TSR**
- ✅ 三层质检漏斗（硬规则 + 车辆互斥 + Mask2Former 交叉验证）
- ✅ 前景模板 IoU 判形状 + 颜色投票（替代 BBox 长宽比）
- ✅ 防露底门控（leak > 12% 拒绝）
- ✅ 素材索引 `asset_index.json`（33,716 张，46 秒）
- ✅ 形状 + 底色匹配采样（exact 97.5%）
- ✅ 红圈发粉问题修复（S 67→93，V 159→75）
- ✅ JPEG 色度保护 + 终态校准
- ✅ 按帧处理（取消轨迹依赖）

**GND**
- ✅ 4A 标线提取（Mask2Former 软概率 ∪ v3 Top-Hat）
- ✅ 只读缓存 + 无状态重绘（首次 6.8s / 后续 0.5s）
- ✅ 磁盘缓存跨进程复用
- ✅ 整 clip 渲染
- 🔄 补齐 24 个缺失 clip（进行中）

**OBJ**
- ✅ 整 clip 渲染（原来硬编码 `[:8]`）

**平台**
- ✅ 渲染 0 帧问题（输出目录在根目录而非 `frames/`）
- ✅ `obj_params` 后缀错配（`_sim.jpg` → `_obj.jpg`）
- ✅ 原图 BBox 不显示（`drawBBox` 覆盖 `draw3dBbox` 尺寸）
- ✅ GND 仿真黑屏（多目录回退 + `clean/` 子目录）
- ✅ 播放黑屏闪烁（预加载 + 双缓冲）
- ✅ 全屏缩放标签变大（字号反向补偿）
- ✅ 全屏 TSR 单栏 → 双栏
- ✅ 左右栏标签分离（`detected_as` vs `class`）
- ✅ 移除「对比/原图/仿真」冗余按钮

### 更早

- TSR v6 六步反射物理还原管线
- GND v3 标线提取（Top-Hat 替代 ratio_map）
- OBJ 3D BBox 缩放对齐
- 三模块批量跑（GND 60 clip / OBJ 65 clip / TSR 全场景）

---

## 附录 A：文件清单

### 核心算法

```
tools/
├── batch_replace.py           TSR 主入口（检测→过滤→采样→渲染→标注）
├── pipeline6.py               TSR 六步渲染管线
├── tsr_filter.py              三层质检漏斗
├── tsr_shape_cover.py         形状判定 + 防露底
├── tsr_trackback.py           轨迹回溯（--temporal 时使用）
├── build_asset_index.py       素材索引构建
├── render_obj_params.py       OBJ 参数入口
├── pipeline_sim.py            OBJ/GND 主流程
├── gnd_render_one.py          GND 参数入口
├── gnd_clean_pipeline.py      GND 四步管线
├── gnd_mask_4a.py             4A 标线提取
├── gnd_mask_v2.py             v3 标线提取（4A 的基础）
├── gnd_session.py             净面只读缓存
├── gnd_asset_render.py        GND 素材贴图
├── compose_core.py            融合算法（磨损/污渍/反光）
├── scene_optics.py            场景光学探针（雾度/清晰度）
└── check_blocks.py            前端括号平衡检查
```

### 配置文件

```
tools/asset_index.json         素材形状+颜色索引（4.7MB）
```

### 前端/后端

```
dashboard/
├── index.html                 前端单文件（488K）
└── server.py                  后端（190KB）
```

---

## 附录 B：常用参数速查

### TSR

```bash
--all-signs              替换所有检测到的标牌
--vary-class             按大类变化子类
--v6                     启用 6 步管线
--temporal               轨迹回溯（默认关）
--match-color            形状+底色匹配（默认开）
--conf 0.35              YOLO 置信度
--limit N                每 clip 最多处理 N 帧
```

### GND

```bash
"frames": 0              整个 clip（默认）
"frames": 20             只处理前 20 帧
"gnd_mode": "clean"      净面模式（默认）
"gnd_items": [...]       指定素材列表
```

### OBJ

```bash
"edit_mode": "add"       仅新增
"edit_mode": "replace"   替换原目标
"frames": 0              整个 clip
```

---

*文档生成时间：2026-11-22*
