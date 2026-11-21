# ue5_synth — 感知仿真数据平台

TSR / OBJ / GND 三模块实拍合成数据生产平台。

## 快速开始

```bash
cd dashboard && python3 server.py 8010
# 打开 http://<host>:8010/
```

## 架构

```
                    ┌─────────────────┐
                    │   任务中心       │  三模块统一派发
                    │  (模块任务面板)  │
                    └────────┬────────┘
                             ↓
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │   TSR   │         │   OBJ   │         │   GND   │
   │ 标牌替换 │         │ 三维目标 │         │ 地面标线 │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                    │                    │
   2D 牌面贴图         2.5D 精灵投影        2.5D 标线投影
   (V4.11)            (人工标定 H)         (人工标定 H)
        └────────────────────┼────────────────────┘
                             ↓
                    ┌─────────────────┐
                    │   仿真工作台     │  原图 | 合成 对比
                    │  + 点选放置      │  人工二次处理
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    9 项门禁      │  export_gate
                    │  passed / failed │  缺证据 = unchecked
                    └─────────────────┘
```

## 目录结构

```
ue5_synth/
├── dashboard/                  平台服务
│   ├── server.py               HTTP 服务 + 所有 API
│   ├── index.html              主页面（单页应用）
│   ├── gallery.html            素材画廊
│   ├── obj_new.html            OBJ 页（旧入口，已内嵌）
│   └── obj.html                GND 页（旧入口，已内嵌）
│
├── tools/                      数据生产工具
│   ├── ——— TSR ———
│   ├── batch_walk.py           批量遍历主流程
│   ├── render_selected.py      指定帧渲染
│   ├── prescreen_yolo.py       YOLO 预筛
│   ├── pipeline_v411.py        V4.11 渲染器入口
│   ├── pipeline6.py            6 步严格流水线
│   ├── batch_replace.py        素材匹配 + 形状判定
│   ├── replace_sign.py         检测与贴图
│   ├── auto_qc.py              像素级质检
│   ├── quality_contract.py     三态质量契约
│   ├── sign_*.py               真实感处理（遮挡/旋转/羽化）
│   ├── walk_cache.py           遍历缓存
│   ├── walk_index.py           增量索引
│   ├── tsr_*_api.py            TSR 接口
│   │
│   ├── ——— OBJ ———
│   ├── obj_sprite.py           2.5D 精灵投影（复用人工标定 H）
│   ├── est_camera.py           从底图估计相机
│   ├── cam_fit.py              从 H 拟合相机（备用）
│   ├── cam_from_calib.py       从 H 反解相机（备用）
│   │
│   ├── ——— GND ———
│   ├── gnd_project.py          2.5D 标线投影
│   ├── ground_masks.py         路面前景遮罩
│   ├── make_gnd_tex.py         磨损纹理生成
│   │
│   └── ——— 通用 ———
│       ├── pixel2world.py      像素↔世界坐标（含地平线守卫）
│       ├── export_gate.py      9 项导出门禁
│       └── check_blocks.py     HTML/JS 结构校验
│
├── engine/Content/Python/      UE 侧脚本
├── modules/                    素材库与标牌定义
├── common/                     通用工具
├── post/                       后处理
├── config/                     配置
├── docs/                       文档
└── linux/ windows/             平台脚本
```

## 三模块现状

| 模块 | 渲染方式 | 几何来源 | 耗时 | 状态 |
|---|---|---|---|---|
| **TSR** 标牌 | 2D 牌面贴图（V4.11） | 检测框 + 透视拟合 | 秒级 | ✅ 生产可用 |
| **OBJ** 目标 | 2.5D 精灵投影 | 人工四点标定 H | **1.5 秒** | ⚠️ 几何自洽，外观待改进 |
| **GND** 标线 | 2.5D 标线投影 | 人工四点标定 H | 秒级 | ⚠️ 几何自洽，拓扑待验证 |

### 关键设计：OBJ/GND 共用同一套几何

```
点击像素 (u,v)
  ↓ pixel2world（含地平线守卫）
世界坐标 (Y 横向, Z 纵向)
  ↓ 同一个 H 投影回来
渲染像素 —— 与点击位置偏差 < 1px
```

**不用 UE 实时相机**：UE 的 `fov + pitch` 参数化无法复现人工标定 H，
实测偏移达 290px。改用精灵投影后偏差 < 1px。

## 数据质量约定

### 9 项门禁

```python
REQUIRED_CHECKS = (
    "calibration",      # 标定可靠
    "original_labels",  # 原始标注完整
    "geometry",         # 几何一致
    "occlusion",        # 遮挡正确
    "appearance",       # 外观合理
    "multi_camera",     # 多相机一致
    "temporal",         # 时序一致
    "topology",         # 拓扑一致
    "split",            # 划分隔离
)
```

**规则：**
- 每项必须 `status == "passed"` **且** 带 `evidence_sha256`
- 缺证据自动加 `<name>_no_evidence` 阻塞
- 服务端生成 checks，浏览器不能绕过
- 任何一项不是 passed → `training_eligible = false`

### 当前实测

```
GND: passed 3/9  (calibration / geometry / occlusion)
OBJ: passed 1/9  (calibration)
→ 两者都 training_eligible = false
```

**这是正确行为**，不是 bug。缺的是真实标注、多相机、时序数据。

### 标注字段

```json
{
  "module": "GND",
  "gate": {"policy": "gate_v1_20261121", "passed": 3, "checked": 9},
  "checks": {"calibration": {"status": "passed", "evidence_sha256": "..."}},
  "training_eligible": false,
  "training_blockers": ["original_labels", "appearance", "..."]
}
```

## 人工二次处理

### 地面标定（必须先做）

1. 点右下角「地面标定」
2. 选底图
3. 在原图上依次点 4 个地面点（建议：左近、右近、右远、左远）
4. 保存 → `_calib_<bg>.json`

**标定质量直接影响 OBJ/GND 的位置精度。**

### 点选放置

```
点「点选放置」→ 画布激活
点地面位置 → 绿色圆点 + 编号
选类别 → 点「按点选位置生成」
```

- OBJ 类别：`cone / car / suv / truck / pedestrian`
- GND 类别：`lane_solid_white / lane_dashed_white / lane_solid_yellow / stop_line / road_edge_white`

## 关键约束

| 约束 | 原因 |
|---|---|
| **地平线附近不能点** | 射线平行地面，数学上无解（守卫 ±25px） |
| **距离 > 60m 不能点** | 投影误差放大 |
| **点选坐标按底图原始尺寸** | 显示图可能被 `&max=1600` 缩小 |
| **annotation 严格按 stem** | 找不到就 404，不回退首帧（防静默错配） |
| **ST 按模块分状态** | 防切换模块时串图 |
| **相机绝不能动** | 底图固定，相机一动物体位置全错 |

## 数据路径

| 内容 | 路径 |
|---|---|
| TSR 素材库 | `/data/tsr_library_v2` |
| TSR 产出 | `/data/render/tsr_batch` |
| OBJ/GND 产出 | `/data/render_hybrid` |
| 底图 | `/data/render_hybrid/bgs/cand_01..12.jpg` |
| 人工标定 | `/data/render_hybrid/bgs/_calib_<bg>.json` |
| 相机估计 | `/data/render_hybrid/bgs/_camera_est.json` |
| GND 图案 | `/data/render_hybrid/gnd_patterns` |
| 增量索引 | `/data/avsynth/state/walk_index.db` |
| OBJ 后处理 | `/data/avsynth/HybridPipeline/scripts` |
| UE 脚本 | `/data/avsynth/Content/Python` |

## 测试

```bash
cd tools
python test_export_gate.py        # 21 项门禁测试
python check_blocks.py            # HTML 结构校验
```

## 文档

见 [docs/](docs/)：

- `01_核心方案/` — BEV 方案、PRD、落地修订
- `02_操作文档/` — 操作 SOP、配置说明
- `03_开发计划/` — 现状与计划、检查清单
- `04_架构说明/` — 前端规范、环境报告
- `99_归档/` — 已被取代的旧版

## 明确未完成

| 项 | 阻塞原因 |
|---|---|
| 真实相机标定 | 需标定板采集（现在只有目视点选） |
| 实例/深度通道 | 需改 UE 导出 RenderTarget |
| 原始标签接入 | 需已审核数据集 |
| 多相机时序 | 需采集多相机数据 |
| 坡道/起伏路面 | 当前只支持平坦地面单应 |
| 真实集 P/R 验证 | 需训练实验 |

**以上六项不做，`training_eligible` 永远是 `false`。**
