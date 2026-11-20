# ue5_synth — 感知仿真数据平台

TSR / OBJ / GND 三模块实拍合成数据生产平台。

## 快速开始

```bash
# 服务
cd dashboard && python3 server.py 8010
# 打开 http://<host>:8010/
```

## 目录结构

```
ue5_synth/
├── dashboard/          平台服务与前端
│   ├── server.py        HTTP 服务（所有 API 路由）
│   ├── index.html       主平台页面（单页应用）
│   ├── gallery.html     素材画廊
│   ├── obj_new.html     OBJ 界面
│   └── obj.html         GND 界面
│
├── tools/              数据生产工具（24 个）
│   ├── batch_walk.py          TSR 主流程（检测→贴图→质检）
│   ├── render_selected.py     TSR 指定帧渲染
│   ├── prescreen_yolo.py      YOLO 预筛
│   ├── pipeline_v411.py       V4.11 渲染器入口
│   ├── pipeline6.py           6 步严格流水线
│   ├── batch_replace.py       素材匹配与形状判定
│   ├── replace_sign.py        检测与贴图基础
│   ├── auto_qc.py             像素级质检
│   ├── quality_contract.py    三态质量契约（passed/failed/unchecked）
│   ├── sign_*.py              真实感处理（遮挡/旋转/羽化）
│   ├── walk_cache.py          遍历缓存
│   ├── walk_index.py          增量索引
│   ├── make_gnd_tex.py        GND 标线磨损纹理生成
│   ├── est_camera.py          从底图估计相机参数
│   └── check_blocks.py        HTML/JS 结构校验
│
├── engine/Content/Python/    UE 侧脚本
├── modules/                  素材库与标牌定义
├── common/                   通用工具
├── post/                     后处理
├── config/                   配置
├── docs/                     文档
└── linux/ windows/           平台脚本
```

## 三模块状态

| 模块 | 渲染方式 | 素材 | 状态 |
|---|---|---|---|
| **TSR** 标牌 | 2D 牌面贴图（V4.11） | `/data/tsr_library_v2` 666 张 | ✅ 生产可用 |
| **OBJ** 目标 | UE 5.3 渲染 3D | Kenney Car Kit（CC0） | ⚠️ 结构可用，待标定 |
| **GND** 标线 | UE 材质 / 平面投影 | 程序生成磨损纹理（7 种） | ⚠️ 纹理就绪，待接入 |

## 数据质量约定

所有输出标注包含三态质量记录：

```json
{
  "quality": {
    "policy": "pixel_and_position_v1",
    "passed": true,
    "status": "passed",           // passed | failed | unchecked
    "semantic_status": "not_checked"
  },
  "training_eligible": false,      // 明确声明训练资格
  "training_blockers": [
    "no_real_label_merge",
    "no_multi_camera_temporal",
    "semantic_not_checked"
  ],
  "label_completeness": "detector_proposals_only"
}
```

**`training_eligible` 必须由完整验收流程设置，不能手工改 true。**

## 任务状态机

```
SUCCEEDED         全部 passed 且数量闭合   ← 唯一可交付
REVIEW_REQUIRED   无 failed，有 review     ← 需人工复核
PARTIAL           部分成功
FAILED            全部失败
```

## 关键约束

- **标注框必须对应实际渲染位置**：`bbox` 用渲染器返回的 `bb46`，`source_bbox` 保留原检测框
- **去重用 `source_bbox`**：`bbox` 会因渲染微调偏移导致同一目标重复标注
- **小目标转 unchecked 而非丢弃**：困难样本是 Recall 的来源
- **先编码再检查**：检查通过后直接写盘，不再二次压缩
- **跨 chunk 全局唯一**：`frame_seed = SEED + (START_INDEX + fi) * 7919`

## 数据路径

| 内容 | 路径 |
|---|---|
| TSR 素材库 | `/data/tsr_library_v2` |
| TSR 产出 | `/data/render/tsr_batch` |
| OBJ 产出 | `/data/render_hybrid` |
| 底图 | `/data/render_hybrid/bgs` |
| 增量索引 | `/data/avsynth/state/walk_index.db` |
| GND 真值 | `/data/render/scene_truth` |

## 文档

见 [docs/](docs/)：

- `BEV三模块实拍合成_质量与PR提升_代码实施方案_20261120.md` — 架构方案
- `BEV落地修订_补丁验收与质量样板_20261121.md` — 实施与验收
- `仿真平台_PRD与技术方案.md` — 产品需求
- `TSR仿真_操作SOP.md` — 操作规范

## 未完成项（明确列出）

| 项 | 阻塞原因 |
|---|---|
| 真实相机标定 | 需标定板采集 |
| 多相机时序合成 | 需多相机数据 |
| instance/depth pass | 需 UE 项目改造 |
| BEV raster | 需训练端协议 |
| 真实标签完整合并 | 需已审核数据集 |
| P/R 收益验证 | 需真实测试集训练 |
