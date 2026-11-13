# UE5 感知合成数据 — 双平台脚本包

> 仓库地址：https://github.com/zhouhuajian-cell/ad_data_sim

对应《UE5感知合成数据双平台实施方案.docx》的**可执行部分**（Windows 创作 + Ubuntu 批量渲染）。
本目录只做脚本与骨架，**引擎内脚本未在编辑器里跑过**（本机无 UE/无 N 卡）——`batch_capture.py` 必须按表 11 阶段三验收后才算可用。

数据链路（任务生成 → 后处理 → 数据集封装 → 操作台）已用合成帧在本地自检通过：

```bash
python3 tools/dryrun_check.py        # 19 项断言，约 10 秒，退出码 0 = 通过
```

## 目录

| 路径 | 放哪 | 作用 |
|---|---|---|
| `config/` | 本机或服务器 | 四维任务生成样例（scenes/lights/targets/cameras），字段契约见 `config/README.md` |
| `engine/Content/Python/capture_math.py` | UE 工程 `Content/Python/` | **纯数学**（相机基向量/投屏/3D 框投影/深度归一化/遮挡近似），不依赖 `unreal`，本地可测 |
| `engine/Content/Python/batch_capture.py` | UE 工程 `Content/Python/` | 引擎内批量捕获：读任务清单 → 生成/随机化 → 相机光照 → 四通道捕获 + 逐帧 JSON；跳过已完成、单帧异常不断批 |
| `windows/sync_project.ps1` | Windows | 工程单向下发 Ubuntu（rsync 优先，退化 tar-over-ssh；排除 Intermediate/Saved/Binaries/.vs） |
| `linux/env_check.sh` | Ubuntu | 环境基线核查（表 3 + 表 11 阶段一），只读不改 |
| `linux/compile_ue5.sh` | Ubuntu | UE5.3 源码编译（浅克隆 + Setup + Build UnrealEditor） |
| `linux/run_batch.sh` | Ubuntu | tmux 里起 `RenderOffscreen` 批量渲染（**绝不用 headless**，那是 NullRHI 不出图） |
| `linux/gpu_watch.sh` | Ubuntu | 1 秒级 GPU/显存/温度/功耗记录，>10.5G 告警 |
| `common/queue_gen.py` | 任意 | 四维笛卡尔展开 → `tasks.jsonl`；输入缺字段直接报错不写文件 |
| `common/case_check.py` | 任意 | 跨平台两校验：uproject↔Source 模块一致 + Content 引用大小写一致 |
| `post/postprocess.py` | Ubuntu（系统 Python） | 后处理七步（线性化曝光→噪声→暗角→色散→运动模糊→域对齐→8bit），参数逐帧落日志 |
| `post/pack_dataset.py` | Ubuntu（系统 Python） | 真值 JSON → COCO/YOLO 标注 + 训练验证切分 |
| `tools/mock_frames.py` | 任意（开发用） | 按真值契约造合成帧（8/16bit），不依赖 UE，用于自检与界面调试 |
| `tools/overlay_boxes.py` | 任意（开发用） | 把真值框画到成品图上，出 20 帧抽检拼图（表 11 真值质量项） |
| `tools/dryrun_check.py` | 任意（开发用） | 端到端自检：queue_gen → mock → postprocess → pack，含投影数学与 batch_capture 纯逻辑（假 unreal）断言 |
| `dashboard/server.py` + `index.html` | Ubuntu（系统 Python） | **可视化操作台**（零依赖单文件，见下节） |

## 使用顺序（对应表 11 四阶段）

0. **本地自检**：`python3 tools/dryrun_check.py` 全 PASS 再往下（改完后处理/队列/投影数学后必跑）。
1. **阶段一 环境**：`bash linux/env_check.sh` 全 PASS 再继续；`compile_ue5.sh` 编译（1~3 小时、内存峰值约 30G）。
2. **阶段二 工程**：Windows 建工程（表 4 插件）→ `case_check.py` 过 → `sync_project.ps1` 下发。
3. **阶段三 渲染**：`queue_gen.py` 出清单（用 `config/` 四维样例）→ `run_batch.sh` 起批 → `gpu_watch.sh` 盯着 → 人为中断验证断点续跑。
   **首次进编辑器前先过 [`docs/phase3_checklist.md`](docs/phase3_checklist.md) 的 T0 核对项**（16bit 落盘 / 语义模板 / 深度线性化 / 框对齐 / 续跑 / 容错），全过再放全量。
4. **阶段四 数据**：`postprocess.py` 七步 → `pack_dataset.py` 封装 → 二十帧人工抽检 + 域对齐目检。

### 数据链路契约

- 原始四通道：`<frame_id>_rgb.png` / `_depth.png` / `_sem.png`，真值 `<frame_id>.json`（附录 B 字段，含 `width`/`height`/`speed`）。
- 后处理成品：`<frame_id>.png`（与真值 JSON 同主文件名，方案 4.5 第 7 步），参数日志 `params_log.jsonl`。
- **16bit RGB 必须用 OpenCV 解码**：实测 Pillow（含 12.x）会把 16bit RGB PNG 静默截断成 8bit，因此缺少
  OpenCV 时脚本**直接报错并给出安装命令**，不做静默降级。Ubuntu：`sudo apt install python3-opencv`；
  Windows：`pip install opencv-python`。16bit 灰度深度图 Pillow 本身支持，不需要 OpenCV。
- 工况映射：真值 `lighting.condition` 命中 `front_light`/`back_light`/`dusk`/`night`/`rain` 时按方案 4.5 的取值区间
  自动套曝光与读出噪声；`params.json` 的显式键优先；运动模糊核长由 `speed_mps × 0.45` 换算并裁到 0~13px
  （系数需按实采图边缘标定后固化）。

## 可视化操作台（dashboard）

浏览器里完成全部操作，不用敲命令：

```bash
python3 dashboard/server.py          # 默认端口 8010（或 UE5_PANEL_PORT / 首参数改）
# 浏览器打开 http://<服务器IP>:8010
```

- **零依赖**：只用 Python 标准库（系统 python3.8 可跑），不碰 ad_mining 的 `/venv`。
- 功能：总览（GPU 型号/显存/内存/swap/磁盘/渲染进度条）、环境核查、任务生成、启动/停止渲染、GPU 监控、
  工程大小写校验、编译 UE5、后处理+打包、**链路自检（dryrun_check.py）**、日志尾巴、作业控制台（可杀）、配置在线改（存 `config.json`）。
- 各动作都是后台作业：输出进 `dashboard/jobs/*.log`，界面轮询展示；编译/渲染类长任务关页面也继续跑。
- 各动作的 Python 用服务端自身解释器（`sys.executable`），不依赖 PATH 里的 `python3`；作业返回值非 0 会标红。
- 本地 Windows 也能起，用于界面调试（GPU/内存探针自动降级；`bash`/`tmux`/`nvidia-smi` 类动作不可用属正常）。
- **安全边界**：只暴露白名单动作（`build_cmd`），参数走 argv 不过 shell，参数做类型校验；端口绑 0.0.0.0（内网访问）。
- ⚠️ 脚本路径默认全按服务器（`/data/...`、`/opt/ue5_synth`）——**要在服务器上跑才有意义**；
  本地调试时把配置里的 `scripts_dir`/`project_dir` 指向本仓库即可跑通「链路自检」与「工程校验」类动作。

## ⚠️ 用生产服务器 10.2.248.34 当 Ubuntu 侧的硬约束

- **编译期内存峰值 26~30G，服务器总内存 31G** → 编译必须在**训练/抽帧/判定全停**的窗口做，且随时准备中断；OOM 已发生 4 次。
- **源码+编译产物约 120G**，跑前先 `df -h` 确认；数据盘另需 1TB（单帧 16~22MB）。
- 渲染期 8~14G 内存 + 独占 GPU → **与 VLM 判定/向量化互斥**（12G 卡装不下两个），开跑前停掉在跑任务。
- 与 ad_mining 的资源竞争清单见 AGENTS.md「训练优先」一条；**本脚本包不自动开机自启、不写 cron**，全部人工触发。

## 已知未完成

- `batch_capture.py` 的四通道捕获按文档 3.4 实现，**UE Python API 未在编辑器内验证**（16bit 渲染目标落盘、
  CustomDepth 语义模板、关卡内 Actor 清场/生成这三处最需要现场调）；文件内用 `TODO(阶段三)` 逐条标出。
  已做的本地保障：投影数学与 manifest 契约由 `tools/dryrun_check.py` 断言；渲染目标按通道复用（避免每帧新建
  撑爆显存）；`image_files.bit_depth` 按实际 PNG 头记录（16bit 链路没打通时如实写 8bit，不假装）；
  主循环的跳过/失败续跑/退出码用假 `unreal` 模块回归。**这些都不等于引擎内可用。**
- 反光膜材质、相机标定属编辑器内手工活（文档 3.2/3.3），无脚本替代。
- 16bit RGB 依赖 OpenCV；服务器装不上时只能退 8bit，动态范围损失需在方案里显式记录（脚本会报错提醒，不会静默截断）。
- 语义/深度两条通道依赖引擎内材质与项目设置（`r.CustomDepth=3`、深度线性化材质），属阶段三现场工作。
