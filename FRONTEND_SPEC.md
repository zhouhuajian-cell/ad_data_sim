# 前端改造需求书 + 后端接口契约

> 编写时间：2026-11-17
> 读者：负责前端重构的开发（含 AI 助手）
> 目标：把操作台从「5 步向导」改成「仿真工作台」式布局（预览视口 + 参数面板 + 实时监控）

---

## 一、必须先读的三条硬约束

### 约束 1：禁止使用任何 CDN

参考设计稿 `fangz.html` 用了这些外部依赖：

```html
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.44.0/tabler-icons.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.min.js"></script>
```

**目标环境是内网，访问不了这些 CDN。** 直接用会导致页面变成无样式的纯文本。

**必须改成：**
- CSS：全部内联手写（或内联 `<style>`，不要外部文件）
- 图标：内联 SVG（`<svg>` 直接写在 HTML 里）
- 图表：手写 Canvas 绘制（约 30 行代码即可画折线图）

**验收**：在**断网环境**下打开页面，样式与图表完整。

### 约束 2：不能用假数据冒充真实状态

参考稿里这些是硬编码的假数据：

```html
<span>UE引擎：已连接</span>
<span>内存：16.2 / 32 GB</span>
<span>显存：5.8 / 12 GB (4070Ti)</span>
<div class="text-center">UE5 预览流画面区域（对接 SceneCapture）</div>
<span>TSR:6 | 车道线:11 | 目标:4 | HardCase:2</span>
```

**必须改成从后端真实获取。** 拿不到时显示「未知 / 未连接」，**不要显示看似正常的假数字**。

**原因**：这是生产工具，假数据会让操作员做出错误判断（例如以为引擎在用，实际早就崩了）。

### 约束 3：单文件，零构建

后端 `server.py` 是**零依赖 Python 标准库 HTTP 服务**，启动时会把 `index.html` 读进内存。

**所以前端必须是单个 `index.html`**，不能有 webpack/vite 构建步骤，不能拆成多个文件（除非改后端支持静态目录）。

---

## 二、页面结构

```
┌─────────────────────────────────────────────────────────────────┐
│ 顶栏：引擎状态 | 显存 | 内存 | 磁盘 | 当前任务                      │
├────────┬────────────────────────────────────┬───────────────────┤
│ 左侧    │  中央视口区                          │  右侧参数面板       │
│ 导航    │  ┌──────────────────────────────┐  │  ┌─────────────┐  │
│        │  │  预览画面（渲染结果/抽检图）    │  │  │ 光照设置     │  │
│ 工作台  │  │  叠加：BBox / 类别 / 遮挡率     │  │  ├─────────────┤  │
│ 场景    │  └──────────────────────────────┘  │  │ 天气         │  │
│ 队列    │  工具条：单帧渲染 / 批量生成        │  │  ├─────────────┤  │
│ 数据集  │                                    │  │ 批量参数      │  │
│ 设置    │  底部：实时日志 + 进度条             │  │  ├─────────────┤  │
│        │                                    │  │ 资源曲线      │  │
└────────┴────────────────────────────────────┴───────────────────┘
```

---

## 三、后端接口契约

### 3.1 已有接口（可直接用）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 全量状态（见下方字段表） |
| GET | `/api/job?id=xxx` | 单个作业详情 |
| GET | `/api/log?job=xxx` | 作业日志（支持 `offset` 增量拉取） |
| GET | `/api/overlay` | 抽检图（PNG 二进制） |
| POST | `/api/run` | 发起动作，body: `{"action": "...", "params": {...}}` |
| POST | `/api/kill` | 终止作业，body: `{"id": "..."}` |
| GET/POST | `/api/config` | 读写配置 |

**`/api/status` 返回的关键字段：**

```json
{
  "cfg": { "queue_file": "...", "raw_dir": "...", "dataset_dir": "...", "uproject": "...", "ue_dir": "..." },
  "gpu": { "name": "NVIDIA GeForce RTX 4070 Ti", "mem_mb": 73, "temp": 39 },
  "mem_avail_g": 29.1,
  "disk_data_free_g": 1100.5,
  "disk_root_free_g": 1200.0,
  "uproject_exists": true,
  "capture_script_exists": true,
  "ue_ready": true,
  "tasks_total": 360,
  "tasks_done": 0,
  "dataset_ready": false,
  "overlay_ready": false,
  "render_running": false,
  "auto": { "enabled": true, "gpu_watch": true, "phase": "idle", "last_log": "..." }
}
```

### 3.2 现有动作（POST /api/run 的 action 值）

| action | params | 作用 |
|---|---|---|
| `queue_gen` | `{}` | 按 config 生成任务清单 |
| `render_start` | `{"queue": "路径"}` | 启动批量渲染（tmux 会话） |
| `render_stop` | `{}` | 终止渲染 |
| `post` | `{"limit": "0"}` | 后处理（RT → 成品图） |
| `voc` | `{}` | 导出 PASCAL VOC XML |
| `hardcase` | `{}` | 导出难例清单 CSV |
| `pack` | `{"format": "coco"}` | 打包训练集 |
| `overlay` | `{"limit": "20"}` | 生成真值抽检拼图 |
| `dryrun` | `{}` | 链路自检 |
| `gpu_watch` | `{}` | 显卡监控记录 |

**统一返回**：`{"id": "作业ID"}`，之后用 `/api/job?id=` 轮询状态（`status`: `running` / `ok` / `fail`）。

### 3.3 需要新增的接口 ⚠️

以下接口**后端还没有**，前端可以先做占位（显示「功能开发中」），或推动后端补齐。

---

#### 【新增 1】单帧渲染预览

```
POST /api/preview
body: {
  "map": "/Game/Maps/TestLevel",
  "light": "front_light",        // front_light | back_light | night | dusk | rain
  "weather": "clear",
  "camera": { "location": [-800,0,200], "rotation": [0,0,0], "hfov_deg": 90 },
  "resolution": [640, 360]
}
返回: { "id": "作业ID" }
完成后: GET /api/preview/image?id=作业ID  → PNG 二进制
```

**后端实现要点**（供后端开发参考）：
- 用 `xvfb-run` + `UnrealEditor-Cmd` + `-Vulkan` 跑单帧
- 单帧约 2 秒（640×360），1080P 约 4~8 秒
- **注意**：必须用 xvfb，`-RenderOffscreen` 模式下 SceneCapture 出黑图

---

#### 【新增 2】参数下发

```
POST /api/scene_params
body: {
  "light_intensity": 0.72,       // 车灯强度
  "backlight_intensity": 0.25,   // 逆光强度
  "weather": "rain",
  "wet_reflection": true,        // 积水反光
  "road_patch": false            // 路面补丁干扰
}
```

**后端实现要点**：写入 `config/lights.json` 或生成临时任务行，下一帧渲染时生效。

---

#### 【新增 3】按 UI 参数批量生成

```
POST /api/batch_gen
body: {
  "total": 200,                  // 总样本数
  "hardcase_ratio": 0.3,         // 难例占比 0~1
  "distance_range": [30, 180],   // 距离区间（米）
  "outputs": { "png": true, "voc": true, "json": true, "csv": true },
  "out": "/data/queue/batch_xxx.jsonl"
}
返回: { "id": "作业ID", "queue_file": "生成的清单路径" }
```

**后端实现要点**：调 `common/hardcase_gen.py`，已有能力：
```bash
python3 common/hardcase_gen.py --kind all --count N -o out.jsonl
# 已有 8 类难例：far_small / occlusion / backlight / night / rain / wear / tiny / edge
```

---

#### 【新增 4】资源历史曲线

```
GET /api/metrics?minutes=10
返回: {
  "gpu_mem": [[ts, 5.8], [ts, 5.9], ...],   // 显存 GB
  "mem_used": [[ts, 16.2], ...],            // 内存 GB
  "gpu_util": [[ts, 87], ...]               // GPU 利用率 %
}
```

**后端实现要点**：`/data/logs/gpu_watch.csv` 已有历史记录（后台线程每 15 秒写一行），只需读出最近 N 分钟返回。

---

#### 【新增 5】数据集统计

```
GET /api/dataset_stats
返回: {
  "total_frames": 360,
  "by_class": { "traffic_sign": 120, "vehicle": 80, "landline": 40, ... },
  "hardcase_count": 45,
  "hardcase_by_rule": { "occlusion": 20, "far_small": 15, "light": 10 },
  "datasets": [ { "name": "coco_20261117", "frames": 360, "path": "..." } ]
}
```

**后端实现要点**：读 `hardcase_list.csv` 和 `annotations/instances_all.json` 汇总。

---

## 四、各控件与接口的对应关系

| 界面控件 | 调用的接口 | 后端现状 |
|---|---|---|
| 顶栏「UE引擎」状态 | `GET /api/status` 的 `ue_ready` | ✅ 有 |
| 顶栏「显存/内存/磁盘」 | 同上 `gpu` / `mem_avail_g` / `disk_data_free_g` | ✅ 有 |
| 中央预览画面 | `POST /api/preview` → `GET /api/preview/image` | ❌ **需新增** |
| 「单帧渲染」按钮 | `POST /api/preview` | ❌ **需新增** |
| 「批量生成难例」按钮 | `POST /api/batch_gen` | ❌ **需新增** |
| 光照强度滑块 | `POST /api/scene_params` | ❌ **需新增** |
| 天气选择 | 同上 | ❌ **需新增** |
| 路面属性勾选 | 同上 | ❌ **需新增** |
| 总样本数 / 难例占比 / 距离区间 | `POST /api/batch_gen` | ❌ **需新增** |
| 输出格式勾选（PNG/VOC/JSON/CSV） | `POST /api/batch_gen` 的 `outputs` | ❌ **需新增** |
| 资源监控曲线 | `GET /api/metrics` | ❌ **需新增**（数据源已有） |
| 底部日志 | `GET /api/log?job=` | ✅ 有 |
| 进度条 | `GET /api/status` 的 `tasks_done / tasks_total` | ✅ 有 |
| 场景模板页 | `GET /api/status` 的 `cfg` | ⚠️ 部分 |
| 数据集&难例页 | `GET /api/dataset_stats` | ❌ **需新增** |

**统计：现有接口覆盖 6 项，需新增 5 个接口覆盖 11 项交互。**

---

## 五、页面模块详细要求

### 5.1 顶栏

```
[图标] PerceptionSim 感知仿真        [引擎: 已就绪 | 未就绪]  [显存 5.8/12G]  [内存 16.2/32G]  [磁盘 1100G]
```

- 状态色：正常=青色、警告=橙色、异常=红色
- **超过阈值变色**：显存 >10.5G 红色、温度 >78°C 橙色、>85°C 红色、磁盘 <150G 橙色、<50G 红色
- 数据来源：`GET /api/status`，轮询间隔 5 秒

### 5.2 左侧导航

5 项，点击切换右侧主区域内容（**不做页面跳转，用 JS 切换显示**）：

| 导航 | 内容 |
|---|---|
| 仿真工作台 | 预览视口 + 参数面板（默认页） |
| 场景模板 | 关卡列表（从 `/api/status` 的 `cfg` 和实际关卡文件读） |
| 渲染任务队列 | 任务清单表格（帧号/状态/耗时） |
| 数据集&难例 | 统计卡片 + 难例分布图 + 数据集列表 |
| 系统配置 | `GET/POST /api/config` 的表单 |

### 5.3 中央视口

**正常态**：显示 `GET /api/preview/image` 返回的 PNG，`<img>` 标签铺满，`object-fit: contain`。

**叠加层**（用绝对定位的 `<canvas>` 或 `<svg>` 画 BBox）：
- BBox 数据来自真值 JSON 的 `objects[].bbox`
- 每个框上标注 `class_name` 和 `occlusion_ratio`

**空态**（没有渲染结果时）：
```html
<div class="empty-state">
  <p>尚未渲染</p>
  <p class="hint">点击「单帧渲染」生成预览</p>
</div>
```
**不要在空态显示任何假的画面或数字。**

**工具条开关**：
- 叠加 BBox：控制是否画框
- 显示遮挡率：控制框上标签
- 显示车道线掩码：控制是否叠加语义图

### 5.4 右侧参数面板

用 Tab 切换 5 个分组：`全局场景 / TSR标识 / 车道线 / 交通目标 / 批量难例`

**全局场景**：
- 光照：白天 / 夜间 / 逆光（三选一按钮组）
- 车灯强度：滑块 0~1（显示当前值）
- 逆光强度：滑块 0~1
- 天气：晴天 / 小雨 / 雾天（按钮组）
- 路面：路面补丁勾选、积水反光勾选

**批量难例**：
- 总样本数（数字输入）
- 难例占比（数字输入，0~100，显示为百分比）
- 距离区间：两个数字输入（米）
- 输出格式：4 个复选框（PNG / VOC XML / JSON真值 / hardcase.csv）

**每项改动后**：调对应接口，**成功才更新 UI 显示值**；失败时回滚并提示。

### 5.5 底部日志条

- 显示最近一条日志（`GET /api/log?job=` 的最后一行）
- 右侧进度条：`tasks_done / tasks_total`
- 点「展开」显示完整日志滚动区

### 5.6 资源监控曲线

手写 Canvas 折线图（**不要用 Chart.js**）：

```javascript
function drawLineChart(canvas, series, colors, labels) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  // 画网格
  // 画两条折线（显存 / 内存）
  // 画图例
}
```

数据来自 `GET /api/metrics?minutes=10`，每 15 秒刷新。

---

## 六、技术约束

| 项 | 要求 |
|---|---|
| 文件 | 单个 `index.html`，≤ 100 KB |
| 依赖 | 零外部依赖（无 CDN、无 npm 包） |
| 浏览器 | 支持 Chrome / Edge 最新版 |
| 编码 | UTF-8 无 BOM |
| 轮询 | `/api/status` 每 5 秒，`/api/metrics` 每 15 秒 |
| 错误处理 | 接口失败时显示错误提示，**不清空已有数据** |

---

## 七、验收清单

- [ ] 断网环境下打开，样式与图表完整（无 CDN 依赖）
- [ ] `GET /api/status` 失败时，顶栏显示「未连接」而非假数字
- [ ] 未渲染时视口显示空态提示，不显示假画面
- [ ] 5 个导航项能正常切换
- [ ] 参数面板改动后能调用对应接口（未实现的接口显示「功能开发中」）
- [ ] 资源曲线能画出真实数据（若 `/api/metrics` 未实现则显示「数据不可用」）
- [ ] 页面在 1920×1080 和 1366×768 下都不出现滚动条错位
- [ ] 控制台无 JS 报错

---

## 八、参考设计稿

`C:\Users\zhouhuajian\Desktop\fangz.html`（275 行）

**可以照抄的**：整体布局比例、配色（深色底 `#111827` / 面板 `#1E293B` / 青色强调 `#36CFC9`）、控件样子

**必须改掉的**：
1. 三个 CDN 引用 → 全部内联
2. 所有硬编码假数据 → 改成接口调用
3. Chart.js → 手写 Canvas
4. Tabler 图标字体 → 内联 SVG

---

## 九、后端对接注意事项

给做后端的人：

1. **`server.py` 启动时把 `index.html` 读进内存**，改完 HTML 必须重启服务才生效：
   ```bash
   kill $(ss -ltnp | grep :8010 | sed 's/.*pid=\([0-9]*\).*/\1/')
   cd /opt/ue5_synth/dashboard && setsid nohup python3 server.py 8010 >> panel.log 2>&1 &
   ```

2. **单帧预览必须用 xvfb**，直接 `-RenderOffscreen` 会出黑图：
   ```python
   cmd = ["xvfb-run", "-a", "--server-args=-screen 0 1280x720x24",
          UE_CMD, UPROJECT, "-ExecutePythonScript=...", "-Vulkan"]
   ```

3. **现有渲染是批量模式**（跑 tmux 会话），单帧预览需要另写一条轻量路径。

4. **不要让预览接口阻塞**：单帧渲染约 2 秒，要异步返回 job id，前端轮询。
