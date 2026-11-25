# 从零开始：三件事做完就能出图

> 现状（2026-11-16 核实）：服务器 `10.2.248.34` 环境全 PASS（驱动 550.135 / Vulkan 1.3 / 126G 空闲 / swap 关）；
> 工程骨架已建好并下发到 `/data/avsynth`（`case_check` ALL PASS）；**还差引擎源码编译**。
> 本机 Windows 既没有 UE 编辑器，也没有 Visual Studio，所以工程也能在服务器上生成，不依赖 Windows。

## 为什么现在点「开始出图」还报错

操作台第 3 步现在提示的最后一项是 **引擎没编译好**。源码编译要先从 Epic 的私有仓库拉代码，
而这需要你的 Epic 账号（GitHub 需要个人访问令牌；SSH 密钥方式对已授权的私有仓库也通常不可用）。
这一步必须由你本人完成，下面是逐条照做即可的命令。

---

## 第 1 步：让服务器能从 Epic 拉源码（一次性，约 5 分钟）

1. 用浏览器登录 Epic，打开 <https://www.unrealengine.com/zh-CN>，进个人 **账户** → **应用与账户**（Apps and Accounts）。
2. 找到 **GitHub**，点 **关联账户**，按提示完成 GitHub 授权。
3. 回到 GitHub，访问 <https://github.com/EpicGames/UnrealEngine>，**能看到仓库**（不再 404）说明关联成功。
4. 创建一个 HTTPS 方式使用的 **个人访问令牌**（GitHub → Settings → Developer settings → Personal access tokens，
   权限勾 `repo`，有效期按公司规范选），令牌只在本机临时使用，**不要贴到任何文档、聊天或脚本里**。

## 第 2 步：在服务器上编译引擎（在服务器终端执行）

```bash
# 把令牌临时放进当前 shell 的环境变量，供这一次克隆使用（不要写进文件、不要放 git 配置）
read -rs UE_GH_TOKEN          # 粘贴令牌后回车，屏幕不显示
echo "$UE_GH_TOKEN" | git clone --depth 1 --branch 5.3 \
  "https://<你的 GitHub 用户名>:${UE_GH_TOKEN}@github.com/EpicGames/UnrealEngine.git" /data/UE_5.3
unset UE_GH_TOKEN

# 依赖 + clang-16（20.04 官方源没有，脚本会用 LLVM 源补），然后编译。约 1~3 小时、内存峰值 26~30G
bash /opt/ue5_synth/linux/compile_ue5.sh /data/UE_5.3
```

- 编译前应用 `env_check.sh` 全 PASS，且**训练/抽帧/VLM 判定全停**（OOM 已发生过 4 次）。
- 编译完成标志：最后一行 `OK: 编译完成`，且存在 `/data/UE_5.3/Engine/Binaries/Linux/UnrealEditor-Cmd`。
- 也可以在操作台「高级操作 → 编译引擎」点按钮跑（前提是源码已在 `/data/UE_5.3`）。

## 第 3 步：编译工程模块（首次出图会自动做，也可手动）

```bash
/data/UE_5.3/Engine/Build/BatchFiles/Linux/Build.sh avsynthEditor Linux Development \
  -Project=/data/avsynth/avsynth.uproject -Progress
```

只编译本工程的 `Source/avsynth`（几十秒到几分钟）。`run_batch.sh` 发现没有产物会自动补这一步。

## 第 4 步：回操作台出图

打开 <http://10.2.248.34:8010>，第 3 步点「开始出图 / 继续出图」。
400 预检提示会消失，后台 tmux 会话 `ue5_render` 开始渲染；进度按 `/data/queue/done` 计数，断点续跑。

**但先别放全量**：按 [阶段三检查表](../03_开发计划/phase3_checklist.md) 的 T0-1~T0-10 核对这几项，
它们必须在真实工程里验证，过了再跑 360 帧：

| 项 | 一句话 |
|---|---|
| T0-1 16bit 落盘 | RGB 必须是 16bit 线性 PNG，不然高光动态范围永久丢失 |
| T0-2 线性 | 引擎不能偷偷做 sRGB/色调映射 |
| T0-3 语义模板 | 唯一值只能是 {1..6, 255}；天空 255 |
| T0-4 深度 | 10m 目标的深度值应约 3277（0~200m 归一化） |
| T0-5 框对齐 | 用 `tools/overlay_boxes.py` 出 20 帧拼图目检，偏差 ≤2px |

## 我在本地能替你做完的（已做完）

- 工程骨架（uproject / C++ 模块 / Config）与下发脚本 `windows/sync_avsynth.ps1`
- 下发脚本的两个真实缺陷已修：Windows tar 的 `--exclude` 顺序、`tar | ssh` 管道截断（22MB 只过去 655 字节）→ 改成 tar→scp→远端解包
- `compile_ue5.sh`：补依赖、20.04 的 clang-16、`-vscode` 生成工程
- `run_batch.sh`：缺工程/脚本/引擎/清单的逐项预检，以及首次自动编译工程模块
- 操作台预检：缺什么直接在页面上说，不再产生红色失败作业

## 我做不到的部分（需要你在现场）

1. Epic/GitHub 账号授权与令牌（安全边界，必须本人）
2. 模型资产、材质、关卡摆放、语义/深度材质（编辑器内手工活，方案 3.2/3.3）
3. 100 帧实机标定单帧耗时（产能估算，方案 6.1）
