#!/usr/bin/env bash
# run_batch.sh — 批量渲染启动器（方案 4.3/4.4）
# 干什么：tmux 会话里以 RenderOffscreen 跑引擎批量捕获；断线/重启后重跑即断点续跑（done 标记跳过）。
# 怎么跑：bash run_batch.sh <xxx.uproject> [任务清单] [UE 引擎根目录]
# 成功判据：tmux 里日志滚动、/data/queue/done 逐条增加；kill 后重跑已完成项被跳过。
# ⚠️ 单卡单进程并发=1：产能靠队列长度堆，不靠并发（同卡两个渲染进程只互相挤显存）。
# ⚠️ 绝不用 -headless：那是 NullRHI，根本不初始化图形后端，出不了图（附录 C：出图全黑的根因）。
# 目录默认按附录 A，可用环境变量覆盖：AD_UE_RAW / AD_UE_META / AD_UE_DONE / AD_UE_LOGS
# 路径约定不带空格（与方案目录一致）；工程要从 Windows 下发，本脚本不会自己找工程。
set -euo pipefail

UPROJECT="${1:?用法: run_batch.sh <xxx.uproject> [tasks.jsonl] [UE 引擎根目录]}"
QUEUE="${2:-/data/queue/tasks.jsonl}"
UE_ROOT="${3:-/data/UE_5.3}"
UE_CMD="$UE_ROOT/Engine/Binaries/Linux/UnrealEditor-Cmd"
LOG_DIR="${AD_UE_LOGS:-/data/logs}"
RAW="${AD_UE_RAW:-/data/render/raw}"
META="${AD_UE_META:-/data/render/meta}"
DONE="${AD_UE_DONE:-/data/queue/done}"
ERRLOG="$LOG_DIR/capture_errors.log"
SESS="ue5_render"

die() {  # die <一句话原因> <一行怎么办>...
  echo "ERROR: $1" >&2
  shift
  for line in "$@"; do echo "       $line" >&2; done
  exit 1
}

# ── 启动前把"缺什么、怎么办"一次说清楚：无人值守时这比跑到一半才报错有用 ──────────
# 注意：PROJECT_DIR 必须在工程存在之后再算，否则 `cd 不存在的目录` 会先被 set -e 干掉
[ -f "$UPROJECT" ] || die "工程文件不存在：$UPROJECT" \
  "本仓库只有脚本、没有 UE 工程。先在 Windows 上把工程下发到服务器：" \
  "  powershell -File windows/sync_project.ps1 -ProjectDir <Windows 工程目录> -Remote <user@服务器> -RemoteDir $(dirname "$UPROJECT")" \
  "工程已放在别的目录？去操作台「高级操作 → 配置」把 uproject / project_dir 改成实际路径。"

PROJECT_DIR="$(cd "$(dirname "$UPROJECT")" && pwd)"
CAPTURE="$PROJECT_DIR/Content/Python/batch_capture.py"

[ -f "$CAPTURE" ] || die "工程里没有批量捕获脚本：$CAPTURE" \
  "把仓库 engine/Content/Python/ 下的 batch_capture.py 和 capture_math.py 复制到" \
  "<Windows 工程>/Content/Python/，再重新下发工程。"

[ -x "$UE_CMD" ] || die "引擎还没编译好（或路径不对）：$UE_CMD" \
  "先在操作台「高级操作 → 编译引擎」跑 compile_ue5.sh（1~3 小时、内存吃满，选空闲窗口）。" \
  "引擎装在别处时，改操作台配置里的 ue_dir，或给本脚本第 3 个参数。"

# 工程模块没编译过时 UnrealEditor-Cmd 会以「缺模块」退出；先增量编译一次工程（只编本工程）
if ! ls "$PROJECT_DIR/Binaries/Linux/"UnrealEditor-avsynth* >/dev/null 2>&1; then
  echo "== 工程模块尚未编译，先增量编译（首次几分钟，不动引擎）"
  "$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh" avsynthEditor Linux Development \
    -Project="$UPROJECT" -Progress 2>&1 | tail -20 || true
  ls "$PROJECT_DIR/Binaries/Linux/"UnrealEditor-avsynth* >/dev/null 2>&1 || die \
    "工程模块编译失败（见上面日志）" \
    "把上面最后 30 行发给人看：常见是 clang 版本不对或 Source 里的 C++ 文件缺失。"
fi

[ -f "$QUEUE" ] || die "任务清单不存在：$QUEUE" \
  "先在操作台第 2 步「生成拍摄清单」（或跑 common/queue_gen.py）。"

mkdir -p "$LOG_DIR" "$RAW" "$META" "$DONE"

if tmux has-session -t "$SESS" 2>/dev/null; then
  echo "会话 $SESS 已在跑，不重复启动（附日志: $LOG_DIR/batch.log）"; exit 0
fi

# RenderOffscreen：保留 Vulkan 设备但不创建窗口；-Vulkan 显式指定后端
# 无 X11 会话时 Vulkan 设备可能创建失败 → 注释里备了 xvfb-run 兜底（环境兜底，非性能优化）
CMD="$UE_CMD $UPROJECT -ExecutePythonScript=$CAPTURE \
 -RenderOffscreen -unattended -nosplash -noSound -NoLiveView -Vulkan \
 -log=$LOG_DIR/batch.log"

# 把目录也传进引擎脚本（batch_capture.py 认这些环境变量），避免两边路径约定不一致
ENV_PREFIX="AD_UE_QUEUE=$QUEUE AD_UE_RAW=$RAW AD_UE_META=$META AD_UE_DONE=$DONE AD_UE_ERRLOG=$ERRLOG"
tmux new-session -d -s "$SESS" "$ENV_PREFIX bash -c '$CMD; echo EXIT=\$? >> $LOG_DIR/batch.log; sleep 300'"
echo "已启动 tmux 会话 $SESS；查看: tmux attach -t $SESS  或  tail -f $LOG_DIR/batch.log"
echo "若 Vulkan 设备创建失败: tmux kill-session -t $SESS 后改用 xvfb-run -a --server-args='-screen 0 1920x1080x24' 包裹 $UE_CMD 再跑"
