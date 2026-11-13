#!/usr/bin/env bash
# run_batch.sh — 批量渲染启动器（方案 4.3/4.4）
# 干什么：tmux 会话里以 RenderOffscreen 跑引擎批量捕获；断线/重启后重跑即断点续跑（done 标记跳过）。
# 怎么跑：bash run_batch.sh /data/avsynth/avsynth.uproject [任务清单]
# 成功判据：tmux 里日志滚动、/data/queue/done 逐条增加；kill 后重跑已完成项被跳过。
# ⚠️ 单卡单进程并发=1：产能靠队列长度堆，不靠并发（同卡两个渲染进程只互相挤显存）。
# ⚠️ 绝不用 -headless：那是 NullRHI，根本不初始化图形后端，出不了图（附录C：出图全黑的根因）。
set -euo pipefail
UPROJECT="${1:?用法: run_batch.sh <xxx.uproject> [tasks.jsonl]}"
QUEUE="${2:-/data/queue/tasks.jsonl}"
UE_DIR="$(cd "$(dirname "$UPROJECT")" && pwd)"
UE_CMD="/data/UE_5.3/Engine/Binaries/Linux/UnrealEditor-Cmd"
LOG_DIR="/data/logs"
SESS="ue5_render"
mkdir -p "$LOG_DIR" /data/render/raw /data/render/meta /data/queue/done

[ -x "$UE_CMD" ] || { echo "ERROR: $UE_CMD 不存在（先跑 compile_ue5.sh）"; exit 1; }
[ -f "$QUEUE" ]  || { echo "ERROR: 任务清单 $QUEUE 不存在（先跑 queue_gen.py）"; exit 1; }

if tmux has-session -t "$SESS" 2>/dev/null; then
  echo "会话 $SESS 已在跑，不重复启动（附日志: $LOG_DIR/batch.log）"; exit 0
fi

# RenderOffscreen：保留 Vulkan 设备但不创建窗口；-Vulkan 显式指定后端
# 无 X11 会话时 Vulkan 设备可能创建失败 → 注释里备了 xvfb-run 兜底（环境兜底，非性能优化）
CMD="$UE_CMD $UPROJECT -ExecutePythonScript=$UE_DIR/Content/Python/batch_capture.py \
 -RenderOffscreen -unattended -nosplash -noSound -NoLiveView -Vulkan \
 -log=$LOG_DIR/batch.log"

tmux new-session -d -s "$SESS" "AD_UE_QUEUE=$QUEUE bash -c '$CMD; echo EXIT=\$? >> $LOG_DIR/batch.log; sleep 300'"
echo "已启动 tmux 会话 $SESS；查看: tmux attach -t $SESS  或  tail -f $LOG_DIR/batch.log"
echo "若 Vulkan 设备创建失败: tmux kill-session -t $SESS 后改用 xvfb-run -a --server-args='-screen 0 1920x1080x24' 包裹 $UE_CMD 再跑"
