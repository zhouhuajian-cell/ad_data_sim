#!/usr/bin/env bash
# gpu_watch.sh — 渲染期资源监控（方案 4.6）
# 干什么：1 秒间隔记录显存/GPU 利用率/温度/功耗到 CSV；显存 >10.5G 打告警行。
# 怎么跑：bash gpu_watch.sh [输出csv]   （Ctrl-C 停）
# 成功判据：CSV 持续追加；超阈值时 stderr 出现 WARNING。
# ⚠️ 显存持续 >10.5G 的处置顺序：先降阴影分辨率 → 再减同屏目标 → **不要先降分辨率**
#    （分辨率会改变小目标像素占位，直接影响检测难度分布）。
CSV="${1:-/data/logs/gpu_watch.csv}"
mkdir -p "$(dirname "$CSV")"
[ -f "$CSV" ] || echo "ts,gpu_mem_mb,gpu_util_pct,temp_c,power_w" > "$CSV"
WARN=0
echo "写入 $CSV （Ctrl-C 停止）"
while true; do
  L=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$L" ] && { sleep 1; continue; }
  MEM=$(echo "$L" | cut -d, -f1 | tr -d ' ')
  echo "$(date +%F_%T),$L" >> "$CSV"
  if [ "${MEM:-0}" -gt 10500 ]; then
    WARN=$((WARN+1))
    # 连续超限才提示（偶发尖峰属正常分配），避免刷屏
    if [ $((WARN % 10)) -eq 0 ]; then
      echo "WARNING: 显存 ${MEM}MB 持续 >10.5G（第 ${WARN} 次）→ 降阴影分辨率/减同屏目标" >&2
    fi
  else
    WARN=0
  fi
  sleep 1
done
