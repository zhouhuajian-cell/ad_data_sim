#!/usr/bin/env bash
# env_check.sh — Ubuntu 侧环境基线核查（表 3 + 表 11 阶段一）
# 干什么：只读检查发行版/驱动/显存/磁盘/内存/swap/tmux/ssh/vulkan，逐项 PASS/FAIL。
# 怎么跑：bash env_check.sh
# 成功判据：末行 "RESULT: ALL PASS" 才允许进入阶段二；FAIL 项按提示补齐。
# ⚠️ 只读，不改任何系统配置。
set -u
FAIL=0
ok(){ printf "PASS  %s\n" "$1"; }
ng(){ printf "FAIL  %s — %s\n" "$1" "$2"; FAIL=$((FAIL+1)); }

# 发行版：基线 22.04；2026-11-13 用户拍板"先在 20.04 试编译"，故 20.04 同样放行（编不过再谈升级）
. /etc/os-release 2>/dev/null
case "${VERSION_ID:-}" in
  22.04) ok "发行版 Ubuntu 22.04" ;;
  20.04) ok "发行版 Ubuntu 20.04（已拍板试编译路线；方案基线 22.04）" ;;
  *)     ng "发行版" "当前 ${VERSION_ID:-未知}，要求 22.04 LTS" ;;
esac

# 驱动：550 系列闭源（560+ 在 UE5.3 下反光材质发黑/黑屏）
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
case "$DRV" in
  550.*) ok "NVIDIA 驱动 $DRV（550 系列）" ;;
  "")    ng "驱动" "nvidia-smi 不可用（未装驱动或无 N 卡）" ;;
  *)     ng "驱动" "当前 $DRV，要求 550 系列（560+ 有材质发黑问题）" ;;
esac

# 显卡与显存
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
[ -n "$GPU" ] && ok "GPU: $GPU" || ng "GPU" "查不到"

# 图形后端 Vulkan 1.3
if command -v vulkaninfo >/dev/null 2>&1; then
  if vulkaninfo 2>/dev/null | grep -qi "apiVersion.*1\.[3-9]"; then ok "Vulkan ≥1.3"; else ng "Vulkan" "版本低于 1.3 或设备识别失败"; fi
else
  ng "Vulkan" "vulkaninfo 不存在（装 vulkan-tools）"
fi

# 容量：系统盘 ≥150G（源码+编译产物约120G）
ROOT_FREE=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${ROOT_FREE:-0}" -ge 150 ] && ok "系统盘可用 ${ROOT_FREE}G ≥150G" || ng "系统盘" "可用 ${ROOT_FREE:-?}G <150G"

# 数据盘：/data 存在且 ≥1T（单帧 16~22MB；没有 /data 就查挂载点）
if mountpoint -q /data 2>/dev/null || [ -d /data ]; then
  D_FREE=$(df -BG --output=avail /data 2>/dev/null | tail -1 | tr -dc '0-9')
  [ "${D_FREE:-0}" -ge 1000 ] && ok "/data 可用 ${D_FREE}G ≥1T" || ng "/data" "可用 ${D_FREE:-?}G <1T"
else
  ng "/data" "不存在（渲染输出要落本地 NVMe，不能落网盘）"
fi

# 内存 32G
MEMG=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
[ "$MEMG" -ge 30 ] && ok "内存 ${MEMG}G" || ng "内存" "${MEMG}G <32G（编译峰值 26~30G）"

# swap 关闭（渲染/编译期间 swap 显著拖慢 Vulkan 提交）
if swapon --show | grep -q .; then ng "swap" "仍开启：$(swapon --show --noheadings | tr '\n' ' ')"; else ok "swap 已关闭"; fi

# 会话工具
command -v tmux >/dev/null 2>&1 && ok "tmux" || ng "tmux" "未安装"
command -v sshd >/dev/null 2>&1 || systemctl is-active --quiet ssh 2>/dev/null && ok "sshd" || ng "sshd" "未运行"

# 磁盘 IO 不在网盘上（输出目录不能是 CIFS/SMB）
if mount | grep -E " /data " | grep -qiE "cifs|smb"; then ng "/data" "挂的是网络盘，逐帧小 IO 会拖慢数倍"; fi

if [ "$FAIL" -eq 0 ]; then echo "RESULT: ALL PASS"; else echo "RESULT: $FAIL 项 FAIL"; fi
exit $FAIL
