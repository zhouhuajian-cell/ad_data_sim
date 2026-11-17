#!/usr/bin/env bash
# build_ue5_from_source.sh — 源码编译 UE5.3（克隆完成后一键接管）
#
# 干什么：在已有 /data/UnrealEngine 源码树的前提下：
#   1) Setup.sh 拉 Linux 原生工具链（clang-16 + sysroot）
#   2) GenerateProjectFiles.sh 生成工程文件
#   3) Build.sh 编译 UnrealEditor（20 核满跑）
#   4) 校验产物可执行
# 为什么用源码而不是预编译版：预编译 5.3 在这台 Ubuntu 20.04 上开 Vulkan 会崩
#   （栈顶 FMallocMimalloc::Malloc，SIGSEGV）；源码编译可避开该分配器问题。
# 怎么跑：bash /data/build_ue5_from_source.sh   （后台跑，日志 /data/ue5_build.log，1~3 小时）
# 成功判据：日志最后 ALL_DONE rc=0，且 Engine/Binaries/Linux/UnrealEditor-Cmd 存在且可执行。
set -uo pipefail

SRC=/data/UnrealEngine
LOG=/data/ue5_build.log
JOBS=${JOBS:-20}

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG" >/dev/null; }

if [ ! -d "$SRC/.git" ]; then
  log "源码目录不存在：$SRC（先跑 clone_ue5_resilient.sh）"
  echo "ALL_DONE rc=2" >> "$LOG"; exit 2
fi
cd "$SRC" || exit 3

# 编译用户用非 root（UE 拒绝 root 编译/运行），目录先交给他
RUN_AS=${RUN_AS:-algo}
if ! id "$RUN_AS" >/dev/null 2>&1; then
  log "运行用户 $RUN_AS 不存在"; echo "ALL_DONE rc=2" >> "$LOG"; exit 2
fi
chown -R "$RUN_AS":"$RUN_AS" "$SRC" 2>/dev/null || true

run_as() { runuser -u "$RUN_AS" -- bash -lc "$*"; }

# ---------------- 1. Setup.sh：拉工具链 ----------------
log "=== 1/3 Setup.sh（下载 clang-16 工具链，约 10~30 分钟）==="
if [ ! -x "$SRC/Engine/Binaries/Linux/UnrealEditor-Cmd" ]; then
  if ! run_as "cd '$SRC' && ./Setup.sh" >> "$LOG" 2>&1; then
    log "Setup.sh 失败（常见：网络拉不动依赖，或磁盘不足）"
    # 不直接退出：Setup.sh 失败但工具链可能已就绪，继续尝试编译
  fi
fi

# ---------------- 2. 生成工程文件 ----------------
log "=== 2/3 GenerateProjectFiles ==="
run_as "cd '$SRC' && ./GenerateProjectFiles.sh -vscode" >> "$LOG" 2>&1 || log "GenerateProjectFiles 有告警（通常不影响编译）"

# ---------------- 3. 编译 UnrealEditor ----------------
log "=== 3/3 编译 UnrealEditor（-j$JOBS，峰值内存可能到 26G）==="
if run_as "cd '$SRC' && ./Engine/Build/BatchFiles/Linux/Build.sh UnrealEditor Linux Development -j$JOBS -Progress" >> "$LOG" 2>&1; then
  log "编译命令返回 0"
else
  log "编译命令失败，详见日志尾部"
fi

# ---------------- 校验 ----------------
UE_CMD="$SRC/Engine/Binaries/Linux/UnrealEditor-Cmd"
rc=0
if [ -x "$UE_CMD" ]; then
  log "产物 OK：$UE_CMD（$(du -h "$UE_CMD" | cut -f1)）"
  # 冒烟：非 root 跑 -version，确认真能启动（这一步能过才谈得上出图）
  out=$(runuser -u "$RUN_AS" -- bash -lc "cd '$SRC' && timeout 90 ./Engine/Binaries/Linux/UnrealEditor-Cmd -version" 2>&1 | head -5)
  log "冒烟输出：$(echo "$out" | tr '\n' ' ' | cut -c1-200)"
else
  log "产物缺失：$UE_CMD"
  rc=1
fi

log "体积：$(du -sh "$SRC" 2>/dev/null | cut -f1)"
log "ALL_DONE rc=$rc"
echo "ALL_DONE rc=$rc" >> "$LOG"
exit $rc
