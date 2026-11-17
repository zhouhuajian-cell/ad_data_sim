#!/usr/bin/env bash
# clone_ue5_resilient.sh — 稳健克隆 UE5.3（应对线路限流与 TLS 抖动）
#
# 背景与教训（都是实测踩出来的）：
#   1) 单 git 连接被限速到 ~150-500 KB/s，多开并行能到 ~3 MB/s，但**并行太高会触发线路限流**，
#      表现为 curl 首页都 000（连不上），静默 3~5 分钟才恢复。
#   2) git 默认的低速保护（1KB/s 持续 10 秒即断开）会把慢速下载误判为失败，
#      报错形如 "GnuTLS recv error (-110): The TLS connection was non-properly terminated"。
#      必须用 http.lowSpeedLimit/lowSpeedTime 放宽。
#   3) /etc/hosts 把 github.com 固定到 140.82.112.3（默认 DNS 解析到的 20.6.7.17 被企业网络拦截）。
#
# 策略：分阶段重试，每阶段失败就退避，绝不密集重试。
#   阶段1 取文件树（blob:none，几分钟）
#   阶段2 并行 checkout（默认 6 路，可在触发限流时降到 2 路）
#   阶段3 校验
# 怎么跑：bash /data/clone_ue5_resilient.sh   （后台跑，日志 /data/ue5_clone.log）
# 成功判据：日志最后一行 ALL_DONE rc=0。
set -uo pipefail

OUT=/data/UnrealEngine
LOG=/data/ue5_clone.log
PARTS=${PARTS:-6}
BRANCH=5.3

log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG" >/dev/null; }

TOKEN=$(cat /dev/shm/ghtok_classic 2>/dev/null | tr -d '\r\n')
if [ -z "$TOKEN" ]; then log "缺少令牌"; echo "ALL_DONE rc=1" >> "$LOG"; exit 1; fi
export B64=$(printf "x-access-token:%s" "$TOKEN" | base64 -w0)
unset TOKEN

GIT=(git -c http.version=HTTP/1.1 -c http.postBuffer=524288000
     -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=999
     -c "http.https://github.com/.extraheader=Authorization: Basic $B64"
     -c core.compression=0)

# 退避等待：第 n 次失败等 n*60 秒，最多等 5 分钟（避免触发限流加剧）
backoff() { local n="$1"; local s=$((n * 60)); [ "$s" -gt 300 ] && s=300; log "退避 ${s}s"; sleep "$s"; }

tidy_hosts() {
  if ! grep -q "140.82.112.3  github.com" /etc/hosts 2>/dev/null; then
    sed -i '/github\.com/d' /etc/hosts
    echo "140.82.112.3  github.com" >> /etc/hosts
    echo "140.82.112.3  codeload.github.com" >> /etc/hosts
    echo "140.82.112.6  api.github.com" >> /etc/hosts
  fi
}

# ---------------- 阶段 1：文件树 ----------------
log "=== 阶段1：取文件树（blob:none）==="
for attempt in 1 2 3 4 5 6; do
  tidy_hosts
  rm -rf "$OUT"
  if "${GIT[@]}" clone --depth 1 --branch "$BRANCH" --filter=blob:none --no-checkout \
       --progress https://github.com/EpicGames/UnrealEngine.git "$OUT" >> "$LOG" 2>&1; then
    if [ -d "$OUT/.git" ]; then log "阶段1 成功（第 $attempt 次）"; break; fi
  fi
  log "阶段1 第 $attempt 次失败"
  [ "$attempt" -ge 6 ] && { log "阶段1 放弃"; echo "ALL_DONE rc=2" >> "$LOG"; exit 2; }
  backoff "$attempt"
done

cd "$OUT" || exit 3
"${GIT[@]}" config remote.origin.promisor true
"${GIT[@]}" config remote.origin.partialclonefilter blob:none
TOTAL=$("${GIT[@]}" ls-tree -r --name-only HEAD | wc -l)
log "文件树就绪：$TOTAL 个文件"

# ---------------- 阶段 2：并行 checkout（带重试）----------------
log "=== 阶段2：并行 $PARTS 路取文件内容 ==="
mapfile -t ITEMS < <("${GIT[@]}" ls-tree --name-only HEAD)
N=${#ITEMS[@]}
log "顶层条目 $N 个，分给 $PARTS 路"

fetch_item() {   # $1=index $2=path
  local idx="$1" path="$2" sub="/data/ue5_part_${1}.log"
  for try in 1 2 3; do
    echo "[$(date +%F_%T)] [$idx] $path 第 $try 次" >> "$sub"
    if "${GIT[@]}" -C "$OUT" checkout HEAD -- "$path" >> "$sub" 2>&1; then
      echo "[$(date +%F_%T)] [$idx] $path OK" >> "$sub"; return 0
    fi
    echo "[$(date +%F_%T)] [$idx] $path 失败" >> "$sub"
    sleep $((try * 20))
  done
  echo "[$(date +%F_%T)] [$idx] $path 放弃" >> "$sub"; return 1
}

FAILED=0
i=0
for item in "${ITEMS[@]}"; do
  idx=$((i % PARTS))
  fetch_item "$idx" "$item" &
  i=$((i + 1))
  # 控制并发：每攒够 PARTS 个就等最老的一批
  if [ $((i % PARTS)) -eq 0 ]; then wait; fi
done
wait
log "阶段2 结束"

# ---------------- 阶段 3：校验 ----------------
GOT=$(find "$OUT" -type f -not -path "*/.git/*" | wc -l)
log "落盘文件：$GOT / 期望 $TOTAL"
rc=0
for f in "Setup.sh" "Engine/Build/BatchFiles/Linux/Build.sh" "Engine/Source/UnrealEditor.Target.cs"; do
  if [ -e "$OUT/$f" ]; then log "关键文件 OK: $f"; else log "关键文件缺失: $f"; rc=1; fi
done
# 对缺失文件做最后一轮串行补拉（限量，避免超时）
MISS=$(cd "$OUT" && "${GIT[@]}" ls-tree -r --name-only HEAD | while read -r f; do [ -e "$f" ] || echo "$f"; done | head -500)
if [ -n "$MISS" ]; then
  log "补拉缺失文件（最多 500 个）"
  echo "$MISS" | while read -r f; do
    "${GIT[@]}" -C "$OUT" checkout HEAD -- "$f" >/dev/null 2>&1 || true
  done
  GOT2=$(find "$OUT" -type f -not -path "*/.git/*" | wc -l)
  log "补拉后落盘：$GOT2 / 期望 $TOTAL"
fi
log "体积：$(du -sh "$OUT" 2>/dev/null | cut -f1)"
log "ALL_DONE rc=$rc"
echo "ALL_DONE rc=$rc" >> "$LOG"
exit $rc
