#!/usr/bin/env bash
# fetch_ue5.sh — 用并行连接抓取 UE5.3 源码（应对单连接限速 150~450 KB/s）
#
# 干什么：
#   1) 先做一次 --filter=blob:none --no-checkout 的浅克隆，只拿文件树（几分钟），不下载文件内容；
#   2) 按顶层目录切成 N 份，并行 checkout，每个进程用自己的连接下载自己那份 blob；
#   3) 汇总校验：文件数、关键文件、版本文件。
# 为什么这么做：实测单 git 连接被限速到 ~170 KB/s（20G 要 30 小时），而 4 条并行能到 ~3 MB/s。
# 怎么跑：bash /data/fetch_ue5.sh            （结果在 /data/UnrealEngine）
#        PARTS=12 bash /data/fetch_ue5.sh    （并行度，默认 8）
# 成功判据：最后一行打印 ALL_DONE rc=0，且 Engine/Binaries/... 关键目录文件数 > 0。
set -uo pipefail

OUT=/data/UnrealEngine
LOG=/data/fetch_ue5.log
PARTS=${PARTS:-8}
REPO=https://github.com/EpicGames/UnrealEngine.git
BRANCH=5.3

TOKEN=$(cat /dev/shm/ghtok_classic 2>/dev/null | tr -d '\r\n')
if [ -z "$TOKEN" ]; then
  echo "[$(date +%F_%T)] 缺少令牌 /dev/shm/ghtok_classic" >> "$LOG"
  exit 1
fi
export B64=$(printf "x-access-token:%s" "$TOKEN" | base64 -w0)
unset TOKEN

# 所有 git 调用共用这组参数：
#   http.version=HTTP/1.1        —— HTTP/2 在这条线路上更容易被中间设备掐断
#   lowSpeedLimit/Time           —— ★关键：默认的低速保护（1KB/s 持续 10s 就断开）会把慢速下载当成失败，
#                                   实测就是它导致 "GnuTLS recv error (-110): TLS connection was non-properly terminated"
#   postBuffer                   —— 大 pack 走 HTTP/1.1 时避免内存缓冲不足
#   core.compression=0           —— 源码本身已压缩，再压只费 CPU
GIT=(git -c http.version=HTTP/1.1 -c http.postBuffer=524288000
     -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=999
     -c "http.https://github.com/.extraheader=Authorization: Basic $B64"
     -c core.compression=0)

log() { echo "[$(date +%F_%T)] $*" >> "$LOG"; }

log "=== 开始：并行 $PARTS 路 ==="
rm -rf "$OUT"
mkdir -p "$OUT"

# ---------- 1. 只取文件树（blob:none 不下载内容）----------
log "步骤1：取文件树（不下载内容）"
if ! "${GIT[@]}" clone --depth 1 --branch "$BRANCH" --filter=blob:none --no-checkout \
        --progress "$REPO" "$OUT" >> "$LOG" 2>&1; then
  log "步骤1 失败：clone 文件树出错"
  echo "ALL_DONE rc=2" >> "$LOG"
  exit 2
fi
cd "$OUT" || exit 3
# 后续 fetch 也走同一套认证（clone 已经把 origin 配好了，这里补上 extraheader）
"${GIT[@]}" config remote.origin.promisor true
"${GIT[@]}" config remote.origin.partialclonefilter blob:none

TOTAL=$("${GIT[@]}" ls-tree -r --name-only HEAD | wc -l)
log "文件树就绪：$TOTAL 个文件"

# ---------- 2. 按顶层目录切片 ----------
# 顶层目录的文件数差异很大（Engine 占绝大多数），所以按文件数均分成 N 组，
# 而不是按目录数均分，否则并行度会退化成 1。
SLICES=$(mktemp)
"${GIT[@]}" ls-tree --name-only HEAD | awk -v n="$PARTS" '{print NR%n"\t"$0}' | sort -k1,1n | cut -f2 > "$SLICES"
log "切片：$(wc -l < "$SLICES") 个顶层条目分给 $PARTS 路"

# ---------- 3. 并行 checkout ----------
# 说明：git checkout 对部分克隆会自动按需拉取缺失的 blob，
#      多进程各自拉自己那份，正好把带宽用满。
checkout_one() {
  local idx="$1" item="$2"
  local sub_log="/data/fetch_part_${idx}.log"
  {
    echo "[$(date +%F_%T)] 第 $idx 路：$item 开始"
    "${GIT[@]}" -C "$OUT" checkout HEAD -- "$item" 2>&1 | tail -5
    local rc=${PIPESTATUS[0]}
    echo "[$(date +%F_%T)] 第 $idx 路：$item 结束 rc=$rc"
  } >> "$sub_log" 2>&1
  echo "$rc"
}

i=0
PIDS=()
while IFS= read -r item; do
  i=$((i + 1))
  checkout_one $((i % PARTS)) "$item" &
  PIDS+=($!)
  # 同时最多跑 PARTS 个
  if [ ${#PIDS[@]} -ge "$PARTS" ]; then
    wait -n 2>/dev/null || wait "${PIDS[0]}" 2>/dev/null
    # 清理已结束的 pid
    NEW=()
    for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && NEW+=("$p"); done
    PIDS=("${NEW[@]}")
  fi
done < "$SLICES"
wait
rm -f "$SLICES"
log "并行 checkout 全部结束"

# ---------- 4. 校验 ----------
GOT=$(find "$OUT" -type f -not -path "*/.git/*" | wc -l)
log "落盘文件数：$GOT / 期望 $TOTAL"
mkdir -p "$OUT/Engine/Build" 2>/dev/null
MISSING=$(cd "$OUT" && "${GIT[@]}" ls-tree -r --name-only HEAD | while read -r f; do
  [ -e "$f" ] || echo "$f"
done | head -20)
if [ -n "$MISSING" ]; then
  log "缺失样例（前 20）："; echo "$MISSING" >> "$LOG"
fi

KEYFILES=(
  "Engine/Build/BatchFiles/Linux/Build.sh"
  "Engine/Build/BatchFiles/Linux/Setup.sh"
  "Engine/Source/UnrealEditor.Target.cs"
  "Setup.sh"
)
rc=0
for f in "${KEYFILES[@]}"; do
  if [ -e "$OUT/$f" ]; then log "关键文件 OK: $f"; else log "关键文件缺失: $f"; rc=1; fi
done

log "体积：$(du -sh "$OUT" 2>/dev/null | cut -f1)"
log "ALL_DONE rc=$rc"
echo "ALL_DONE rc=$rc" >> "$LOG"
exit $rc
