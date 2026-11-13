#!/usr/bin/env bash
# compile_ue5.sh — UE5.3 源码编译（方案 4.1）
# 干什么：浅克隆 5.3 分支 → Setup.sh 拉依赖 → GenerateProjectFiles.sh → 编译 UnrealEditor。
# 怎么跑：bash compile_ue5.sh [/data/UE_5.3]
# 成功判据：产物 Engine/Binaries/Linux/UnrealEditor 存在；make 不再报错。
# ⚠️ 内存峰值 26~30G、耗时 1~3 小时：必须在训练/抽帧/判定全停的窗口跑（OOM 已发生过 4 次）；
#    编译期间别做大批量文件读写。先跑 env_check.sh 全 PASS。
# 前置：Epic 账号已关联 GitHub（否则 clone 返回 403）。
set -euo pipefail
UE_DIR="${1:-/data/UE_5.3}"
NPROC="$(nproc)"
echo "== 目标 $UE_DIR，并行 $NPROC，开始 $(date +%F_%T)"

if [ ! -d "$UE_DIR/.git" ]; then
  # 只取 5.3 单分支最近一次提交：显著缩短拉取与后续编译时间
  git clone -b 5.3 --single-branch --depth 1 \
    https://github.com/EpicGames/UnrealEngine.git "$UE_DIR"
fi
cd "$UE_DIR"

# 关联检查（403 就是账号没关联，别重试浪费时间）
if git ls-remote HEAD >/dev/null 2>&1; then :; else
  echo "ERROR: GitHub 未与 Epic 关联（403），先完成账号关联再跑"; exit 1
fi

./Setup.sh
./GenerateProjectFiles.sh

# UE5 的目标名是 UnrealEditor（UE4Editor 在 UE5 已不存在，照搬会报目标不存在）
./Engine/Build/BatchFiles/Linux/Build.sh UnrealEditor Linux Development -j="$NPROC"

test -x Engine/Binaries/Linux/UnrealEditor && echo "OK: 编译完成 $(date +%F_%T)" \
  || { echo "ERROR: UnrealEditor 产物缺失"; exit 1; }
