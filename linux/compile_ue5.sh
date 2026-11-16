#!/usr/bin/env bash
# compile_ue5.sh — UE5.3 源码编译（方案 4.1）
# 干什么：装依赖 → 浅克隆 5.3 分支 → Setup.sh 拉工具链 → GenerateProjectFiles.sh → 编译 UnrealEditor。
# 怎么跑：bash compile_ue5.sh [/data/UE_5.3]          SKIP_DEPS=1 跳过 apt（依赖已全装时）
# 成功判据：产物 Engine/Binaries/Linux/UnrealEditor 存在；最后一行 "OK: 编译完成"。
# ⚠️ 内存峰值 26~30G、耗时 1~3 小时：必须在训练/抽帧/判定全停的窗口跑（OOM 已发生过 4 次）；
#    编译期间别做大批量文件读写。先跑 env_check.sh 全 PASS。
# 前置：Epic 账号已关联 GitHub（否则 clone 返回 403）。
#
# 20.04 兼容说明（2026-11-13 拍板先在 20.04 试编译）：
#   Epic 5.3 的 Linux 官方要求是 Ubuntu 22.04 + clang 16.0.6
#   （https://dev.epicgames.com/documentation/unreal-engine/linux-development-requirements-for-unreal-engine?application_version=5.3），
#   而 Ubuntu 20.04 官方源没有 clang-16。这里从 LLVM 官方 apt 源补 clang-16，并把 cc/c++ 指过去；
#   Setup.sh 还会自带下载 Linux 原生工具链 v22（clang-16.0.6 + 固定 sysroot），两者二选一都能过 UBT 的版本检查。
#   若 20.04 上 UBT 仍因 libc++/sysroot 不匹配失败，按方案基线升 22.04（env_check 已把它标为基线）。
set -euo pipefail
UE_DIR="${1:-/data/UE_5.3}"
NPROC="$(nproc)"
SKIP_DEPS="${SKIP_DEPS:-0}"
echo "== 目标 $UE_DIR，并行 $NPROC，开始 $(date +%F_%T)"

# ---- 编译依赖：幂等安装（已装会跳过） ------------------------------------------------------
if [ "$SKIP_DEPS" != "1" ]; then
  echo "== 安装编译依赖（幂等）"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq \
    build-essential clang lld libc++-dev libc++abi-dev \
    mono-devel mono-dgm dotnet-sdk-6.0 \
    libx11-dev libxcb1-dev libgl1-mesa-dev libglu1-mesa-dev \
    libxkbcommon-dev libxkbcommon-x11-0 libfontconfig1-dev libfreetype6-dev \
    curl gnupg git tzdata >/dev/null || echo "WARN: 部分 apt 包安装失败，编译可能继续但仍需补齐"
fi

# ---- clang-16：5.3 官方要求；20.04 官方源没有 → LLVM 官方 apt 源（幂等）----------------------
if ! command -v clang-16 >/dev/null 2>&1; then
  echo "== 未发现 clang-16，尝试从 LLVM 官方 apt 源安装"
  . /etc/os-release
  LLVM_CODENAME="${UBUNTU_CODENAME:-$(lsb_release -cs 2>/dev/null || echo focal)}"
  if ! curl -fsSL "https://apt.llvm.org/llvm-snapshot.gpg.key" \
       | gpg --dearmor -o /usr/share/keyrings/llvm.gpg 2>/dev/null; then
    echo "ERROR: 拉取 LLVM 签名失败（网络问题？）。可手动装 clang-16 后 SKIP_DEPS=1 重跑"
    exit 1
  fi
  echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/${LLVM_CODENAME}/ llvm-toolchain-${LLVM_CODENAME}-16 main" \
    > /etc/apt/sources.list.d/llvm-16.list
  export DEBIAN_FRONTEND=noninteractive
  if ! apt-get update -qq || ! apt-get install -y -qq clang-16 lld-16 libclang-16-dev >/dev/null; then
    echo "ERROR: clang-16 安装失败；20.04 官方源没有该版本，需 LLVM 源可用或升 22.04"
    rm -f /etc/apt/sources.list.d/llvm-16.list
    exit 1
  fi
fi
CLANG_BIN="$(command -v clang-16 || true)"
CLANGXX_BIN="$(command -v clang++-16 || true)"
if [ -z "$CLANG_BIN" ] || [ -z "$CLANGXX_BIN" ]; then
  echo "ERROR: clang-16 / clang++-16 不可用"; exit 1
fi
# UBT 的 Linux 平台优先用环境变量里的编译器；20.04 自带的 clang-10 不能被选中
export CLANG="$CLANG_BIN"
export CC="$CLANG_BIN"
export CXX="$CLANGXX_BIN"
echo "== 使用 $("$CLANG_BIN" --version 2>/dev/null | head -1)"

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

# Setup.sh 会下载 Linux 原生工具链 v22（clang-16.0.6 + 固定 sysroot）；-vscode 只为生成工程文件
./Setup.sh
./GenerateProjectFiles.sh -vscode

# UE5 的目标名是 UnrealEditor（UE4Editor 在 UE5 已不存在，照搬会报目标不存在）
./Engine/Build/BatchFiles/Linux/Build.sh UnrealEditor Linux Development -j="$NPROC"

test -x Engine/Binaries/Linux/UnrealEditor && echo "OK: 编译完成 $(date +%F_%T)" \
  || { echo "ERROR: UnrealEditor 产物缺失"; exit 1; }
