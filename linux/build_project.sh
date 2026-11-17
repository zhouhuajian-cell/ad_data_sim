#!/usr/bin/env bash
# build_project.sh — 编译 avsynth 工程模块（唯一正确方式）
#
# 干什么：用源码引擎编译本工程。**必须带 LINUX_MULTIARCH_ROOT 等环境变量**，
#        否则 UBT 会用系统 clang（clang-10）并生成错误的 BuildId，
#        导致引擎把所有插件静默跳过（表现为 Python 完全不执行、无任何报错）。
#
# 背景（踩过的坑）：
#   1) 工程 Binaries/Linux/UnrealEditor.modules 里的 BuildId 必须等于引擎的 BuildId。
#      源码引擎是 UUID（如 84b22d57-...），旧预编译引擎是数字（如 27405482）。
#   2) 不一致时，引擎不会报错，只是**不加载任何模块/插件**，日志里连 LogPython 都没有。
#   3) 用错引擎的 UBT 编译（例如面板配置 ue_dir 指向 /data/UE_5.3）会再次写坏 BuildId。
#   4) 面板跑 setup_assets / preview 等作业时，如果工程模块与引擎不匹配，编辑器会自动重编，
#      同样会用错的 UBT —— 所以 ue_dir 配置必须指向源码引擎。
#
# 怎么跑：bash linux/build_project.sh
# 成功判据：输出的 BuildId 与引擎 UnrealEditor.modules 的 BuildId 一致。
set -uo pipefail

UE_ROOT="${UE_ROOT:-/data/UnrealEngine}"
PROJECT="${PROJECT:-/data/avsynth/avsynth.uproject}"
RUN_AS="${RUN_AS:-algo}"

SDK="$UE_ROOT/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v22_clang-16.0.6-centos7"
BIN="$SDK/x86_64-unknown-linux-gnu/bin"

if [ ! -x "$BIN/clang++" ]; then
  echo "ERROR: 找不到引擎自带工具链 $BIN/clang++" >&2
  exit 2
fi
if [ ! -f "$PROJECT" ]; then
  echo "ERROR: 工程不存在 $PROJECT" >&2
  exit 2
fi

echo "== 引擎 $UE_ROOT"
echo "== 工程 $PROJECT"
echo "== 工具链 $("$BIN/clang++" --version | head -1)"

chown -R "$RUN_AS":"$RUN_AS" "$(dirname "$PROJECT")" 2>/dev/null || true

runuser -u "$RUN_AS" -- env \
  LINUX_MULTIARCH_ROOT="$SDK" \
  CLANG="$BIN/clang++" \
  CC="$BIN/clang" \
  CXX="$BIN/clang++" \
  bash -c "cd '$UE_ROOT' && ./Engine/Build/BatchFiles/Linux/Build.sh avsynthEditor Linux Development -Project='$PROJECT' -Progress"

RC=$?
echo "== 编译返回码 $RC"

# 校验 BuildId 是否一致（这是最关键的检查，别只看返回码）
PROJ_MOD="$PROJECT"; PROJ_DIR="$(dirname "$PROJECT")"
PROJ_MODULES="$PROJ_DIR/Binaries/Linux/UnrealEditor.modules"
ENG_MODULES="$UE_ROOT/Engine/Binaries/Linux/UnrealEditor.modules"

if [ -f "$PROJ_MODULES" ] && [ -f "$ENG_MODULES" ]; then
  P=$(python3 -c "import json;print(json.load(open('$PROJ_MODULES')).get('BuildId'))" 2>/dev/null)
  E=$(python3 -c "import json;print(json.load(open('$ENG_MODULES')).get('BuildId'))" 2>/dev/null)
  echo "== 工程 BuildId: $P"
  echo "== 引擎 BuildId: $E"
  if [ "$P" != "$E" ]; then
    echo "ERROR: BuildId 不一致！引擎会静默跳过所有插件（Python 不执行）" >&2
    echo "       常见原因：用了别的引擎的 UBT 编译工程，或面板 config.ue_dir 指错了" >&2
    exit 3
  fi
  echo "== BuildId 一致，模块可被引擎加载"
else
  echo "ERROR: 找不到 modules 清单" >&2
  exit 3
fi

if [ "$RC" -ne 0 ]; then
  echo "ERROR: 编译失败（rc=$RC）" >&2
  exit "$RC"
fi
echo "== OK"
