#!/usr/bin/env bash
# run_spawn_sign.sh — 用上传素材在指定关卡生成一个路标（供面板 /api/signs/spawn 调用）
#
# 干什么：把「素材材质 + 关卡 + 位置 + 朝向 + 缩放」传给 traffic_sign_template.py，
#         用源码引擎在无头模式下执行，生成牌面+立柱并保存关卡。
# 用法：bash run_spawn_sign.sh <材质路径> <uproject或关卡> <pos> <yaw> <scale>
#   例：bash run_spawn_sign.sh /Game/Assets/Materials/M_Img_car /Game/Maps/codex_validation 2000,300,220 180 1.0
# 成功判据：日志出现 "[sign] OK save" 且 /tmp/sign_template.json 里 fatal 为空。
set -uo pipefail

UE_ROOT="${UE_ROOT:-/data/UnrealEngine}"
PROJECT="${PROJECT:-/data/avsynth/avsynth.uproject}"
RUN_AS="${RUN_AS:-algo}"
UE_CMD="$UE_ROOT/Engine/Binaries/Linux/UnrealEditor-Cmd"
SCRIPT="/data/avsynth/Content/Python/traffic_sign_template.py"

MAT="${1:-}"
MAP_ARG="${2:-/Game/Maps/codex_validation}"
POS="${3:-2000,300,220}"
YAW="${4:-180}"
SCALE="${5:-1.0}"

[ -n "$MAT" ] || { echo "ERROR: 缺少材质路径" >&2; exit 2; }
[ -x "$UE_CMD" ] || { echo "ERROR: 引擎不存在 $UE_CMD" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "ERROR: 模板脚本不存在 $SCRIPT" >&2; exit 2; }

# 第二参可能是 uproject（面板传 cfg.uproject）也可能是关卡路径，统一归一成关卡
if [ "${MAP_ARG#/Game/}" != "$MAP_ARG" ]; then
  MAP_PATH="$MAP_ARG"
else
  MAP_PATH="${AD_SIGN_MAP:-/Game/Maps/codex_validation}"
fi

chown -R "$RUN_AS":"$RUN_AS" /data/avsynth/Content/Python 2>/dev/null || true

runuser -u "$RUN_AS" -- env \
  AD_SIGN_MAT="$MAT" \
  AD_SIGN_MAP="$MAP_PATH" \
  AD_SIGN_POS="$POS" \
  AD_SIGN_YAW="$YAW" \
  AD_SIGN_SCALE="$SCALE" \
  bash -c "cd '$UE_ROOT' && timeout 600 '$UE_CMD' '$PROJECT' \
    -ExecutePythonScript='$SCRIPT' -NullRHI -unattended -nosplash -noSound -stdout"

RC=$?
echo "SPAWN_RC=$RC"
echo "MAP=$MAP_PATH  MAT=$MAT  POS=$POS  YAW=$YAW  SCALE=$SCALE"
if [ -f /tmp/sign_template.json ]; then
  echo "--- 结果 ---"
  cat /tmp/sign_template.json
fi
exit "$RC"
