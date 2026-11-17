#!/usr/bin/env bash
# Prepare the project module, then run isolated validation asset generation.
set -euo pipefail
PROJECT="$1"
UE_ROOT="$2"
PROJECT_DIR="$(dirname "$PROJECT")"
UE_CMD="$UE_ROOT/Engine/Binaries/Linux/UnrealEditor-Cmd"
SCRIPT="$PROJECT_DIR/Content/Python/${3:-setup_assets.py}"
RUN_AS="${AD_UE_RUN_AS:-algo}"
test -f "$PROJECT" && test -x "$UE_CMD" && test -f "$SCRIPT"
if [ "$(id -u)" = "0" ]; then
  chown -R "$RUN_AS:$RUN_AS" "$PROJECT_DIR" 2>/dev/null || true
  AS=(runuser -u "$RUN_AS" --)
else
  AS=()
fi
module_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("BuildId", ""))' "$PROJECT_DIR/Binaries/Linux/UnrealEditor.modules" 2>/dev/null || true)
engine_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("BuildId", ""))' "$UE_ROOT/Engine/Binaries/Linux/UnrealEditor.modules" 2>/dev/null || true)
if ! ls "$PROJECT_DIR/Binaries/Linux/libUnrealEditor-avsynth"*.so >/dev/null 2>&1 \
   || [ -z "$engine_id" ] || [ "$module_id" != "$engine_id" ]; then
  echo "工程模块缺失或 BuildId 不匹配，开始编译 avsynthEditor（module=$module_id engine=$engine_id）"
  "${AS[@]}" "$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh" avsynthEditor Linux Development \
    "-Project=$PROJECT" -Progress
  module_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("BuildId", ""))' "$PROJECT_DIR/Binaries/Linux/UnrealEditor.modules" 2>/dev/null || true)
  [ "$module_id" = "$engine_id" ] || { echo "ERROR: 编译后 BuildId 仍不匹配" >&2; exit 1; }
fi
if [ "${3:-}" = "preview_scene.py" ]; then
  rm -f "$PROJECT_DIR/Saved/web_preview.png" "$PROJECT_DIR/Saved/web_preview.exr"
fi
"${AS[@]}" xvfb-run -a --server-args='-screen 0 1280x720x24' \
  "$UE_CMD" "$PROJECT" "-ExecutePythonScript=$SCRIPT" -Vulkan -unattended -nosplash

if [ "${3:-}" = "preview_scene.py" ]; then
  PROJECT_DIR="$PROJECT_DIR" python3 - <<'PY'
import os, sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
p=os.path.join(os.environ['PROJECT_DIR'],'Saved','web_preview')
if not os.path.isfile(p+'.png') and os.path.isfile(p+'.exr'):
    im=cv2.imread(p+'.exr',cv2.IMREAD_UNCHANGED)
    if im is None: sys.exit('EXR 读取失败')
    import numpy as np
    im=np.clip(im[:,:,:3],0,1)
    if not cv2.imwrite(p+'.png',(im*255).astype('uint8')): sys.exit('PNG 写入失败')
if not os.path.isfile(p+'.png') or os.path.getsize(p+'.png') < 100: sys.exit('预览图未生成')
print('PREVIEW_OK',p+'.png')
PY
fi
