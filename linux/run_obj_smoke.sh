#!/usr/bin/env bash
# Server smoke test. Run as algo. Tools next to this script come from the server.
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "请用 algo 账号运行。" >&2
  exit 1
fi

tools="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project=/data/avsynth
engine=/data/UnrealEngine
python3 - "$project/Binaries/Linux/UnrealEditor.modules" "$engine/Engine/Binaries/Linux/UnrealEditor.modules" <<'PY'
import json
import sys
from pathlib import Path

project, engine = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
if not project.get("BuildId") or project["BuildId"] != engine.get("BuildId"):
    sys.exit("avsynth 模块和当前 UE 引擎的 BuildId 不一致。")
PY
out="$(mktemp -d /data/render/obj_smoke.XXXXXX)"
echo "输出目录 $out"

if ! OBJ_SMOKE_OUTPUT="$out" timeout 600 xvfb-run -a -s '-screen 0 1280x720x24' \
  "$engine/Engine/Binaries/Linux/UnrealEditor-Cmd" "$project/avsynth.uproject" \
  "-ExecutePythonScript=$project/Content/Python/preview_obj.py" \
  -Vulkan -unattended -nosplash -nosound > "$out/ue.log" 2>&1; then
  echo "UE 执行失败，详见 $out/ue.log" >&2
  exit 1
fi

if [ ! -f "$out/capture.ok" ]; then
  echo "UE 未完成三帧捕获，详见 $out/ue.log" >&2
  exit 1
fi

python3 "$tools/postprocess.py" --raw "$out" --meta "$out" --out "$out/images"
python3 "$tools/pack_dataset.py" --meta "$out" --images "$out/images" --out "$out/coco"
python3 "$tools/overlay_boxes.py" --meta "$out" --images "$out/images" \
  --out "$out/overlay" --limit 3 --cols 3
echo "抽检图 $out/overlay/contact_sheet.jpg"
