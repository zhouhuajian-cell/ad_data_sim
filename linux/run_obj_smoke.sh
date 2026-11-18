#!/usr/bin/env bash
# Server smoke test. Run as algo. Tools next to this script come from the server.
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "请用 algo 账号运行。" >&2
  exit 1
fi

tools="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module="${1:-obj}"
sample="${2:-traffic}"
case "$module:$sample" in
  obj:traffic|obj:truck|obj:workzone|gnd:traffic) ;;
  *) echo "小样类型不受支持。" >&2; exit 1 ;;
esac
exec 9>/data/render/.obj_smoke.lock
if ! flock -n 9; then
  echo "已有交通场景小样正在运行。" >&2
  exit 1
fi
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
out="$(mktemp -d "/data/render/${module}_smoke.XXXXXX")"
echo "输出目录 $out"

if ! OBJ_SMOKE_OUTPUT="$out" UE_SMOKE_MODULE="$module" UE_SMOKE_SAMPLE="$sample" \
  timeout 600 xvfb-run -a -s '-screen 0 1280x720x24' \
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
if [ "$module" = obj ]; then
  python3 "$tools/pack_dataset.py" --meta "$out" --images "$out/images" --out "$out/coco"
else
  python3 - "$out" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
frames = [json.loads(path.read_text()) for path in sorted(out.glob("gnd_*.json"))]
(out / "ground_annotations.json").write_text(json.dumps({"frames": frames}, indent=2))
PY
fi
if [ "$sample" = workzone ]; then
  python3 - "$out" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
frames = [json.loads(path.read_text()) for path in sorted(out.glob("obj_workzone_*.json"))]
(out / "scene_annotations.json").write_text(json.dumps({"frames": frames}, indent=2))
PY
fi
python3 "$tools/overlay_boxes.py" --meta "$out" --images "$out/images" \
  --out "$out/overlay" --limit 3 --cols 3
echo "抽检图 $out/overlay/contact_sheet.jpg"
