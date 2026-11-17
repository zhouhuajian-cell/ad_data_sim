#!/usr/bin/env bash
# full_pipeline.sh — 端到端数据生产：生成清单 → 渲染 → 后处理 → VOC → 难例 → 打包
#
# 干什么：把今天验证通过的完整链路固化下来，一条命令从任务清单跑到可用数据集。
# 怎么跑：
#   bash linux/full_pipeline.sh                 # 用小批 3 帧自检
#   FRAMES=24 bash linux/full_pipeline.sh       # 跑 24 帧
#   QUEUE=/data/queue/tasks_hc.jsonl bash linux/full_pipeline.sh   # 用指定清单
# 成功判据：结尾打印 PIPELINE_OK，且 dataset 目录里 images/annotations/hardcase 齐全。
#
# 关键前提（都踩过坑）：
#   1) 工程模块 BuildId 必须与引擎一致，否则 Python 完全不执行且无报错
#      —— 先跑 linux/build_project.sh 校验。
#   2) 必须用 xvfb 起虚拟显示，-RenderOffscreen 下 SceneCapture 只出黑图。
#   3) 任务清单里的 map 必须是真实存在的关卡（config/scenes.json 要指向实际关卡）。
set -uo pipefail

ROOT="${ROOT:-/opt/ue5_synth}"
UE_ROOT="${UE_ROOT:-/data/UnrealEngine}"
PROJECT="${PROJECT:-/data/avsynth/avsynth.uproject}"
RENDER="${RENDER:-/data/render}"
RUN_AS="${RUN_AS:-algo}"
FRAMES="${FRAMES:-3}"
QUEUE="${QUEUE:-/data/queue/tasks_hc.jsonl}"

UE_CMD="$UE_ROOT/Engine/Binaries/Linux/UnrealEditor-Cmd"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$PROJECT")}"
CAPTURE="$PROJECT_DIR/Content/Python/batch_capture.py"

say() { echo "[$(date +%F_%T)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ -x "$UE_CMD" ] || die "引擎不存在：$UE_CMD（先跑 build_ue5_from_source.sh）"
[ -f "$CAPTURE" ] || die "采集脚本不存在：$CAPTURE"
[ -f "$QUEUE" ] || die "任务清单不存在：$QUEUE（先跑 common/hardcase_gen.py）"

# ---------- 0. 前置：BuildId 校验 ----------
say "== 0/6 校验工程模块 BuildId"
PROJ_MODULES="$PROJECT_DIR/Binaries/Linux/UnrealEditor.modules"
ENG_MODULES="$UE_ROOT/Engine/Binaries/Linux/UnrealEditor.modules"
if [ -f "$PROJ_MODULES" ] && [ -f "$ENG_MODULES" ]; then
  P=$(python3 -c "import json;print(json.load(open('$PROJ_MODULES')).get('BuildId'))" 2>/dev/null)
  E=$(python3 -c "import json;print(json.load(open('$ENG_MODULES')).get('BuildId'))" 2>/dev/null)
  [ "$P" = "$E" ] || die "BuildId 不一致（工程=$P 引擎=$E），先跑 bash linux/build_project.sh"
  say "   BuildId OK：$P"
else
  say "   跳过（缺 modules 清单，先跑 build_project.sh）"
fi

# ---------- 1. 准备目录与清单 ----------
say "== 1/6 准备（取前 $FRAMES 帧）"
rm -rf "$RENDER/raw" "$RENDER/meta" "$RENDER/done" 2>/dev/null
mkdir -p "$RENDER/raw" "$RENDER/meta" "$RENDER/done" "$RENDER/final" "$RENDER/dataset"
head -n "$FRAMES" "$QUEUE" > /tmp/pipeline_queue.jsonl
chown -R "$RUN_AS":"$RUN_AS" "$RENDER" /tmp/pipeline_queue.jsonl 2>/dev/null || true

# ---------- 2. 渲染 ----------
say "== 2/6 渲染（xvfb + Vulkan）"
cd "$UE_ROOT" || die "进不去 $UE_ROOT"
runuser -u "$RUN_AS" -- xvfb-run -a --server-args="-screen 0 1280x720x24" \
  env AD_UE_QUEUE=/tmp/pipeline_queue.jsonl \
      AD_UE_RAW="$RENDER/raw" AD_UE_META="$RENDER/meta" AD_UE_DONE="$RENDER/done" \
      AD_UE_ERRLOG=/data/logs/capture_errors.log \
  timeout 3600 "$UE_CMD" "$PROJECT" -ExecutePythonScript="$CAPTURE" \
  -unattended -nosplash -noSound -NoLiveView -Vulkan -stdout > /tmp/pipeline_render.log 2>&1
RC=$?
grep -aE "\[capture\].*完成" /tmp/pipeline_render.log | tail -1

DONE=$(ls "$RENDER/done" 2>/dev/null | wc -l)
say "   渲染完成 $DONE 帧（rc=$RC）"
[ "$DONE" -gt 0 ] || { tail -5 /tmp/pipeline_render.log; die "一帧都没渲染出来，见 /tmp/pipeline_render.log"; }

# ---------- 3. 后处理 ----------
say "== 3/6 后处理（位深统一 / 运动模糊 / 8bit 成品）"
cd "$ROOT" || die "进不去 $ROOT"
python3 post/postprocess.py --raw "$RENDER/raw" --meta "$RENDER/meta" --out "$RENDER/final" 2>&1 | tail -2
FIN=$(ls "$RENDER/final"/*.png 2>/dev/null | wc -l)
[ "$FIN" -gt 0 ] || die "后处理没产出成品图"

# ---------- 4. VOC ----------
say "== 4/6 导出 PASCAL VOC XML"
python3 post/voc_export.py --meta "$RENDER/meta" --images "$RENDER/final" \
  --out "$RENDER/dataset/voc" 2>&1 | tail -1

# ---------- 5. 难例 ----------
say "== 5/6 难例判定"
python3 post/hardcase_export.py --meta "$RENDER/meta" --images "$RENDER/final" \
  --out "$RENDER/dataset/hardcase_list.csv" 2>&1 | tail -8

# ---------- 6. 打包 ----------
say "== 6/6 打包 COCO（带难例清单）"
python3 post/pack_dataset.py --meta "$RENDER/meta" --images "$RENDER/final" \
  --out "$RENDER/dataset" --format coco \
  --hardcase-csv "$RENDER/dataset/hardcase_list.csv" 2>&1 | tail -3

# ---------- 校验 ----------
say "== 产物校验"
FAIL=0
for p in "$RENDER/final" "$RENDER/dataset/voc/Annotations" "$RENDER/dataset/annotations"; do
  n=$(find "$p" -type f 2>/dev/null | wc -l)
  say "   $p: $n 个文件"
  [ "$n" -gt 0 ] || { say "   ✗ 空目录"; FAIL=1; }
done
[ -f "$RENDER/dataset/hardcase_list.csv" ] || { say "   ✗ 缺难例清单"; FAIL=1; }

echo "------------------------------------------------------------"
if [ "$FAIL" -eq 0 ]; then
  echo "PIPELINE_OK  帧=$DONE  成品=$FIN  数据集=$RENDER/dataset"
else
  echo "PIPELINE_FAIL 见上面的空目录"
  exit 1
fi
