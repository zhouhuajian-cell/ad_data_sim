import io
import os

TOOLS = "/opt/ue5_synth/tools"

# 生产必需（23 个）
KEEP = [
    # 核心管线
    "batch_walk.py", "render_selected.py", "prescreen_yolo.py",
    "pipeline_v411.py", "pipeline6.py", "batch_replace.py",
    "replace_sign.py", "auto_qc.py", "quality_contract.py",
    "tsr_param_preview.py",
    # 支撑
    "sign_realism2.py", "sign_occlusion.py", "sign_rotate.py",
    "fix_grow.py", "auto_tune.py", "walk_cache.py", "walk_index.py",
    # 接口
    "tsr_detect_api.py", "tsr_preview_api.py",
    # 新工具
    "make_gnd_tex.py", "est_camera.py", "check_blocks.py", "safe_patch.py",
]

# 新增：GND/OBJ 相关（在别的目录，但归到 tools 归类里说明）
DOC = {
    "batch_capture.py": "UE 硬件捕获脚本（在 /data/avsynth/Content/Python/）",
    "preview_scene.py": "场景预览（在 /data/avsynth/Content/Python/）",
    "setup_assets.py":  "UE 资产初始化（在 /data/avsynth/Content/Python/）",
}

all_py = sorted(f for f in os.listdir(TOOLS) if f.endswith(".py"))
all_bak = sorted(f for f in os.listdir(TOOLS) if ".bak" in f or ".py." in f)

to_delete_py = [f for f in all_py if f not in KEEP]

print("=" * 70)
print("清理计划")
print("=" * 70)

print("\n【保留】%d 个生产文件：" % len([f for f in KEEP if f in all_py]))
for f in KEEP:
    exists = "✅" if f in all_py else "❌"
    print("  %s %s" % (exists, f))

print("\n【删除】%d 个一次性脚本" % len(to_delete_py))
# 按前缀分类统计
cats = {}
for f in to_delete_py:
    if f.startswith("add_"):
        cats.setdefault("add_*（HTML/API 注入）", []).append(f)
    elif f.startswith("fix_"):
        cats.setdefault("fix_*（修复脚本）", []).append(f)
    elif f.startswith("patch_") or f.startswith("apply_"):
        cats.setdefault("patch_*/apply_*（补丁）", []).append(f)
    elif f.startswith("_") or f.startswith("chk") or f.startswith("check_"):
        cats.setdefault("chk*/_*（调试检查）", []).append(f)
    elif f.startswith("rm_") or f.startswith("kill_") or f.startswith("del_"):
        cats.setdefault("rm_*/kill_*（删除工具）", []).append(f)
    elif f.startswith("test_") or f.startswith("smoke_"):
        cats.setdefault("test_*/smoke_*（测试）", []).append(f)
    else:
        cats.setdefault("其它", []).append(f)

for k in sorted(cats, key=lambda x: -len(cats[x])):
    print("  %-32s %3d 个" % (k, len(cats[k])))

print("\n【删除】%d 个备份文件" % len(all_bak))

print("\n【保留的 3 个跨目录文件（不删，只归类）】")
for f, d in DOC.items():
    print("  %-22s %s" % (f, d))

# 白名单落盘
with open("/tmp/keep_final.txt", "w", encoding="utf-8") as fh:
    for f in KEEP:
        fh.write(f + "\n")

print("\n白名单 → /tmp/keep_final.txt")
