#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modules/build_scene.py — 用 TSR/OBJ/GND 三个模块组装一个完整场景

干什么：按一份「场景描述 JSON」在 UE 关卡里摆好所有目标并保存，产出：
  - 关卡文件（.umap）
  - 真值清单 JSON（每个目标的世界盒/类别/stencil），供 batch_capture 投影

为什么单独有这个组装脚本：
  三个模块各自只负责「生成什么」，谁来摆、摆多少、怎么组合是场景的事。
  把组合逻辑抽出来，加新场景只要写一份 JSON，不用改代码。

场景描述格式（示例见 modules/scenes_demo.json）：
  {
    "name": "demo_straight",
    "map": "/Game/Maps/codex_validation",
    "gnd": [{"type": "lane_solid_white", "start_cm": [0,-550], "length_cm": 4000, "yaw": 0}],
    "tsr": [{"class": "speed_limit", "pos_cm": [2000, 300, 220], "yaw": 180, "material": ""}],
    "obj": [{"class": "car", "pos_cm": [1500, -180, 0], "yaw": 180}]
  }

用法（服务器）：
  xvfb-run -a UnrealEditor-Cmd <uproject> -ExecutePythonScript=modules/build_scene.py -NullRHI -unattended
  环境变量：AD_SCENE_JSON=<场景描述路径>（默认 modules/scenes_demo.json）
"""
import json
import os
import sys
import traceback

# 三个模块都在本目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsr as tsr_mod      # noqa: E402
import obj as obj_mod      # noqa: E402
import gnd as gnd_mod      # noqa: E402

DEFAULT_SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes_demo.json")
OUT_TRUTH = "/data/render/scene_truth.json"

report = {"steps": []}


def rec(name, ok, value=""):
    report["steps"].append({"step": name, "ok": ok, "value": str(value)[:300]})
    print("[scene] %s %s %s" % ("OK " if ok else "ERR", name, value))


try:
    import unreal

    scene_path = os.environ.get("AD_SCENE_JSON", DEFAULT_SCENE)
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    rec("load_scene", True, "%s（%s）" % (scene.get("name"), scene_path))

    map_path = scene.get("map", "/Game/Maps/codex_validation")
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    les.load_level(map_path)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    rec("load_level", True, map_path)

    # 清理上次生成的目标（按标签前缀识别，不误删场景自带物体）
    removed = 0
    for a in list(eas.get_all_level_actors()):
        lab = a.get_actor_label()
        if lab.startswith(("TSR_", "OBJ_", "GND_")):
            eas.destroy_actor(a)
            removed += 1
    rec("cleanup", True, "移除 %d 个旧目标" % removed)

    truths = []

    # ---- GND ----
    for item in scene.get("gnd", []):
        try:
            gnd_mod.spawn_line(unreal, eas, item["type"],
                               tuple(item.get("start_cm", [0, 0])),
                               float(item.get("length_cm", 2000)),
                               float(item.get("yaw", 0.0)),
                               float(item.get("wear", 0.0)))
            start = item.get("start_cm", [0, 0])
            length = float(item.get("length_cm", 2000))
            yaw = float(item.get("yaw", 0.0))
            import math
            rad = math.radians(yaw)
            cx = start[0] + math.cos(rad) * length * 0.5
            cy = start[1] + math.sin(rad) * length * 0.5
            truths.append(gnd_mod.truth_entry(item["type"], [cx, cy, gnd_mod.GROUND_Z_CM],
                                              length, yaw, float(item.get("wear", 0.0))))
        except Exception as e:
            rec("gnd_" + str(item.get("type")), False, e)
    rec("gnd_total", True, "%d 条标线" % len(scene.get("gnd", [])))

    # ---- TSR ----
    for item in scene.get("tsr", []):
        try:
            tsr_mod.spawn_sign(unreal, eas, les, item["class"],
                               item.get("material", ""),
                               tuple(item.get("pos_cm", [2000, 300, 220])),
                               float(item.get("yaw", 180.0)),
                               float(item.get("scale", 1.0)))
            truths.append(tsr_mod.truth_entry(item["class"],
                                              tuple(item.get("pos_cm", [2000, 300, 220])),
                                              float(item.get("yaw", 180.0)),
                                              float(item.get("scale", 1.0)),
                                              float(item.get("wear", 0.0))))
        except Exception as e:
            rec("tsr_" + str(item.get("class")), False, e)
    rec("tsr_total", True, "%d 个路标" % len(scene.get("tsr", [])))

    # ---- OBJ ----
    for item in scene.get("obj", []):
        try:
            obj_mod.spawn_object(unreal, eas, item["class"],
                                 tuple(item.get("pos_cm", [1500, 0, 0])),
                                 float(item.get("yaw", 0.0)),
                                 float(item.get("scale", 1.0)))
            truths.append(obj_mod.truth_entry(item["class"],
                                              tuple(item.get("pos_cm", [1500, 0, 0])),
                                              float(item.get("yaw", 0.0)),
                                              float(item.get("scale", 1.0))))
        except Exception as e:
            rec("obj_" + str(item.get("class")), False, e)
    rec("obj_total", True, "%d 个交通目标" % len(scene.get("obj", [])))

    les.save_current_level()
    rec("save", True, map_path)

    # 真值清单
    os.makedirs(os.path.dirname(OUT_TRUTH), exist_ok=True)
    with open(OUT_TRUTH, "w", encoding="utf-8") as f:
        json.dump({"scene": scene.get("name"), "map": map_path,
                   "objects": truths}, f, indent=1, ensure_ascii=False)
    rec("truth", True, "%d 个目标 → %s" % (len(truths), OUT_TRUTH))
    report["object_count"] = len(truths)

except Exception:
    report["fatal"] = traceback.format_exc()[:1500]

with open("/tmp/build_scene.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("DONE")
