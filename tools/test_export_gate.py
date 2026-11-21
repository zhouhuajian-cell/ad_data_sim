"""验证 export_gate 的 fail-closed 语义。"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/opt/ue5_synth/tools")
from export_gate import (REQUIRED_CHECKS, build_checks_from_artifacts,
                         export_decision, make_check)

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print("  OK   %s %s" % (name, detail))
    else:
        failed += 1
        print("  FAIL %s %s" % (name, detail))


print("=== 1) 全空 -> 全阻塞 ===")
d = export_decision({})
check("不是 eligible", d["training_eligible"] is False)
check("9 项全阻塞", len(d["training_blockers"]) >= 9,
      "blockers=%d" % len(d["training_blockers"]))
check("passed=0", d["passed"] == 0)

print("\n=== 2) 全部 passed 且有证据 -> eligible ===")
checks = {}
for n in REQUIRED_CHECKS:
    checks[n] = make_check("passed", evidence_obj={"k": n})
d = export_decision(checks)
check("是 eligible", d["training_eligible"] is True)
check("无阻塞", d["training_blockers"] == [])
check("passed=9", d["passed"] == 9)

print("\n=== 3) 全部 passed 但无证据 -> 不 eligible ===")
checks = {n: {"status": "passed", "note": ""} for n in REQUIRED_CHECKS}
d = export_decision(checks)
check("不是 eligible", d["training_eligible"] is False)
check("每个都报 no_evidence",
      all(b.endswith("_no_evidence") for b in d["training_blockers"]),
      "n=%d" % len(d["training_blockers"]))

print("\n=== 4) 一项 unchecked -> 阻塞 ===")
checks = {n: make_check("passed", evidence_obj={"k": n})
          for n in REQUIRED_CHECKS}
checks["temporal"] = make_check("unchecked", note="没数据")
d = export_decision(checks)
check("不是 eligible", d["training_eligible"] is False)
check("temporal 在阻塞里", "temporal" in d["training_blockers"])

print("\n=== 5) 一项 failed -> 阻塞 ===")
checks = {n: make_check("passed", evidence_obj={"k": n})
          for n in REQUIRED_CHECKS}
checks["geometry"] = make_check("failed", evidence_obj={"err": 1.2})
d = export_decision(checks)
check("不是 eligible", d["training_eligible"] is False)
check("geometry 在阻塞里", "geometry" in d["training_blockers"])

print("\n=== 6) build_checks_from_artifacts ===")
c = build_checks_from_artifacts(
    {"camera": "manual_calib",
     "lines": [{"alpha_mean": 0.01, "class": "lane"}],
     "masks": {"paint_kept_mean": 0.08, "blocked_ratio": 0.53,
               "road_mean": 0.14}},
    bg_path="/data/render_hybrid/bgs/cand_04.jpg")
d = export_decision(c)
check("默不 eligible", d["training_eligible"] is False)
check("calibration passed", c["calibration"]["status"] == "passed")
check("occlusion passed", c["occlusion"]["status"] == "passed")
check("original_labels unchecked",
      c["original_labels"]["status"] == "unchecked")
check("appearance unchecked", c["appearance"]["status"] == "unchecked")

print("\n=== 7) 非法 status 报错 ===")
try:
    make_check("maybe")
    check("非法 status 应报错", False)
except ValueError:
    check("非法 status 报错", True)

print("\n=== 8) 无标定 -> calibration unchecked ===")
c = build_checks_from_artifacts(
    {"camera": "auto_estimate", "lines": [], "masks": {}},
    bg_path="/data/render_hybrid/bgs/nonexistent.jpg")
check("calibration unchecked", c["calibration"]["status"] == "unchecked")
check("geometry unchecked", c["geometry"]["status"] == "unchecked")
check("occlusion unchecked", c["occlusion"]["status"] == "unchecked")

print("\n" + "=" * 50)
print("通过 %d / 失败 %d" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
