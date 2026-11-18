"""Enable the workzone preset on a dashboard with BEV installed."""
from datetime import datetime
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "server.py"
original = path.read_text(encoding="utf-8")
updated = original
for old, new in (
    ('(mode != "obj" or sample != "truck")', '(mode != "obj" or sample not in ("truck", "workzone"))'),
    ('    return {"job": job, "busy": busy, "result": result,\n            "result_sample": "truck" if is_truck else "traffic"}',
     '''    is_workzone = result and mode == "obj" and smoke_artifact(result, "scene")
    return {"job": job, "busy": busy, "result": result,
            "result_sample": "workzone" if is_workzone else ("truck" if is_truck else "traffic")}'''),
    ('    path = os.path.join(SMOKE_ROOT, run, relative)',
     '''    if kind == "scene":
        if match.group(1) != "obj":
            return None
        relative = "scene_annotations.json"
    path = os.path.join(SMOKE_ROOT, run, relative)'''),
    ('([^/]+)/(sheet|annotations)', '([^/]+)/(sheet|annotations|scene)'),
):
    if updated.count(old) != 1:
        raise RuntimeError("Missing or duplicate workzone deployment anchor: " + old)
    updated = updated.replace(old, new, 1)

compile(updated, str(path), "exec")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path.with_name(path.name + ".bak.workzone." + stamp).write_text(original, encoding="utf-8")
pending = path.with_suffix(".py.workzone-tmp")
pending.write_text(updated, encoding="utf-8")
pending.chmod(path.stat().st_mode & 0o777)
pending.replace(path)
