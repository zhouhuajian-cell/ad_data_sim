"""Apply OBJ/GND routes to the deployed dashboard without replacing TSR changes."""
from pathlib import Path
from datetime import datetime
import sys


def insert_before(source, anchor, code):
    if source.count(anchor) != 1:
        raise RuntimeError("Dashboard anchor is missing or ambiguous: " + anchor)
    return source.replace(anchor, code + anchor, 1)


dashboard = Path(sys.argv[1])
server_file = dashboard / "server.py"
index_file = dashboard / "index.html"
server = original_server = server_file.read_text(encoding="utf-8")
index = original_index = index_file.read_text(encoding="utf-8")

if "def smoke_artifact(" in server:
    raise SystemExit("OBJ/GND routes are already installed")

server = insert_before(server, "def build_cmd(action, p, cfg):", '''
SMOKE_SCRIPT = "/data/avsynth/ObjSmokeTools/run_obj_smoke.sh"
SMOKE_ROOT = "/data/render"
SMOKE_ANNOTATIONS = {"obj": "coco/annotations/instances_all.json",
                     "gnd": "ground_annotations.json"}
SMOKE_LAUNCH_LOCK = threading.Lock()


def smoke_artifact(run, kind):
    match = re.fullmatch(r"(obj|gnd)_smoke\\.[A-Za-z0-9]{6}", run)
    if not match:
        return None
    relative = ("overlay/contact_sheet.jpg" if kind == "sheet"
                else SMOKE_ANNOTATIONS[match.group(1)])
    path = os.path.join(SMOKE_ROOT, run, relative)
    return path if os.path.realpath(path) == path and os.path.isfile(path) else None


def smoke_status(mode):
    with JOBS_LOCK:
        jobs = [dict(j) for j in JOBS.values()]
    latest = max((j for j in jobs if j["action"] == mode + "_smoke"),
                 key=lambda j: j["started_ts"], default=None)
    job = None
    if latest:
        log = tail(latest["log"], 30)
        match = re.search(r"^输出目录 (/data/render/((?:obj|gnd)_smoke\\.[A-Za-z0-9]{6}))$", log, re.M)
        if match and (latest["status"] == "fail" or latest["status"] == "running"):
            ue_log = os.path.join(match.group(1), "ue.log")
            if os.path.realpath(ue_log) == ue_log:
                log += "\\n" + tail(ue_log, 12)
        job = {"status": latest["status"], "log": log}

    completed = []
    for name in os.listdir(SMOKE_ROOT):
        if name.startswith(mode + "_smoke."):
            sheet = smoke_artifact(name, "sheet")
            if sheet and smoke_artifact(name, "annotations"):
                completed.append((os.path.getmtime(sheet), name))
    busy = render_session_running() or any(
        j["status"] == "running" and j["action"] in
        ("obj_smoke", "gnd_smoke", "setup_assets", "preview", "render_start")
        for j in jobs)
    return {"job": job, "busy": busy,
            "result": max(completed, default=(0, None))[1]}


''')

server = insert_before(server, '    if action == "setup_assets":\n', '''
    if action in ("obj_smoke", "gnd_smoke"):
        return ["runuser", "-u", "algo", "--", "bash", SMOKE_SCRIPT, action.split("_")[0]]
''')
server = insert_before(server, '    if action in ("setup_assets", "preview"):\n', '''
    if action in ("obj_smoke", "gnd_smoke"):
        if not os.path.isfile(SMOKE_SCRIPT):
            return "交通场景小样脚本未安装"
        if smoke_status(action.split("_")[0])["busy"]:
            return "UE 正在运行作业，请稍后再试"
''')
server = insert_before(server, '        if u.path in ("/", "/index.html"):\n', '''
        if u.path in ("/obj", "/gnd"):
            with open(os.path.join(BASE, "obj.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        status = re.fullmatch(r"/api/(obj|gnd)/status", u.path)
        if status:
            return self._send(200, smoke_status(status.group(1)))
        result = re.fullmatch(r"/api/(obj|gnd)/result/([^/]+)/(sheet|annotations)", u.path)
        if result:
            mode, run, kind = result.groups()
            artifact = smoke_artifact(run, kind) if run.startswith(mode + "_") else None
            if artifact:
                with open(artifact, "rb") as f:
                    return self._send(200, f.read(), "image/jpeg" if kind == "sheet" else "application/json")
            return self._send(404, {"error": "结果文件不存在"})
''')
server = insert_before(server, '        if u.path == "/api/preview":\n', '''
        smoke = re.fullmatch(r"/api/(obj|gnd)/run", u.path)
        if smoke:
            with SMOKE_LAUNCH_LOCK:
                rec, error = start_job(smoke.group(1) + "_smoke", {})
            if error:
                return self._send(409, {"error": error})
            return self._send(202, {"id": rec["id"]})
''')
index = insert_before(index, "</nav>", '''
<div style="display:grid;gap:8px;padding:12px">
  <a href="/obj" style="color:#76dfdb">OBJ 交通目标</a>
  <a href="/gnd" style="color:#76dfdb">GND 地面标线</a>
</div>''')
compile(server, str(server_file), "exec")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for path, original, modified in ((server_file, original_server, server),
                                  (index_file, original_index, index)):
    path.with_name(path.name + ".bak.smoke." + stamp).write_text(original, encoding="utf-8")
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(modified, encoding="utf-8")
    pending.chmod(path.stat().st_mode & 0o777)
    pending.replace(path)
