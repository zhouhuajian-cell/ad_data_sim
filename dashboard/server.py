# -*- coding: utf-8 -*-
# server.py — UE5 合成流水线可视化操作台（零依赖单文件，服务器系统 python3 直接跑）
# 干什么：网页上做 环境核查/生成任务/启动停止渲染/进度/后处理/日志查看，只暴露白名单动作。
# 怎么跑：python3 server.py [端口]  → 浏览器开 http://<服务器IP>:8010
# 成功判据：/api/status 出 JSON；点"环境核查"出 PASS/FAIL 列表；渲染按钮对应 tmux 会话起停。
# ⚠️ 只调白名单脚本、参数走 argv 不过 shell；不碰 ad_mining 与其 venv。
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")
JOBS_DIR = os.path.join(BASE, "jobs")
DEFAULT_CONFIG = {
    "scripts_dir": "/opt/ue5_synth",
    "queue_file": "/data/queue/tasks.jsonl",
    "done_dir": "/data/queue/done",
    "raw_dir": "/data/render/raw",
    "meta_dir": "/data/render/meta",
    "final_dir": "/data/render/final",
    "dataset_dir": "/data/render/dataset",
    "uproject": "/data/avsynth/avsynth.uproject",
    "ue_dir": "/data/UE_5.3",
    "logs_dir": "/data/logs",
    "queue_cfg_dir": "/opt/ue5_synth/config",
    "project_dir": "/data/avsynth",
    "real_dir": "",
}
# 解释器：用跑本服务的这个 python（Ubuntu 的 python3 / Windows 的 python.exe 都对）
# 环境里取不到 sys.executable 时回退 python3，保持老行为。
PY = sys.executable or "python3"
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_PROCS = {}  # jid -> Popen；Popen 不能塞进 JOBS（/api/job 会 json.dumps rec）

def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            cfg.update(json.load(open(CONFIG_PATH, encoding="utf-8")))
        except Exception:
            pass
    return cfg

def save_config(patch):
    cfg = load_config()
    for k, v in (patch or {}).items():
        if k in DEFAULT_CONFIG and isinstance(v, str):
            cfg[k] = v
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    return cfg

def int_param(p, key, default=None):
    """取整型参数：缺省/空串用 default，非法值抛 ValueError（由 /api/run 转成 400）。"""
    v = p.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError("%s 必须是整数，收到 %r" % (key, v))

def build_cmd(action, p, cfg):
    """白名单动作 → argv 列表（不走 shell，参数原样传递）。非法参数抛 ValueError。"""
    s, py = cfg["scripts_dir"], PY
    if action == "envcheck":
        return ["bash", os.path.join(s, "linux/env_check.sh")]
    if action == "compile":
        return ["bash", os.path.join(s, "linux/compile_ue5.sh"), cfg["ue_dir"]]
    if action == "queue_gen":
        d = p.get("cfg_dir") or cfg["queue_cfg_dir"]
        return [py, os.path.join(s, "common/queue_gen.py"),
                "--scenes", os.path.join(d, "scenes.json"),
                "--lights", os.path.join(d, "lights.json"),
                "--targets", os.path.join(d, "targets.json"),
                "--cameras", os.path.join(d, "cameras.json"),
                "-o", p.get("out") or cfg["queue_file"],
                "--seed", str(int_param(p, "seed", 20261113))]
    if action == "case_check":
        return [py, os.path.join(s, "common/case_check.py"),
                p.get("project_dir") or cfg["project_dir"]]
    if action == "render_start":
        return ["bash", os.path.join(s, "linux/run_batch.sh"),
                cfg["uproject"], p.get("queue") or cfg["queue_file"]]
    if action == "render_stop":
        return ["tmux", "kill-session", "-t", "ue5_render"]
    if action == "gpu_watch":
        return ["bash", os.path.join(s, "linux/gpu_watch.sh"),
                os.path.join(cfg["logs_dir"], "gpu_watch.csv")]
    if action == "post":
        cmd = [py, os.path.join(s, "post/postprocess.py"),
               "--raw", cfg["raw_dir"], "--meta", cfg["meta_dir"],
               "--out", p.get("out") or cfg["final_dir"]]
        if p.get("params"):
            cmd += ["--params", p["params"]]
        if p.get("real") or cfg["real_dir"]:
            cmd += ["--real", p.get("real") or cfg["real_dir"]]
        limit = int_param(p, "limit")
        if limit is not None:
            if limit < 0:
                raise ValueError("limit 不能是负数，收到 %r" % limit)
            cmd += ["--limit", str(limit)]
        return cmd
    if action == "pack":
        fmt = str(p.get("format") or "coco")
        if fmt not in ("coco", "yolo"):
            raise ValueError("format 只能是 coco 或 yolo，收到 %r" % fmt)
        return [py, os.path.join(s, "post/pack_dataset.py"),
                "--meta", cfg["meta_dir"], "--images", cfg["final_dir"],
                "--out", p.get("out") or cfg["dataset_dir"],
                "--format", fmt, "--seed", str(int_param(p, "seed", 20261113))]
    if action == "dryrun":
        # 链路自检：工具本体的参数契约由它自己管，这里只保证解释器与脚本路径
        return [py, os.path.join(s, "tools/dryrun_check.py")]
    return None

def start_job(action, params):
    cfg = load_config()
    try:
        cmd = build_cmd(action, params or {}, cfg)
    except ValueError as e:
        return None, "参数错误: %s" % e
    if not cmd:
        return None, "未知动作: %s" % action
    with JOBS_LOCK:
        for j in JOBS.values():
            if j["action"] == action and j["status"] == "running":
                return None, "已有同类作业在跑（%s），等它结束或先杀掉它" % j["id"]
    jid = uuid.uuid4().hex[:12]
    os.makedirs(JOBS_DIR, exist_ok=True)
    logp = os.path.join(JOBS_DIR, "%s_%s.log" % (action, jid))
    logf = None
    try:
        logf = open(logp, "w", encoding="utf-8", errors="replace")
        logf.write("$ %s\n[启动 %s]\n" % (" ".join(cmd), time.strftime("%F %T")))
        logf.flush()
        # Linux 生产：独立进程组，kill 时整组收掉（bash/tmux 子树）；Windows 无 setsid 概念
        kw = {"cwd": BASE, "env": dict(os.environ, PYTHONIOENCODING="utf-8")}
        if os.name == "posix":
            kw["start_new_session"] = True
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, **kw)
    except Exception as e:
        if logf:
            logf.close()
        return None, "启动失败: %s" % e
    rec = {"id": jid, "action": action, "cmd": cmd, "log": logp,
           "started": time.strftime("%F %T"), "pid": proc.pid,
           "status": "running", "ret": None, "ended": ""}
    with JOBS_LOCK:
        JOBS[jid] = rec
        JOB_PROCS[jid] = proc  # Popen 单独存：rec 要能 json.dumps
    def monitor():
        ret = proc.wait()
        logf.close()  # 子进程结束关日志句柄，避免长跑服务句柄泄漏
        with JOBS_LOCK:
            rec["status"] = "ok" if ret == 0 else "fail"
            rec["ret"] = ret
            rec["ended"] = time.strftime("%F %T")
            JOB_PROCS.pop(jid, None)
    threading.Thread(target=monitor, daemon=True).start()
    return rec, None

def kill_job(jid):
    with JOBS_LOCK:
        rec = JOBS.get(jid)
        proc = JOB_PROCS.get(jid)
    if not rec or rec["status"] != "running":
        return False
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(rec["pid"]), 9)  # Linux 生产：杀整个进程组
        elif proc is not None:
            proc.terminate()  # Windows 没有 killpg：先温和结束
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()  # 3 秒还不退再强杀
        else:
            return False
        return True
    except Exception:
        return False

def tail(path, lines=200, max_bytes=512 * 1024):
    try:
        if not os.path.exists(path):
            return "(文件不存在: %s)" % os.path.basename(path)
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                blob = f.read().split(b"\n", 1)[-1]
            else:
                blob = f.read()
        txt = blob.decode("utf-8", "replace")
        return "\n".join(txt.splitlines()[-lines:])
    except Exception as e:
        return "(读取失败: %s)" % e

def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def probe_status():
    cfg = load_config()
    st = {"cfg": cfg, "time": time.strftime("%F %T")}
    try:
        mi = {}
        for ln in open("/proc/meminfo"):
            k, _, v = ln.partition(":")
            mi[k] = v.strip()
        st["mem_total_g"] = round(int(mi["MemTotal"].split()[0]) / 1048576, 1)
        st["mem_avail_g"] = round(int(mi["MemAvailable"].split()[0]) / 1048576, 1)
        sw = open("/proc/swaps").read().strip().splitlines()
        st["swap_on"] = len(sw) > 1
    except Exception:
        st["mem_total_g"] = st["mem_avail_g"] = -1
        st["swap_on"] = None
    for key, path in (("disk_root", "/"), ("disk_data", "/data")):
        try:
            st[key + "_free_g"] = round(shutil.disk_usage(path).free / 1073741824, 1)
        except Exception:
            st[key + "_free_g"] = -1
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0]
        # 型号名里可能带逗号（如 "NVIDIA A100, 80GB"），从右往左切 3 刀拿数值字段
        name, m, u, t = [x.strip() for x in out.rsplit(",", 3)]
        st["gpu"] = {"name": name or "未知型号", "mem_mb": int(m),
                     "util": int(u), "temp": int(t)}
    except Exception:
        st["gpu"] = None
    try:
        st["render_running"] = subprocess.run(
            ["tmux", "has-session", "-t", "ue5_render"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        st["render_running"] = False
    st["tasks_total"] = count_lines(cfg["queue_file"])
    try:
        st["tasks_done"] = len(os.listdir(cfg["done_dir"]))
    except Exception:
        st["tasks_done"] = 0
    with JOBS_LOCK:
        st["jobs"] = [{k: j[k] for k in ("id", "action", "status", "ret", "started", "ended")}
                      for j in sorted(JOBS.values(), key=lambda x: x["started"], reverse=True)[:12]]
    return st

try:
    HTML_PAGE = open(os.path.join(BASE, "index.html"), encoding="utf-8").read()
except Exception:
    HTML_PAGE = "<h1>index.html 缺失</h1>"

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, obj, ctype="application/json; charset=utf-8"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/status":
            return self._send(200, probe_status())
        if u.path == "/api/job":
            jid = (q.get("id") or [""])[0]
            with JOBS_LOCK:
                rec = JOBS.get(jid)
            if not rec:
                return self._send(404, {"error": "no job"})
            return self._send(200, dict(rec, out=tail(rec["log"], 300)))
        if u.path == "/api/log":
            name = (q.get("name") or ["batch.log"])[0]
            if "/" in name or "\\" in name or ".." in name:
                return self._send(400, {"error": "bad name"})
            try:
                lines = int((q.get("lines") or ["200"])[0])
            except (TypeError, ValueError):
                lines = 200
            return self._send(200, {"name": name,
                                    "out": tail(os.path.join(load_config()["logs_dir"], name),
                                                min(lines, 1000))})
        return self._send(404, {"error": "not found"})
    def do_POST(self):
        u = urlparse(self.path)
        p = self._body()
        if u.path == "/api/run":
            # 前端发的是 {action, params:{...}}；也兼容平铺的 {action, limit, ...} 老调用
            params = p.get("params") if isinstance(p, dict) else None
            if not isinstance(params, dict):
                params = p if isinstance(p, dict) else {}
            rec, err = start_job(p.get("action", ""), params)
            if err:
                return self._send(400, {"error": err})
            return self._send(200, {"id": rec["id"]})
        if u.path == "/api/kill":
            return self._send(200, {"killed": kill_job(p.get("id", ""))})
        if u.path == "/api/config":
            return self._send(200, save_config(p))
        return self._send(404, {"error": "not found"})

def main():
    port = int(os.environ.get("UE5_PANEL_PORT") or (sys_argv_port() or 8010))
    os.makedirs(JOBS_DIR, exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print("[panel] http://0.0.0.0:%d  (Ctrl-C 停)" % port)
    srv.serve_forever()

def sys_argv_port():
    return sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else None

if __name__ == "__main__":
    main()
