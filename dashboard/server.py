# -*- coding: utf-8 -*-
# server.py — UE5 合成流水线可视化操作台（零依赖单文件，服务器系统 python3 直接跑）
# 干什么：网页上做 环境核查/生成任务/启动停止渲染/进度/后处理/日志查看，只暴露白名单动作。
# 怎么跑：python3 server.py [端口]  → 浏览器开 http://<服务器IP>:8010
# 成功判据：/api/status 出 JSON；点"环境核查"出 PASS/FAIL 列表；渲染按钮对应 tmux 会话起停。
# ⚠️ 只调白名单脚本、参数走 argv 不过 shell；不碰 ad_mining 与其 venv。
# ⚠️ 后台自动化：显卡记录随面板常驻；渲染结束（或停止）后自动处理+打包+抽检，
#    失败重试与跳过已完成都在这层做，界面上不再提供对应按钮（见 AUTO_* 常量）。
import json
import csv
import datetime
import re
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import quote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

try:                      # 单实例锁只在 POSIX 上有；Windows 本地调试时跳过（不拦启动）
    import fcntl
except ImportError:       # pragma: no cover
    fcntl = None

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")
JOBS_DIR = os.path.join(BASE, "jobs")
LOG_DIR = os.path.join(BASE, "logs")          # 后台自动化自己的日志（显卡记录、自动处理序列）
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
    "overlay_dir": "/data/render/overlay",
    # 方案模块 5/6 的产物位置：VOC XML 目录与难例清单（pack 会自动读取清单做过滤）
    "voc_dir": "/data/render/dataset/voc",
    "hardcase_csv": "/data/render/dataset/hardcase_list.csv",
    # 渲染参数（界面上只暴露这两个；其余用脚本默认值，改这里即可）
    "render_queue": "/data/queue/tasks.jsonl",
    "render_resolution": "1920x1080",
}
# 解释器：用跑本服务的这个 python（Ubuntu 的 python3 / Windows 的 python.exe 都对）
# 环境里取不到 sys.executable 时回退 python3，保持老行为。
PY = sys.executable or "python3"
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_PROCS = {}  # jid -> Popen；Popen 不能塞进 JOBS（/api/job 会 json.dumps rec）

# ---- 后台自动化常量 ----
AUTO_POST = os.environ.get("UE5_AUTO_POST", "1") == "1"          # 渲染结束后自动处理+打包+抽检
AUTO_GPU_WATCH = os.environ.get("UE5_AUTO_GPU", "1") == "1"      # 常驻显卡记录
AUTO_POLL_SEC = float(os.environ.get("UE5_AUTO_POLL", "15"))     # 自动化轮询间隔（秒）
AUTO_POST_IDLE_SEC = float(os.environ.get("UE5_POST_IDLE", "60"))  # 渲染停止后等多久再自动跑（防抖）
AUTO_POST_RETRY_SEC = 900.0                                       # 自动处理失败后多久重试
GPU_WATCH_PROCS = {}   # per-action 单例标记（见 _start_action_proc）

def now_str():
    return time.strftime("%F %T")

def log_auto(msg):
    """后台自动化日志：面板控制台 + dashboard/logs/auto.log（保留最近 2000 行）。"""
    line = "[%s] %s" % (now_str(), msg)
    print("[auto] " + msg)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        p = os.path.join(LOG_DIR, "auto.log")
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        # 简单截断：超过 1MB 就只留后 2000 行
        if os.path.getsize(p) > 1024 * 1024:
            with open(p, encoding="utf-8", errors="replace") as f:
                keep = f.read().splitlines()[-2000:]
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(keep) + "\n")
    except Exception:
        pass

def normalize_net_path(p):
    """把 Windows 网盘路径 / UNC 路径 / 各种斜杠统一转换成 Linux 挂载路径。

    支持：
      \\192.168.20.206\Data_Platform\zhj_datamining\...   （UNC）
      //192.168.20.206/Data_Platform/zhj_datamining/...
      Z:\Data_Platform\zhj_datamining\...
      /mnt/Data_Platform/zhj_datamining/...               （已是 Linux，原样返回）

    服务器上 CIFS 挂载点约定为 /mnt/<共享名>。
    例如 共享 Data_Platform → /mnt/Data_Platform
    """
    if not p:
        return p
    raw = str(p).strip().strip('"').strip("'")

    # 已是合法 Linux 路径
    if raw.startswith("/") and not raw.startswith("//"):
        return raw.replace("\\", "/")

    # UNC: \\host\share\rest  或  //host/share/rest
    m = re.match(r"^[\\/]{2}([^\\/]+)[\\/]([^\\/]+)(?:[\\/](.*))?$", raw)
    if m:
        share = m.group(2)
        rest = (m.group(3) or "").replace("\\", "/")
        base = "/mnt/" + share
        return os.path.normpath(base + ("/" + rest if rest else ""))

    # 盘符: Z:\share\rest
    m2 = re.match(r"^([A-Za-z]):[\\/]([^\\/]+)(?:[\\/](.*))?$", raw)
    if m2:
        share = m2.group(2)
        rest = (m2.group(3) or "").replace("\\", "/")
        base = "/mnt/" + share
        return os.path.normpath(base + ("/" + rest if rest else ""))

    # 相对路径 → 转成 / 开头
    if not raw.startswith("/"):
        raw = "/" + raw.replace("\\", "/").replace(":", "").lstrip("/")
    return os.path.normpath(raw)


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            # utf-8-sig：Windows 上用 PowerShell/记事本写的配置带 BOM 也能读（否则会静默回退默认）
            with open(CONFIG_PATH, encoding="utf-8-sig") as f:
                cfg.update(json.load(f))
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

def read_metrics(minutes=10):
    """Read the actual GPU watcher CSV. Missing measurements stay absent."""
    cfg = load_config()
    result = {"gpu_mem": [], "mem_used": [], "gpu_util": []}
    path = os.path.join(cfg["logs_dir"], "gpu_watch.csv")
    cutoff = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ts = datetime.datetime.strptime(row["ts"], "%Y-%m-%d_%H:%M:%S")
                    if ts < cutoff:
                        continue
                    result["gpu_mem"].append([row["ts"], round(float(row["gpu_mem_mb"]) / 1024, 3)])
                    result["gpu_util"].append([row["ts"], float(row["gpu_util_pct"])])
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return result

def read_dataset_stats():
    cfg = load_config()
    result = {"total_frames": 0, "by_class": {}, "hardcase_count": 0,
              "hardcase_by_rule": {}, "datasets": []}
    path = os.path.join(cfg["dataset_dir"], "instances_all.json")
    if not os.path.isfile(path):
        path = os.path.join(cfg["dataset_dir"], "annotations", "instances_all.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        result["total_frames"] = len(data.get("images", []))
        classes = {c["id"]: c["name"] for c in data.get("categories", [])}
        for ann in data.get("annotations", []):
            name = classes.get(ann.get("category_id"), "unknown")
            result["by_class"][name] = result["by_class"].get(name, 0) + 1
        result["datasets"].append({"name": os.path.basename(cfg["dataset_dir"]),
                                   "frames": result["total_frames"], "path": path})
    except (OSError, ValueError, TypeError, KeyError):
        pass
    try:
        with open(cfg["hardcase_csv"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        result["hardcase_count"] = len(set(r.get("frame_id") for r in rows if r.get("frame_id")))
        for row in rows:
            for rule in (row.get("rules") or "").split(";"):
                if rule:
                    result["hardcase_by_rule"][rule] = result["hardcase_by_rule"].get(rule, 0) + 1
    except OSError:
        pass
    return result

def image_inbox():
    return os.path.join(load_config()["project_dir"], "AssetInbox")

def list_images():
    folder = image_inbox()
    if not os.path.isdir(folder):
        return []
    return [{"name": n, "bytes": os.path.getsize(os.path.join(folder, n))}
            for n in sorted(os.listdir(folder))
            if re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:png|jpg|jpeg)", n, re.I)
            and os.path.isfile(os.path.join(folder, n))]

def int_param(p, key, default=None):
    """取整型参数：缺省/空串用 default，非法值抛 ValueError（由 /api/run 转成 400）。"""
    v = p.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError("%s 必须是整数，收到 %r" % (key, v))


# ===========================================================================
# [HYBRID-SQLITE] HybridPipeline 任务状态（SQLite）
# ===========================================================================
HYBRID_DB = "/data/avsynth/state/jobs.db"
HYBRID_ROOT = "/data/render_hybrid"
HYBRID_SCRIPT_DIR = "/data/avsynth/HybridPipeline/scripts"


def _hybrid_conn():
    import sqlite3
    if not os.path.isfile(HYBRID_DB):
        return None
    c = sqlite3.connect("file:%s?mode=ro" % HYBRID_DB, uri=True, timeout=10.0)
    c.row_factory = sqlite3.Row
    return c


def hybrid_jobs(limit=50, offset=0):
    c = _hybrid_conn()
    if c is None:
        return {"jobs": [], "total": 0, "db": HYBRID_DB, "exists": False}
    try:
        total = c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        rows = c.execute("""
            SELECT job_id, module, pipeline_ver, status, total_frames,
                   passed_frames, failed_frames, output_dir,
                   created_at, updated_at, error_msg
            FROM jobs ORDER BY updated_at DESC LIMIT ? OFFSET ?
        """, (int(limit), int(offset))).fetchall()
        return {"jobs": [dict(r) for r in rows], "total": total,
                "db": HYBRID_DB, "exists": True}
    except Exception as e:
        return {"jobs": [], "total": 0, "error": str(e)[:200], "exists": True}
    finally:
        c.close()


def hybrid_job(job_id):
    c = _hybrid_conn()
    if c is None:
        return {"error": "数据库不存在", "db": HYBRID_DB}
    try:
        r = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not r:
            return {"error": "任务不存在"}
        d = dict(r)
        try:
            d["config"] = __import__("json").loads(d.pop("config_json") or "{}")
        except Exception:
            d["config"] = {}
        cnt = c.execute("""SELECT status, COUNT(*) n FROM frames
                           WHERE job_id=? GROUP BY status""",
                        (job_id,)).fetchall()
        d["frame_status"] = {x["status"]: x["n"] for x in cnt}
        return d
    finally:
        c.close()


def hybrid_frames(job_id, status="", limit=100, offset=0):
    c = _hybrid_conn()
    if c is None:
        return {"frames": [], "total": 0}
    try:
        if status:
            total = c.execute("""SELECT COUNT(*) FROM frames
                                 WHERE job_id=? AND status=?""",
                              (job_id, status)).fetchone()[0]
            rows = c.execute("""
                SELECT frame_id, layout, status, n_objects, max_occl,
                       composed_path, mask_path, error_code, updated_at
                FROM frames WHERE job_id=? AND status=?
                ORDER BY frame_id LIMIT ? OFFSET ?
            """, (job_id, status, int(limit), int(offset))).fetchall()
        else:
            total = c.execute("SELECT COUNT(*) FROM frames WHERE job_id=?",
                              (job_id,)).fetchone()[0]
            rows = c.execute("""
                SELECT frame_id, layout, status, n_objects, max_occl,
                       composed_path, mask_path, error_code, updated_at
                FROM frames WHERE job_id=?
                ORDER BY frame_id LIMIT ? OFFSET ?
            """, (job_id, int(limit), int(offset))).fetchall()
        return {"frames": [dict(r) for r in rows], "total": total}
    finally:
        c.close()


def hybrid_frame(job_id, frame_id):
    c = _hybrid_conn()
    if c is None:
        return {"error": "数据库不存在"}
    try:
        r = c.execute("""SELECT * FROM frames WHERE job_id=? AND frame_id=?""",
                      (job_id, frame_id)).fetchone()
        if not r:
            return {"error": "帧不存在"}
        d = dict(r)
        try:
            d["meta"] = __import__("json").loads(d.pop("meta_json") or "{}")
        except Exception:
            d["meta"] = {}
        return d
    finally:
        c.close()


def hybrid_artifact(path):
    """安全地读 /data/render_hybrid 下的文件。"""
    p = os.path.realpath(path)
    root = os.path.realpath(HYBRID_ROOT)
    if not p.startswith(root + os.sep):
        return None, "forbidden"
    if not os.path.isfile(p):
        return None, "not found"
    with open(p, "rb") as f:
        return f.read(), os.path.splitext(p)[1].lower()


def hybrid_submit(frames, bg, out_name="", module="obj", layout=""):
    """提交一个后台批量任务（nohup 起子进程）。"""
    import subprocess
    import time as _t
    frames = int(frames)
    if frames <= 0 or frames > 5000:
        return {"error": "frames 需在 1~5000"}
    if not os.path.isfile(bg):
        return {"error": "底图不存在: %s" % bg}
    job = out_name.strip() or ("batch_%s" % _t.strftime("%Y%m%d_%H%M%S"))
    out_dir = os.path.join(HYBRID_ROOT, job)
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "worker.log")
    lay = " --layout %s" % layout if layout else ""
    cmd = [
        "su", "-", "algo", "-c",
        "cd %s && /usr/bin/python3 obj_batch_worker.py run "
        "--frames %d --out %s --bg %s%s" % (HYBRID_SCRIPT_DIR, frames,
                                            out_dir, bg, lay),
    ]
    with open(log, "ab") as lf:
        try:
            subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
        except Exception as e:
            return {"error": "启动失败: %s" % e}
    return {"ok": True, "job": job, "out": out_dir, "frames": frames,
            "log": log}

SMOKE_SCRIPT = "/data/avsynth/ObjSmokeTools/run_obj_smoke.sh"
SMOKE_ROOT = "/data/render"
SMOKE_ANNOTATIONS = {"obj": "coco/annotations/instances_all.json",
                     "gnd": "ground_annotations.json"}
SMOKE_LAUNCH_LOCK = threading.Lock()


def smoke_artifact(run, kind):
    match = re.fullmatch(r"(obj|gnd)_smoke\.[A-Za-z0-9]{6}", run)
    if not match:
        return None
    relative = ("overlay/contact_sheet.jpg" if kind == "sheet"
                else SMOKE_ANNOTATIONS[match.group(1)])
    if kind == "scene":
        if match.group(1) != "obj":
            return None
        relative = "scene_annotations.json"
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
        match = re.search(r"^输出目录 (/data/render/((?:obj|gnd)_smoke\.[A-Za-z0-9]{6}))$", log, re.M)
        if match and (latest["status"] == "fail" or latest["status"] == "running"):
            ue_log = os.path.join(match.group(1), "ue.log")
            if os.path.realpath(ue_log) == ue_log:
                log += "\n" + tail(ue_log, 12)
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
    result = max(completed, default=(0, None))[1]
    is_truck = result and mode == "obj" and os.path.isfile(
        os.path.join(SMOKE_ROOT, result, "obj_truck_0001.json"))
    is_workzone = result and mode == "obj" and smoke_artifact(result, "scene")
    return {"job": job, "busy": busy, "result": result,
            "result_sample": "workzone" if is_workzone else ("truck" if is_truck else "traffic")}


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

    if action in ("obj_smoke", "gnd_smoke"):
        mode = action.split("_")[0]
        sample = p.get("sample", "traffic")
        if sample != "traffic" and (mode != "obj" or sample not in ("truck", "workzone")):
            raise ValueError("不支持的小样类型")
        return ["runuser", "-u", "algo", "--", "bash", SMOKE_SCRIPT, mode, sample]
    if action == "setup_assets":
        return ["bash", os.path.join(cfg["scripts_dir"], "linux", "run_setup.sh"),
                cfg["uproject"], cfg["ue_dir"]]
    if action == "preview":
        return ["bash", os.path.join(cfg["scripts_dir"], "linux", "run_setup.sh"),
                cfg["uproject"], cfg["ue_dir"], "preview_scene.py"]
    if action == "case_check":
        return [py, os.path.join(s, "common/case_check.py"),
                p.get("project_dir") or cfg["project_dir"]]
    if action == "render_start":
        return ["bash", os.path.join(s, "linux/run_batch.sh"),
                cfg["uproject"], p.get("queue") or cfg["queue_file"], cfg["ue_dir"]]
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
    if action == "voc":
        # 方案模块 5：导出 PASCAL VOC XML（与图像同名，额外产出 ImageSets）
        return [py, os.path.join(s, "post/voc_export.py"),
                "--meta", cfg["meta_dir"], "--images", cfg["final_dir"],
                "--out", p.get("out") or cfg.get("voc_dir") or os.path.join(cfg["dataset_dir"], "voc")]
    if action == "hardcase":
        # 方案模块 5+6：难例判定与清单（pack 会读取它做过滤）
        return [py, os.path.join(s, "post/hardcase_export.py"),
                "--meta", cfg["meta_dir"], "--images", cfg["final_dir"],
                "--out", p.get("out") or cfg.get("hardcase_csv")
                or os.path.join(cfg["dataset_dir"], "hardcase_list.csv")]
    if action == "pack":
        fmt = str(p.get("format") or "coco")
        if fmt not in ("coco", "yolo"):
            raise ValueError("format 只能是 coco 或 yolo，收到 %r" % fmt)
        cmd = [py, os.path.join(s, "post/pack_dataset.py"),
               "--meta", cfg["meta_dir"], "--images", cfg["final_dir"],
               "--out", p.get("out") or cfg["dataset_dir"],
               "--format", fmt, "--seed", str(int_param(p, "seed", 20261113))]
        # 难例清单存在时自动纳入打包（ratio 默认 1.0 = 全保留，难例本身就是要补的数据）
        hc_csv = cfg.get("hardcase_csv") or os.path.join(cfg["dataset_dir"], "hardcase_list.csv")
        if os.path.exists(hc_csv):
            cmd += ["--hardcase-csv", hc_csv,
                    "--hardcase-ratio", str(float(p.get("hardcase_ratio", 1.0)))]
        return cmd
    # [TSR-API] tsr_batch 动作：跑 tools/batch_replace.py
    if action == "tsr_batch":
        cmd = p.get("cmd") or []
        if not cmd:
            raise ValueError("tsr_batch 缺少 cmd")
        return [ALGO_PYTHON] + [str(c) for c in cmd] if False else cmd

    if action == "dryrun":
        # 链路自检：工具本体的参数契约由它自己管，这里只保证解释器与脚本路径
        return [py, os.path.join(s, "tools/dryrun_check.py")]
    if action == "overlay":
        # 真值抽检图：把 bbox 画到成品图上，供人工目检（表 11 抽检项）
        return [py, os.path.join(s, "tools/overlay_boxes.py"),
                "--meta", cfg["meta_dir"], "--images", cfg["final_dir"],
                "--out", cfg["overlay_dir"],
                "--limit", str(int_param(p, "limit", 20)),
                "--cols", "4"]
    return None

def precheck(action, cfg):
    """动作前置检查：启动作业前把"缺什么、怎么办"直接说清楚。

    为什么要这层：脚本自己报错时人已经点完按钮了，看到的是 tmux/bash 的英文报错；
    无人值守场景下，缺工程/缺引擎是最常见的"看起来点得动其实跑不了"。
    """

    if action in ("obj_smoke", "gnd_smoke"):
        if not os.path.isfile(SMOKE_SCRIPT):
            return "交通场景小样脚本未安装"
        if smoke_status(action.split("_")[0])["busy"]:
            return "UE 正在运行作业，请稍后再试"
    if action in ("setup_assets", "preview"):
        if render_session_running():
            return "渲染正在运行，先停止渲染再制作验证资产"
        if not os.path.isfile(os.path.join(cfg["project_dir"], "Content", "Python", "setup_assets.py")):
            return "工程缺少 setup_assets.py，先同步工程脚本"
        if not os.path.isfile(cfg["uproject"]):
            return "UE 工程文件不存在"
    if action == "render_start":
        up = cfg["uproject"]
        if not os.path.exists(up):
            return ("还没准备好：服务器上找不到工程文件 %s。\n"
                    "先在 Windows 上运行 windows/sync_project.ps1 把工程下发到服务器；"
                    "工程在别的目录时，去「高级操作 → 配置」把 uproject / project_dir 改成实际路径。" % up)
        cap = os.path.join(cfg["project_dir"], "Content", "Python", "batch_capture.py")
        if not os.path.exists(cap):
            return ("还没准备好：工程里没有批量捕获脚本 %s。\n"
                    "把仓库 engine/Content/Python/ 下的 batch_capture.py 和 capture_math.py "
                    "复制到 <Windows 工程>/Content/Python/，再重新下发工程。" % cap)
        ue_cmd = os.path.join(cfg["ue_dir"], "Engine", "Binaries", "Linux", "UnrealEditor-Cmd")
        if not os.path.exists(ue_cmd):
            return ("还没准备好：引擎还没编译好（找不到 %s）。\n"
                    "先在「高级操作 → 编译引擎」跑一次（1~3 小时、内存吃满，选空闲窗口）；"
                    "引擎装在别处就去配置里改 ue_dir。" % ue_cmd)
        if not os.path.exists(cfg["queue_file"]):
            return ("还没准备好：任务清单 %s 不存在。\n先在第 2 步「生成拍摄清单」。" % cfg["queue_file"])
    if action == "post" and not os.path.isdir(cfg["raw_dir"]):
        return ("还没准备好：原始图目录 %s 不存在。\n先在第 3 步出图，出图会自动建这个目录。" % cfg["raw_dir"])
    if action == "pack" and not os.path.isdir(cfg["meta_dir"]):
        return ("还没准备好：真值目录 %s 不存在。\n先在第 3 步出图，真值 JSON 会写到这个目录。" % cfg["meta_dir"])
    if action == "overlay" and count_dir(cfg["final_dir"], ".png", cap=1) == 0:
        return ("还没准备好：还没有处理后的图。\n先在第 4 步「开始处理」，再生成抽检图。")
    return None

# [TSR-API] TSR 仿真相关常量与辅助
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
TOOLS_DIR = os.path.abspath(TOOLS_DIR)
DEFAULT_TSR_LIBRARY = "/data/tsr_library_v2"   # 官方矢量图库渲染的高清素材
TSR_OUT_DIR = "/data/render/tsr_v3_10"   # TSR-V2-DIR
# 像素/AI 相关脚本必须用 ad_mining 的 venv（有 ultralytics/opencv），面板本体零依赖
ALGO_PYTHON = "/opt/ad_mining/venv/bin/python"


def _tsr_slug(s):
    """把国家码/文件名变成安全的路径片段，防目录穿越。"""
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(s)).strip("_")[:96]


def _run_algo(args, timeout=300):
    """在 ad_mining 的 venv 里跑一个 tools 脚本，返回它 stdout 的 JSON。"""
    import subprocess
    exe = ALGO_PYTHON if os.path.isfile(ALGO_PYTHON) else "python3"
    try:
        p = subprocess.run([exe] + [str(a) for a in args],
                           capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "").strip()
        if not out:
            return {"ok": False, "error": (p.stderr or "脚本无输出")[:300]}
        return json.loads(out.splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


def start_job(action, params, auto=False):
    """启动作业。auto=True 表示由后台自动化触发（不参与"同类作业去重"之外的界面提示）。"""
    cfg = load_config()
    try:
        cmd = build_cmd(action, params or {}, cfg)
    except ValueError as e:
        return None, "参数错误: %s" % e
    if not cmd:
        return None, "未知动作: %s" % action
    err = precheck(action, cfg)
    if err:
        return None, err
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
           "started": time.strftime("%F %T"), "started_ts": time.time(), "pid": proc.pid,
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

def render_session_running():
    try:
        return subprocess.run(
            ["tmux", "has-session", "-t", "ue5_render"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False

def action_running(action):
    with JOBS_LOCK:
        return any(j["action"] == action and j["status"] == "running" for j in JOBS.values())

def last_result(action):
    """最近一次该动作的作业记录（自动化据此判断"上次失败了没 / 结果还新不新"）。

    ⚠️ 不能按 id 排序：id 是随机 12 位十六进制，字典序与时间无关（实测踩到——一直拿到旧的失败记录）。"""
    with JOBS_LOCK:
        cand = [j for j in JOBS.values() if j["action"] == action]
    if not cand:
        return None
    return max(cand, key=lambda x: x.get("started_ts") or 0.0)

# ---- 后台自动化 1：显卡记录常驻 ------------------------------------------------------
GPU_WATCH_PROC = {"proc": None}

class _ExistingProc(object):
    """已在运行、但不由本进程 Popen 出来的子进程：只用于 poll() 判断"还活着吗"。"""
    def __init__(self, pid):
        self.pid = pid
    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except OSError:
            return 1

def ensure_gpu_watch():
    """显卡记录随面板常驻（每秒一行 CSV）。只留一个实例，面板重启后自动补起。

    ⚠️ 只靠内存里的 Popen 判断不够：面板每次重启都会新起一个，旧面板留下的不会退出（实测累积了 6 个
    同时写同一个 CSV）。这里先按进程名查全系统实例，已有就复用，超过了就只留最早的一个。"""
    if not AUTO_GPU_WATCH:
        return
    p = GPU_WATCH_PROC.get("proc")
    if p is not None and p.poll() is None:
        return
    if action_running("gpu_watch"):
        return
    code, out = 0, ""
    try:
        r = subprocess.run(["pgrep", "-f", "gpu_watch.sh"], capture_output=True, text=True, timeout=5)
        code, out = r.returncode, r.stdout
    except Exception:
        code = 1
    pids = [int(x) for x in out.split() if x.strip().isdigit() and int(x) != os.getpid()]
    if pids:
        # 只保留最早那个（此时 pids 已按 pgrep 输出顺序≈pid 顺序），其余清掉，避免多写同一 CSV
        keep = min(pids)
        for pid in pids:
            if pid != keep:
                try:
                    os.kill(pid, 15)
                    log_auto("清理重复的显卡记录进程 pid=%d（保留 %d）" % (pid, keep))
                except Exception:
                    pass
        log_auto("显卡记录已在运行（pid=%d），复用" % keep)
        GPU_WATCH_PROC["proc"] = _ExistingProc(keep)   # 复用分支也要给状态一个可判断的对象
        return
    cfg = load_config()
    csv = os.path.join(cfg["logs_dir"], "gpu_watch.csv")
    cmd = ["bash", os.path.join(cfg["scripts_dir"], "linux/gpu_watch.sh"), csv]
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        logf = open(os.path.join(LOG_DIR, "gpu_watch.out"), "a", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=BASE)
        GPU_WATCH_PROC["proc"] = proc
        log_auto("显卡记录已启动（pid=%d → %s）" % (proc.pid, csv))
    except Exception as e:
        log_auto("显卡记录启动失败：%s（这台机器可能没有 bash/gpu_watch.sh）" % e)

# ---- 后台自动化 2：渲染结束 → 处理 → 打包 → 抽检 --------------------------------------
AUTO_STATE = {"phase": "idle", "render_seen": False, "idle_since": 0.0,
              "last_try": 0.0, "last_done": -1}

def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

def _ended_ts(rec):
    try:
        return time.mktime(time.strptime(rec["ended"], "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return 0.0

def _dispatch(action, params, label):
    rec, err = start_job(action, params, auto=True)
    if err:
        log_auto("[%s] 跳过：%s" % (label, err.replace("\n", " ")))
        return False
    log_auto("[%s] 已启动（job=%s）" % (label, rec["id"]))
    return True

def auto_pipeline_tick():
    """渲染结束后自动跑后处理链；每一步都可重入（脚本自身会跳过已完成的帧）。

    判定顺序（每一步都要有日志，出问题能直接看出卡在哪个条件）：
      有帧可处理 → 处理 → 打包 → 抽检 → done；任一步失败按退避重试。
    """
    if not AUTO_POST:
        return
    cfg = load_config()
    if render_session_running():
        AUTO_STATE["render_seen"] = True
        AUTO_STATE["idle_since"] = 0.0
        AUTO_STATE["phase"] = "rendering"
        return
    done = count_dir(cfg["done_dir"], ".done")
    fin = count_dir(cfg["final_dir"], ".png")
    if done <= 0:
        AUTO_STATE["phase"] = "idle"
        return
    # 渲染刚停：等一小段防抖，避免最后几帧还在落盘时就开跑
    if AUTO_STATE["idle_since"] == 0.0:
        AUTO_STATE["idle_since"] = time.time()
        return
    if time.time() - AUTO_STATE["idle_since"] < AUTO_POST_IDLE_SEC:
        AUTO_STATE["phase"] = "waiting"
        return
    for action in ("post", "voc", "hardcase", "pack", "overlay"):
        if action_running(action):
            AUTO_STATE["phase"] = "running:" + action
            return
    lp = last_result("post")
    post_ok = bool(lp and lp["status"] == "ok")
    lp_end = (_ended_ts(lp) if post_ok else 0.0) or AUTO_STATE["idle_since"]
    sheet = os.path.join(cfg["overlay_dir"], "contact_sheet.jpg")
    need_post = (not post_ok) or fin < done
    # 方案模块 5：VOC XML 与难例清单也要跟着真值更新（缺产物或比 post 旧就重跑）
    voc_ann = os.path.join(cfg.get("voc_dir") or os.path.join(cfg["dataset_dir"], "voc"), "Annotations")
    hc_csv = cfg.get("hardcase_csv") or os.path.join(cfg["dataset_dir"], "hardcase_list.csv")
    need_voc = not need_post and (not has_entries(voc_ann) or _mtime(voc_ann) < lp_end)
    need_hc = not need_post and (not os.path.exists(hc_csv) or _mtime(hc_csv) < lp_end)
    need_pack = not need_post and not need_voc and not need_hc and (
        not has_entries(cfg["dataset_dir"]) or _mtime(cfg["dataset_dir"]) < lp_end)
    need_overlay = not need_post and not need_voc and not need_hc and not need_pack and (
        not os.path.exists(sheet) or _mtime(sheet) < lp_end)
    if not (need_post or need_voc or need_hc or need_pack or need_overlay):
        AUTO_STATE["phase"] = "done"
        if not AUTO_STATE.get("_done_logged"):
            AUTO_STATE["_done_logged"] = True
            log_auto("自动链已完成：处理 %d 帧，训练集与抽检图均已更新" % fin)
        return
    # 处理失败必须退避：否则会每隔 15 秒刷一个失败作业（实测踩到）
    if need_post and AUTO_STATE["last_done"] == done and \
            time.time() - AUTO_STATE["last_try"] < AUTO_POST_RETRY_SEC:
        AUTO_STATE["phase"] = "retry-wait"
        return
    if need_post:
        if _dispatch("post", {"limit": "0"}, "自动处理"):
            AUTO_STATE["phase"] = "dispatched:post"
            AUTO_STATE["last_try"] = time.time()
            AUTO_STATE["last_done"] = done
        return
    if need_voc:
        if _dispatch("voc", {}, "自动导出 VOC"):
            AUTO_STATE["phase"] = "dispatched:voc"
        return
    if need_hc:
        if _dispatch("hardcase", {}, "自动难例清单"):
            AUTO_STATE["phase"] = "dispatched:hardcase"
        return
    if need_pack:
        if _dispatch("pack", {"format": "coco"}, "自动打包"):
            AUTO_STATE["phase"] = "dispatched:pack"
        return
    if need_overlay:
        if _dispatch("overlay", {"limit": "20"}, "自动抽检"):
            AUTO_STATE["phase"] = "dispatched:overlay"
        return

def auto_loop():
    """后台守护线程：显卡记录常驻 + 自动后处理链。不依赖界面上任何人点按钮。"""
    time.sleep(3)
    last_tick = 0.0
    while True:
        try:
            ensure_gpu_watch()
            if time.time() - last_tick >= AUTO_POLL_SEC:
                last_tick = time.time()
                auto_pipeline_tick()
        except Exception as e:
            log_auto("自动化循环异常：%s" % e)
        time.sleep(5)

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

def count_dir(path, suffix="", cap=20000):
    """数目录里的文件（带上限，避免几十万帧时每次轮询都全量扫）。"""
    n = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_file() and (not suffix or e.name.endswith(suffix)):
                    n += 1
                    if n >= cap:
                        return cap
    except Exception:
        return 0
    return n

def has_entries(path):
    try:
        with os.scandir(path) as it:
            return next(it, None) is not None
    except Exception:
        return False

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
    st["render_running"] = render_session_running()
    st["tasks_total"] = count_lines(cfg["queue_file"])
    st["tasks_done"] = count_dir(cfg["done_dir"], ".done")
    st["final_count"] = count_dir(cfg["final_dir"], ".png")
    st["dataset_ready"] = has_entries(cfg["dataset_dir"])
    st["overlay_ready"] = os.path.exists(
        os.path.join(cfg["overlay_dir"], "contact_sheet.jpg"))
    # 出图前环境是否就绪（缺工程/脚本/引擎时，前端直接提示怎么办，不用点下去等报错）
    st["uproject_exists"] = os.path.exists(cfg["uproject"])
    st["capture_script_exists"] = os.path.exists(
        os.path.join(cfg["project_dir"], "Content", "Python", "batch_capture.py"))
    st["ue_ready"] = os.path.exists(
        os.path.join(cfg["ue_dir"], "Engine", "Binaries", "Linux", "UnrealEditor-Cmd"))
    # 后台自动化状态：显卡记录是否常驻 + 自动后处理链当前阶段
    gw = GPU_WATCH_PROC.get("proc")
    st["auto"] = {
        "enabled": AUTO_POST,
        "gpu_watch": bool(gw is not None and gw.poll() is None),
        "phase": AUTO_STATE["phase"],
        "last_log": (tail(os.path.join(LOG_DIR, "auto.log"), 3)
                     if os.path.exists(os.path.join(LOG_DIR, "auto.log")) else ""),
    }
    with JOBS_LOCK:
        st["jobs"] = [{k: j[k] for k in ("id", "action", "status", "ret", "started", "ended")}
                      for j in sorted(JOBS.values(), key=lambda x: x["started"], reverse=True)[:12]]
    return st

try:
    HTML_PAGE = open(os.path.join(BASE, "index.html"), encoding="utf-8").read()
except Exception:
    HTML_PAGE = "<h1>index.html 缺失</h1>"

# RENDER-JOBS: 参数重绘任务队列（串行，避免 GPU OOM）
import queue as _q
import threading as _th
import uuid as _uuid

_RJ = {}
_RJ_Q = _q.Queue()
_RJ_LOCK = _th.Lock()
_MANUAL_DIR = '/data/render/manual_task_records'
_CLIPS_CACHE={'value':None,'at':0}
_CLIPS_CACHE_LOCK=_th.Lock()

def _manual_save(jid, rec):
    os.makedirs(_MANUAL_DIR, exist_ok=True)
    data=dict(rec);data['id']=jid
    tmp=os.path.join(_MANUAL_DIR,'.'+jid+'.tmp')
    with open(tmp,'w',encoding='utf8') as f: json.dump(data,f,ensure_ascii=False)
    os.replace(tmp,os.path.join(_MANUAL_DIR,jid+'.json'))

def _manual_load(jid):
    if not re.fullmatch(r'(?:obj|gnd|tsr)_\d+_[0-9a-f]{6}',str(jid or '')):return None
    try:
        with open(os.path.join(_MANUAL_DIR,jid+'.json'),encoding='utf8') as f:return json.load(f)
    except (OSError,ValueError):return None

def _manual_versions():
    try:
        with open('/opt/ue5_synth/dashboard/release_versions.json',encoding='utf8') as f:return json.load(f).get('versions',{})
    except (OSError,ValueError):return {}

def _manual_list(kind):
    import glob
    result=[]
    for path in sorted(glob.glob(os.path.join(_MANUAL_DIR,kind+'_*.json')),reverse=True)[:200]:
        try:
            with open(path,encoding='utf8') as f:r=json.load(f)
            result.append(r)
        except (OSError,ValueError):continue
    return result

def _manual_recover():
    for kind in ('obj','gnd','tsr'):
        for rec in _manual_list(kind):
            if rec.get('state') in ('queued','running'):
                rec.update(state='failed',err='服务重启导致任务中断，请重新发起',t1=time.time())
                _manual_save(rec['id'],rec)
_manual_recover()


def _render_worker():
    while True:
        try:
            jid, kind, batch, clip, params = _RJ_Q.get()
        except Exception:
            break
        rec = _RJ.get(jid)
        if rec is None:
            continue
        out_dir = (os.path.join('/data/render','manual_gnd_'+jid) if kind == "gnd"
                   else os.path.join('/data/render_hybrid','obj3d_'+jid))
        queue_lock=None
        try:
            os.makedirs('/data/render_hybrid',exist_ok=True)
            queue_lock=open('/data/render_hybrid/.objtask.lock','a')
            if fcntl:fcntl.flock(queue_lock,fcntl.LOCK_EX)
            with _RJ_LOCK:
                rec["state"] = "running"
                _manual_save(jid,rec)
            os.makedirs(out_dir, exist_ok=True)
            payload = os.path.join(out_dir, "_p_%s.json" % jid)
            with open(payload, "w", encoding="utf-8") as f:
                if kind=='gnd' and params.get('_world_source'):
                    world_params={k:v for k,v in params.items() if not k.startswith('_')}
                    world_params.update(mat=params.get('lane_type','lane_dashed_white'),marking_mode='replace',render_mode='world_preview')
                    data={'batch':batch,'clip':clip,'source':params['_world_source'],'out':out_dir,'params':world_params}
                elif kind=='obj':
                    obj_params=dict(params)
                    obj_params['render_mode']='canonical_clip'
                    obj_params.setdefault('trajectory_mode','follow_road')
                    obj_params.setdefault('lights',{'enabled':bool(params.get('lights_enabled',True)),'target':'driving','brightness':1.0,'mode':'steady'})
                    data={"batch":batch,"clip":clip,"params":obj_params,"out":out_dir}
                    alias=os.path.join('/data/render','obj3d_'+jid)
                    if not os.path.lexists(alias):os.symlink(out_dir,alias,target_is_directory=True)
                else:data={"batch": batch, "clip": clip, "params": params,"out": out_dir}
                json.dump(data, f, ensure_ascii=False)
            if kind=='obj':
                with open(os.path.join(out_dir,'config.json'),'w',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False)
            script = os.path.join(TOOLS_DIR, "gnd_render_one.py" if kind == "gnd" else "render_obj_task.py")
            if kind=='gnd' and params.get('_world_source'):script=os.path.join(TOOLS_DIR,'gnd_clip_world.py')
            env = dict(os.environ)
            env["PYTHONPATH"] = TOOLS_DIR
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            r = subprocess.run([ALGO_PYTHON, script, "--config", payload],
                               capture_output=True, text=True,
                               timeout=3600, env=env)
            # COUNT-FIX: 输出可能在根目录或 frames/ 子目录，两处都数
            n = 0
            for _cand in (out_dir, os.path.join(out_dir, "frames")):
                if os.path.isdir(_cand):
                    _cnt = len([f for f in os.listdir(_cand)
                                if f.endswith(("_sim.jpg", "_gnd.jpg",
                                               "_obj.jpg"))])
                    n = max(n, _cnt)
            if kind=='obj':
                target=os.path.join('/data/render_hybrid','obj3d_'+jid)
                manifest=os.path.join(target,'annotations','__clip__.json')
                trajectory=os.path.join(target,'trajectory_manifest.json')
                if os.path.isfile(trajectory):
                    with open(trajectory,encoding='utf8') as f: tm=json.load(f)
                    m={'job':'obj3d_'+jid,'batch':batch,'clip':clip,'total':int(tm.get('frames_requested') or 0),'synthetic':int(tm.get('frames_rendered') or 0),'motion_assumption':tm.get('trajectory_mode'),'training_eligible':False}
                    os.makedirs(os.path.dirname(manifest),exist_ok=True)
                    with open(manifest,'w',encoding='utf8') as f:json.dump(m,f,ensure_ascii=False)
                if os.path.isfile(manifest):
                    with open(manifest,encoding='utf8') as f: m=json.load(f)
                    n=int(m.get('synthetic') or 0)
                    rec['total']=int(m.get('total') or 0)
                    rec['output_job']='obj3d_'+jid
            else:
                rec['output_job']='manual_gnd_'+jid
                rec['suffix']='_gnd.jpg' if params.get('_world_source') else '_sim.jpg'
                if params.get('_world_source'):
                    trajectory=os.path.join(out_dir,'trajectory_manifest.json')
                    if os.path.isfile(trajectory):
                        with open(trajectory,encoding='utf8') as f: gm=json.load(f)
                        rec['total']=int(gm.get('frames_total') or 0)
                        n=int(gm.get('frames_with_paint') or 0)
            with _RJ_LOCK:
                rec["state"] = "done" if r.returncode == 0 and n>0 else "failed"
                rec["n"] = n
                if r.returncode != 0 or n==0:
                    rec["err"] = ((r.stderr or r.stdout or "")[-600:] if r.returncode != 0 else '任务执行结束，但没有生成有效仿真帧')
                rec["t1"] = time.time()
        except subprocess.TimeoutExpired:
            with _RJ_LOCK:
                rec["state"] = "failed"
                rec["err"] = "渲染超过服务器最大时限（60分钟）"
        except Exception as e:
            with _RJ_LOCK:
                rec["state"] = "failed"
                rec["err"] = str(e)[:400]
        finally:
            if queue_lock:queue_lock.close()
            with _RJ_LOCK:
                rec.setdefault('t1',time.time());_manual_save(jid,rec)
            _RJ_Q.task_done()
            try:
                os.remove(os.path.join(out_dir, "_p_%s.json" % jid))
            except Exception:
                pass


_th.Thread(target=_render_worker, daemon=True).start()


# STUDIO-CONSTS: 素材工坊路径
RAW_EXTERNAL = "/data/assets_global/OBJ_3D/raw_external"
CANON_DIR = "/data/assets_global/OBJ_3D/canonical"
OBJ3D_INDEX = "/data/assets_global/OBJ_3D/asset_index_obj3d.json"
BLENDER = "/data/blender-4.2.23-linux-x64/blender"


def _map_spec_params(params):
    """SPEC-PARAM-MAP: 前端字段 → cfg 字段。"""
    if not isinstance(params, dict):
        return params
    p = dict(params)
    # 指定素材
    if p.get("spec_on") and p.get("sprite_path"):
        p["sprite_path"] = p["sprite_path"]
    else:
        p.pop("sprite_path", None)
    # 指定位置
    if p.get("pos_on"):
        try:
            _x = float(p.get("pos_x", 20))
            _y = float(p.get("pos_y", 0))
            p["manual_pos"] = [[_x, _y]]
        except Exception:
            pass
    else:
        p.pop("manual_pos", None)
    # 清理前端专用字段
    for k in ("spec_on", "pos_on", "pos_x", "pos_y"):
        p.pop(k, None)
    return p


def _start_render_job(kind, batch, clip, params):
    params = _map_spec_params(params)

    jid = "%s_%d_%s" % (kind, int(time.time()), _uuid.uuid4().hex[:6])
    with _RJ_LOCK:
        _RJ[jid] = {"state": "queued", "kind": kind, "batch": batch,
                    "clip": clip, "n": 0, "t0": time.time(),"manual":True,"params":params,"versions":_manual_versions(),"source":params.get('_world_source') or os.path.join('/data/test_data/real_clips',batch,clip)}
        if params.get('_world_source'):_RJ[jid]['world_source']=params['_world_source']
        if params.get('_world_source'):_RJ[jid]['warning']='验证预览：道路按自车轨迹近似；反光为亮度近似，遮挡依赖路面分割，尚不可直接用于训练'
        _manual_save(jid,_RJ[jid])
        if len(_RJ) > 40:
            for o in sorted(_RJ, key=lambda k: _RJ[k]["t0"])[:10]:
                if _RJ[o]["state"] in ("done", "failed"):
                    _RJ.pop(o, None)
    _RJ_Q.put((jid, kind, batch, clip, params))
    return jid


def _render_job_status(jid):
    with _RJ_LOCK:
        rec = dict(_RJ.get(jid) or {})
    if not rec:
        rec = _manual_load(jid)
        if not rec:return None
    rec["id"] = jid
    return rec


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
    def _send_image(self, max_w=900):
        """返回一个「读文件→按需缩放→发出」的闭包。缩放用 PIL，没有就原样发。"""
        def _go(path):
            try:
                with open(path, "rb") as f:
                    blob = f.read()
            except OSError:
                return self._send(404, {"error": "image not found"})
            try:
                from PIL import Image
                import io as _io
                im = Image.open(path)
                if im.width > max_w:
                    h = int(im.height * max_w / float(im.width))
                    im = im.resize((max_w, h), Image.LANCZOS)
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                buf = _io.BytesIO()
                im.save(buf, "JPEG", quality=88)
                blob = buf.getvalue()
                return self._send(200, blob, "image/jpeg")
            except Exception:
                ctype = "image/png" if path.lower().endswith(".png") else "image/jpeg"
                return self._send(200, blob, ctype)
        return _go

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

        if u.path == '/api/render/jobs':
            kind=(q.get('module') or [''])[0].lower()
            if kind not in ('obj','gnd','tsr'):return self._send(400,{'ok':False,'error':'invalid module'})
            return self._send(200,{'ok':True,'tasks':_manual_list(kind)})
        if u.path == '/api/platform/versions':
            try:
                with open('/opt/ue5_synth/dashboard/release_versions.json',encoding='utf8') as f: data=json.load(f)
                return self._send(200, {'ok':True, **data})
            except Exception as e: return self._send(500, {'ok':False,'error':str(e)[:200]})
        if u.path == '/api/platform/countries':
            try:
                with open('/usr/share/iso-codes/json/iso_3166-1.json',encoding='utf8') as f: data=json.load(f)
                return self._send(200, {'ok':True,'countries':[{'code':x['alpha_2'],'name':x['name']} for x in data['3166-1']]})
            except Exception as e: return self._send(500, {'ok':False,'error':str(e)[:200]})

        if u.path in ('/api/taskclip/assets', '/api/taskclip/status', '/api/taskclip/frame', '/api/taskclip/list'):
            try:
                import clip_task_api as _tc
                if u.path.endswith('/assets'): return self._send(200, {'ok': True, 'assets': _tc.assets()})
                if u.path.endswith('/list'): return self._send(200, {'ok': True, 'tasks': _tc.list_gnd_tasks((q.get('module') or ['gnd'])[0].lower())})
                job=(q.get('job') or [''])[0]
                if u.path.endswith('/status'): return self._send(200, _tc.status(job))
                return self._send(200, _tc.frame(job, int((q.get('index') or ['0'])[0])), 'image/jpeg')
            except (ValueError, OSError, KeyError) as e: return self._send(400, {'ok':False,'error':str(e)[:300]})
            except Exception as e: return self._send(500, {'ok':False,'error':str(e)[:300]})

        if u.path in ('/api/studio/v2/image3d-status', '/api/studio/v2/image3d-model'):
            try:
                if TOOLS_DIR not in sys.path: sys.path.insert(0, TOOLS_DIR)
                import studio_image3d
                task=(q.get('task') or [''])[0]
                if u.path.endswith('status'):
                    return self._send(200, studio_image3d.status(task))
                with open(studio_image3d.model(task),'rb') as f: data=f.read()
                return self._send(200, data, 'model/gltf-binary')
            except Exception as e:
                return self._send(400, {'ok':False, 'error':str(e)[:500]})

        if u.path == '/api/studio/v2/2d-thumb':
            try:
                import sys as _studio_sys
                if TOOLS_DIR not in _studio_sys.path: _studio_sys.path.insert(0, TOOLS_DIR)
                from studio_2d import preview
                path=preview((q.get('task') or [''])[0])
                if not os.path.isfile(path): return self._send(404, {'error':'not found'})
                with open(path,'rb') as f: data=f.read()
                return self._send(200,data,'image/png')
            except (ValueError,OSError) as e:
                return self._send(400,{'error':str(e)})

        if u.path == '/api/studio/v2/model':
            try:
                import sys as _studio_sys
                if TOOLS_DIR not in _studio_sys.path: _studio_sys.path.insert(0, TOOLS_DIR)
                from studio_safe import model_path
                path=model_path((q.get('task') or [''])[0], (q.get('version') or [''])[0])
                if not os.path.isfile(path): return self._send(404, {'error':'not found'})
                self.send_response(200)
                self.send_header('Content-Type','model/gltf-binary')
                self.send_header('Content-Length',str(os.path.getsize(path)))
                self.send_header('Cache-Control','private, no-store')
                self.end_headers()
                with open(path,'rb') as f:
                    while True:
                        chunk=f.read(1024*1024)
                        if not chunk: break
                        self.wfile.write(chunk)
                return
            except (ValueError,OSError) as e:
                return self._send(400,{'error':str(e)})

        if u.path == '/api/studio/v2/thumb':
            try:
                import sys as _studio_sys
                if TOOLS_DIR not in _studio_sys.path: _studio_sys.path.insert(0, TOOLS_DIR)
                from studio_safe import thumb
                path = thumb((q.get('task') or [''])[0], (q.get('version') or [''])[0])
                if not os.path.isfile(path): return self._send(404, {'error':'not found'})
                with open(path,'rb') as f: data=f.read()
                return self._send(200,data,'image/png')
            except (ValueError, OSError) as e:
                return self._send(400,{'error':str(e)})

        if u.path in ("/gallery", "/gallery.html"):
            with open(os.path.join(BASE, "gallery.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")

        # [PRO-UI] 新版 Pro 界面
        if u.path in ("/pro", "/pro.html"):
            fp = os.path.join(BASE, "pro.html")
            try:
                with open(fp, "rb") as f:
                    return self._send(200, f.read(),
                                      "text/html; charset=utf-8")
            except OSError:
                return self._send(404, {"error": "pro.html 缺失"})


        # [WALK-JOBS] 列出最近的遍历任务（供刷新后恢复状态）

        # [GND-PROJECT] 读取产出图
        if u.path == "/api/gnd/result":
            d = (q.get("dir") or [""])[0]
            n = (q.get("name") or [""])[0]
            if not d or not n or "/" in d or ".." in d or "/" in n or ".." in n:
                return self._send(400, {"error": "bad param"})
            p = os.path.join("/data/render_hybrid", d, n)
            if not os.path.isfile(p):
                return self._send(404, {"error": "not found"})
            with open(p, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return


        # [JOB-IMAGES] 列出某任务目录下的产出图
        if u.path == "/api/job/images":
            job = (q.get("job") or [""])[0]
            if not job:
                return self._send(400, {"error": "缺少 job"})
            cands = [
                os.path.join(TSR_OUT_DIR, "walk_" + job),
                os.path.join(TSR_OUT_DIR, job),
                os.path.join("/mnt/Data_Platform/zhj_datasim", job),
                os.path.join("/data/render_hybrid", job),
            ]
            dp = next((c for c in cands if os.path.isdir(c)), "")
            if not dp:
                return self._send(404, {"error": "目录不存在"})
            try:
                imgs = sorted([
                    f for f in os.listdir(dp)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    and not f.startswith("contact")
                    and "_mask" not in f and "_fg" not in f and "_shadow" not in f
                ])
                return self._send(200, {"job": job, "dir": dp, "images": imgs})
            except Exception as e:
                return self._send(500, {"error": str(e)})


        # [JOB-ANNOTATION] 读取任务目录下某帧的标注
        if u.path == "/api/job/annotation":
            job = (q.get("job") or [""])[0]
            stem = (q.get("stem") or [""])[0]
            if not job or "/" in job or ".." in job:
                return self._send(400, {"error": "参数错误"})
            cands = [
                os.path.join(TSR_OUT_DIR, "walk_" + job),
                os.path.join(TSR_OUT_DIR, job),
                os.path.join("/mnt/Data_Platform/zhj_datasim", job),
                os.path.join("/data/render_hybrid", job),
            ]
            dp = next((c for c in cands if os.path.isdir(c)), "")
            if not dp:
                return self._send(404, {"error": "目录不存在"})
            ad = os.path.join(dp, "annotations")
            fp = ""
            # 严格按 stem 找；找不到就 404，绝不回退到别的帧（否则静默错配）
            if stem and "/" not in stem and ".." not in stem:
                cand = os.path.join(ad, stem + ".json")
                if os.path.isfile(cand):
                    fp = cand
            if not fp:
                return self._send(404, {"error": "该帧标注不存在",
                                        "stem": stem})
            with open(fp, "r", encoding="utf-8") as f:
                return self._send(200, json.load(f))

        # [JOB-IMAGE] 读取单张产出图
        if u.path == "/api/job/image":
            job = (q.get("job") or [""])[0]
            name = (q.get("name") or [""])[0]
            if not job or not name or "/" in name or ".." in name:
                return self._send(400, {"error": "参数错误"})
            cands = [
                os.path.join(TSR_OUT_DIR, "walk_" + job),
                os.path.join(TSR_OUT_DIR, job),
                os.path.join("/mnt/Data_Platform/zhj_datasim", job),
                os.path.join("/data/render_hybrid", job),
            ]
            dp = next((c for c in cands if os.path.isdir(c)), "")
            if not dp:
                return self._send(404, {"error": "目录不存在"})
            fp = os.path.join(dp, name)
            if not os.path.isfile(fp):
                return self._send(404, {"error": "文件不存在"})
            with open(fp, "rb") as f:
                data = f.read()
            ext = os.path.splitext(name)[1].lower()
            ct = {".png": "image/png", ".jpg": "image/jpeg",
                  ".jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        # ============ [REAL] 真实数据集接口（只读 /data/test_data） ============
        REAL_ROOT = "/data/test_data/real_clips"

        def _real_clips():
            # EXCLUDE-TRUCK: 排除大车占道场景（不适合 OBJ 仿真）
            _EXCL = ("extremely_closetruck", "extremely_closecar")
            # CLIPS-MULTI-ROOT: 扫描 real_clips + real_clips_extra
            out = []
            seen = set()
            # Scan output indexes once, not once for every network batch.
            import glob as _glob
            _preferred=set();_available=set()
            for _af in _glob.glob('/data/render/obj_good/annotations/*.json')[:400]:
                try:
                    with open(_af,encoding='utf8') as f: b=(json.load(f) or {}).get('batch')
                    if b:_preferred.add(b)
                except (OSError,ValueError):pass
            for _cfg in _glob.glob('/data/render_hybrid/*/config.json'):
                if os.path.basename(os.path.dirname(_cfg)).startswith('gnd'):continue
                try:
                    with open(_cfg,encoding='utf8') as f:b=(json.load(f) or {}).get('batch')
                    if b:_available.add(b)
                except (OSError,ValueError):pass
            for root in (REAL_ROOT, "/data/test_data/real_clips_extra"):
                if not os.path.isdir(root):
                    continue
                for n in sorted(os.listdir(root)):
                    if n in seen:
                        continue
                    p = os.path.join(root, n)
                    if n not in _available:
                        continue
                    if not os.path.isdir(p):
                        continue
                    # MULTI-CAM-SCAN: 支持多种前向相机
                    _CAMS = ("cam_st_x2f_front", "cam_ld_x8d_front",
                             "cam_st_x8d_front", "cam_hk_x8b_front",
                             "images")
                    try:
                        clips = [d for d in os.listdir(p)
                                 if any(os.path.isdir(os.path.join(p, d, c))
                                        for c in _CAMS)]
                    except OSError:
                        continue
                    if not clips:
                        continue
                    njpg = 0
                    for c in clips:
                        for _cam in _CAMS:
                            cd = os.path.join(p, c, _cam)
                            if not os.path.isdir(cd):
                                continue
                            try:
                                njpg += len([f for f in os.listdir(cd)
                                             if f.lower().endswith(".jpg")])
                            except OSError:
                                pass
                            break
                    if any(x in n for x in _EXCL):
                        continue
                    # ONLY-WITH-OUTPUT: 只保留 obj_good 里有该批次的
                    _pref = int(n in _preferred)
                    # ONLY-RENDERED-CLIPS: 仿真工作台只列有仿真产出的批次
                    if n not in _available:
                        continue
                    out.append({"name": n, "clips": len(clips),
                                "frames": njpg, "preferred": _pref})
                    seen.add(n)
            return out

        def _real_clip_frames(clip_name, batch):
            # MULTI-CAM-FRAMES: 支持多种相机目录
            bp = os.path.join(REAL_ROOT, batch)
            if not os.path.isdir(bp):
                return []
            cp = os.path.join(bp, clip_name)
            cd = None
            for _cam in ("cam_st_x2f_front", "cam_ld_x8d_front",
                         "cam_st_x8d_front", "cam_hk_x8b_front",
                         "images"):
                _p = os.path.join(cp, _cam)
                if os.path.isdir(_p):
                    cd = _p
                    break
            if cd is None:
                return []
            fs = sorted(f for f in os.listdir(cd) if f.lower().endswith(".jpg"))
            return [{"stem": os.path.splitext(f)[0], "file": f} for f in fs]

        # 列出数据包
        # -------- RENDER-JOBS-LIST: 任务中心列表 --------
        # -------- SELFCHECK-API: 供用户端自检 --------
        if u.path == "/api/selfcheck":
            import hashlib as _hh
            try:
                _body = b""
                with open(os.path.join(BASE, "index.html"), "rb") as _f:
                    _body = _f.read()
                _h = _hh.md5(_body).hexdigest()[:12]
                _txt = _body.decode("utf-8", "ignore")
                _marks = {}
                for _m in ("BBOX-ANCHOR-FIX", "BBOX-ANN-RESOLVE",
                           "FS-SCALE-HOIST", "FS-IMG-FIT",
                           "FSMOD-SYNC-FIX", "FS-COORD-FIX",
                           "THUMB-DROPDOWN-COMPONENT", "ROUTE-A-CLIPVIEW",
                           "CLIPPLAY-HYBRID-SCAN", "COORD-ORDER-FIX"):
                    _marks[_m] = _txt.count(_m)
                return self._send(200, {
                    "ok": True,
                    "html_bytes": len(_body),
                    "html_md5": _h,
                    "markers": _marks,
                    "server_pid": os.getpid(),
                    "server_time": __import__("time").strftime(
                        "%Y-%m-%d %H:%M:%S"),
                    "build": "SELFCHECK-API v1",
                })
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)[:200]})

        # -------- LIGHT-API: 车灯组索引 --------
        if u.path == "/api/light/index":
            try:
                LIGHT_IDX = "/data/assets_global/OBJ_3D/light_index.json"
                if not os.path.isfile(LIGHT_IDX):
                    return self._send(404, {"ok": False,
                                            "error": "索引未生成，先跑 build_light_index.py"})
                with open(LIGHT_IDX, encoding="utf-8") as fh:
                    d = json.load(fh)
                return self._send(200, d)
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)[:200]})

        if u.path == "/api/render/jobs":
            try:
                with _RJ_LOCK:
                    items = []
                    for _jid, _rec in _RJ.items():
                        _r = dict(_rec)
                        _r["id"] = _jid
                        items.append(_r)
                items.sort(key=lambda x: x.get("t0") or 0, reverse=True)
                return self._send(200, {"ok": True, "jobs": items[:500]})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)[:200]})

        # -------- DELIVERY-EXPORT-API: 数据交付导出 --------
        if u.path == "/api/delivery/presets":
            # 常用交付路径预设
            _pre = [
                {"label": "网盘 zhj_datasim（默认）",
                 "path": "/mnt/Data_Platform/zhj_datasim/delivery"},
                {"label": "本地 /data/delivery",
                 "path": "/data/delivery"},
                {"label": "网盘 zhj_datasim/OBJ",
                 "path": "/mnt/Data_Platform/zhj_datasim/OBJ"},
            ]
            _mods = [
                {"key": "obj", "label": "OBJ 3D 目标"},
                {"key": "tsr", "label": "TSR 交通标牌"},
                {"key": "gnd", "label": "GND 地面标线"},
            ]
            return self._send(200, {"ok": True, "presets": _pre,
                                    "modules": _mods})

        if u.path == "/api/delivery/check":
            # 检查目标路径是否可写
            try:
                qs = parse_qs(u.query or "")
                path = (qs.get("path") or [""])[0].strip()
                if not path:
                    return self._send(400, {"ok": False,
                                            "error": "missing path"})
                os.makedirs(path, exist_ok=True)
                tst = os.path.join(path, ".dsh_write_test")
                with open(tst, "w") as f:
                    f.write("ok")
                os.remove(tst)
                # 剩余空间
                import shutil as _sh2
                tot, used, free = _sh2.disk_usage(path)
                return self._send(200, {
                    "ok": True, "writable": True,
                    "free_gb": round(free / 1073741824.0, 1),
                    "total_gb": round(tot / 1073741824.0, 1)})
            except Exception as e:
                return self._send(200, {"ok": True, "writable": False,
                                        "error": str(e)[:200]})

        if u.path == "/api/delivery/export":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                modules = body.get("modules") or ["obj"]
                out_root = str(body.get("path") or "").strip()
                job_id = str(body.get("job") or "").strip()
                if not out_root:
                    return self._send(400, {"ok": False,
                                            "error": "missing path"})
                os.makedirs(out_root, exist_ok=True)

                import subprocess as _sp5
                results = []
                env = dict(os.environ)
                env["PYTHONPATH"] = TOOLS_DIR

                for mod in modules:
                    try:
                        if mod == "obj":
                            script = os.path.join(TOOLS_DIR,
                                                  "export_to_label.py")
                            ann = os.path.join("/data", "render_hybrid",
                                               job_id, "annotations")
                            if not (job_id and os.path.isdir(ann)):
                                results.append({
                                    "module": "obj", "ok": False,
                                    "error": "无标注目录（先跑渲染）"})
                                continue
                            out = os.path.join(out_root, "OBJ")
                            ok_n = 0
                            for fp in sorted(os.listdir(ann))[:200]:
                                if not fp.endswith(".json"):
                                    continue
                                r5 = _sp5.run(
                                    [ALGO_PYTHON, script,
                                     "--ann", os.path.join(ann, fp),
                                     "--out", out],
                                    capture_output=True, text=True,
                                    timeout=120, env=env)
                                if r5.returncode == 0:
                                    ok_n += 1
                            results.append({"module": "obj", "ok": True,
                                            "count": ok_n, "out": out})
                        else:
                            results.append({
                                "module": mod, "ok": False,
                                "error": "需要 %s 仿真产出" % mod})
                    except Exception as e5:
                        results.append({"module": mod, "ok": False,
                                        "error": str(e5)[:200]})

                return self._send(200, {"ok": True, "path": out_root,
                                        "results": results})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)[:300]})

        # -------- CHECK-CLIP-API: 数据目录自检 --------
        if u.path == "/api/data/check":
            try:
                qs = parse_qs(u.query or "")
                path = (qs.get("path") or [""])[0].strip()
                force = (qs.get("type") or ["auto"])[0]
                if not path:
                    return self._send(400, {"ok": False,
                                            "error": "missing path"})
                ap = [os.path.join(TOOLS_DIR, "check_clip.py"), path]
                if force in ("obj", "gnd"):
                    ap += ["--type", force]
                ap.append("--json")
                import subprocess as _sp4
                r4 = _sp4.run([ALGO_PYTHON] + ap, capture_output=True,
                              text=True, timeout=180)
                try:
                    import json as _j5
                    d5 = _j5.loads(r4.stdout or "{}")
                    return self._send(200, {"ok": True, "result": d5})
                except Exception:
                    return self._send(200, {"ok": False,
                                            "raw": (r4.stdout or "")[:2000],
                                            "err": (r4.stderr or "")[:500]})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)[:200]})

        # -------- STUDIO-SPLIT-GET: 素材工坊 GET 接口 --------
        if u.path == "/api/studio/classes":
            # STUDIO-CLASSES-FIX: 不导入 bpy 依赖，直接内联类目表
            _TBL = {
                "car": 4.6, "suv": 4.8, "pickup": 5.4, "van": 5.1,
                "truck": 8.5, "bus": 12.0,
                "tricycle": 2.6, "motorcycle": 2.1, "bicycle": 1.8,
                "rickshaw": 2.8, "low_speed_ev": 2.9,
                "scooter": 1.2, "skateboard": 0.8,
                "pedestrian": 1.7, "rider": 1.8, "child": 1.2,
                "camel": 2.2, "kangaroo": 1.6, "animal": 1.5,
                "cone": 0.7, "barrier": 1.0,
            }
            return self._send(200, {"ok": True, "classes": _TBL})

        if u.path == "/api/studio/status":
            try:
                qs = parse_qs(u.query or "")
                uid = (qs.get("uid") or [""])[0]
                meta = os.path.join(RAW_EXTERNAL, uid, "meta.json")
                if not os.path.isfile(meta):
                    return self._send(200, {"ok": True, "exists": False})
                import json as _j
                return self._send(200, {"ok": True, "exists": True,
                                        "meta": _j.load(open(meta, encoding="utf-8"))})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)[:200]})

        if u.path == "/api/real/clips":
            try:
                with _CLIPS_CACHE_LOCK:
                    if _CLIPS_CACHE['value'] is None or time.monotonic()-_CLIPS_CACHE['at']>30:
                        _CLIPS_CACHE.update(value=_real_clips(),at=time.monotonic())
                    cached=_CLIPS_CACHE['value']
                return self._send(200, {"root": REAL_ROOT,"clips": cached})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        # REAL-CLIP-FRAMES-API: 列出某批次下的 seq 目录
        if u.path == "/api/real/clips/frames":
            import glob as _gf
            try:
                qs = parse_qs(u.query or "")
                batch = (qs.get("batch") or [""])[0]
                if not batch:
                    return self._send(400, {"error": "missing batch"})
                # FRAMES-MULTIROOT: 两个根目录都要找
                _roots = [REAL_ROOT,
                          "/data/test_data/real_clips_extra"]
                cd = None
                for _r0 in _roots:
                    _c = os.path.join(_r0, batch)
                    if os.path.isdir(_c):
                        cd = _c
                        break
                if cd is None:
                    return self._send(200, {"batch": batch,
                                            "frames": [],
                                            "note": "batch not found"})
                # FRAMES-LOGIC-FIX: 子目录直接含 cam_st_x2f_front 就是 seq
                frames = []
                # FRAMES-LEVEL-OUTGATE: 只列有仿真产出的 clip
                _done2 = set()
                import glob as _gg2
                for _cf2 in _gg2.glob("/data/render_hybrid/*/config.json"):
                    try:
                        with open(_cf2, encoding="utf-8") as _fh2:
                            _cd2 = json.load(_fh2) or {}
                        if _cd2.get("batch") == batch and _cd2.get("clip"):
                            _done2.add(_cd2["clip"])
                    except Exception:
                        pass
                for d in sorted(os.listdir(cd)):
                    dp = os.path.join(cd, d)
                    if not os.path.isdir(dp):
                        continue
                    if d not in _done2:
                        continue
                    for cam in ("cam_st_x2f_front", "cam_ld_x8d_front",
                                "cam_st_x8d_front", "cam_hk_x8b_front",
                                "images"):
                        cp = os.path.join(dp, cam)
                        if os.path.isdir(cp):
                            nd = len(_gf.glob(os.path.join(cp, "*.jpg")))
                            if nd == 0:
                                nd = len(_gf.glob(os.path.join(cp, "*.png")))
                            frames.append({"name": d, "frames": nd})
                            break
                    else:
                        # 再深一层（batch/xxx/yyy/cam_...）
                        for sub in sorted(os.listdir(dp)):
                            sp2 = os.path.join(dp, sub)
                            if not os.path.isdir(sp2):
                                continue
                            for cam in ("cam_st_x2f_front", "images"):
                                cp = os.path.join(sp2, cam)
                                if os.path.isdir(cp):
                                    nd = len(_gf.glob(os.path.join(cp, "*.jpg")))
                                    frames.append({"name": sub, "frames": nd})
                                    break
                return self._send(200, {"batch": batch, "frames": frames})
            except Exception as e:
                return self._send(500, {"error": str(e)})


        # 列出某包下的 clip
        # SPEC-ASSET-POS: 素材库列表（供人工指定）
        if u.path == "/api/obj/assets":
            import glob as _ga
            # OBJ-PATCH-ROOT: Blender 渲染贴片库（新素材主源）
            _roots = ["/data/assets_global/OBJ_PATCH",
                      "/data/obj_assets_real", "/data/obj_assets_pc",
                      "/data/obj_assets"]
            _out = {}
            for _r in _roots:
                if not os.path.isdir(_r):
                    continue
                for _cls in sorted(os.listdir(_r)):
                    _cd = os.path.join(_r, _cls)
                    if not os.path.isdir(_cd):
                        continue
                    _fs = sorted(_ga.glob(os.path.join(_cd, "*.png")))
                    if not _fs:
                        continue
                    _lst = _out.setdefault(_cls, [])
                    # DEDUP-BY-MODEL: 同一模型的 12 个 yaw 只保留 yaw180 作代表
                    _bymodel = {}
                    for _f in _fs:
                        _bn = os.path.basename(_f)
                        _base = _bn.split("_yaw")[0] \
                            if "_yaw" in _bn else _bn[:-4]
                        _yaw = 180
                        if "_yaw" in _bn:
                            try:
                                _yaw = int(_bn.split("_yaw")[1][:3])
                            except Exception:
                                _yaw = 180
                        _prev = _bymodel.get(_base)
                        # 优先 yaw180，其次最接近 180
                        if (_prev is None or
                                abs(_yaw - 180) < abs(_prev[0] - 180)):
                            _bymodel[_base] = (_yaw, _f)
                    for _base, (_yaw, _f) in sorted(_bymodel.items()):
                        _thumb = "/api/obj/asset_thumb?" + \
                                 "path=" + quote(_f)
                        _n_yaw = len([x for x in _fs
                                      if x.split("_yaw")[0] == _f.split(
                                          "_yaw")[0]])
                        _lst.append({"name": _base,
                                     "path": _f, "thumb": _thumb,
                                     "w": 0, "yaw": _yaw,
                                     "yaw_count": _n_yaw})
            for _c in _out:
                _out[_c] = _out[_c][:200]
            return self._send(200, {"classes": _out,
                                    "total": sum(len(v) for v in
                                                 _out.values())})

        # 素材缩略图
        if u.path == "/api/obj/asset_thumb":
            import glob as _gt
            _p = (q.get("path") or [""])[0]
            if (not _p or ".." in _p or
                    not any(_p.startswith(_r) for _r in
                            ("/data/assets_global/OBJ_PATCH",
                             "/data/obj_assets_real", "/data/obj_assets_pc",
                             "/data/obj_assets"))):
                return self._send(400, {"error": "bad path"})
            if not os.path.isfile(_p):
                return self._send(404, {"error": "not found"})
            # THUMB-PIL: 用 PIL 生成缩略图（服务端无 cv2）
            try:
                from PIL import Image as _PILImage
                import io as _io
                _im = _PILImage.open(_p).convert("RGBA")
                _im.thumbnail((120, 96), _PILImage.LANCZOS)
                _bg = _PILImage.new("RGBA", _im.size, (40, 40, 40, 255))
                _bg.alpha_composite(_im)
                _buf2 = _io.BytesIO()
                _bg.convert("RGB").save(_buf2, format="PNG")
                _data = _buf2.getvalue()
            except Exception as _te:
                return self._send(500, {"error": "thumb: " + str(_te)[:50]})
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_data)))
            self.send_header("Cache-Control", "max-age=600")
            self.end_headers()
            self.wfile.write(_data)
            return

        # GND 素材列表
        if u.path == "/api/gnd/assets":
            _gr = "/data/gnd_assets"
            _out = []
            if os.path.isdir(_gr):
                for _n in sorted(os.listdir(_gr)):
                    _d = os.path.join(_gr, _n)
                    if not os.path.isdir(_d):
                        continue
                    _meta = {}
                    _mp = os.path.join(_d, "meta.json")
                    if os.path.isfile(_mp):
                        try:
                            with open(_mp, encoding="utf-8") as _mf:
                                _meta = json.load(_mf)
                        except Exception:
                            _meta = {}
                    _meta["key"] = _n
                    _meta["thumb"] = ("/api/gnd/asset_thumb?key=" +
                                      quote(_n) + "&wear=" +
                                      str(int(_meta.get("_w", 0))))
                    _out.append(_meta)
            return self._send(200, {"assets": _out})

        # GND 素材缩略图
        if u.path == "/api/gnd/asset_thumb":
            _gkey = (q.get("key") or [""])[0]
            _wear = int((q.get("wear") or ["0"])[0] or 0)
            if not _gkey or "/" in _gkey or ".." in _gkey:
                return self._send(400, {"error": "bad key"})
            _fn = "tex.png"
            if _wear >= 45:
                _fn = "tex_wear55.png"
            elif _wear >= 15:
                _fn = "tex_wear25.png"
            _p = os.path.join("/data/gnd_assets", _gkey, _fn)
            if not os.path.isfile(_p):
                _p = os.path.join("/data/gnd_assets", _gkey, "tex.png")
            if not os.path.isfile(_p):
                return self._send(404, {"error": "not found"})
            try:
                from PIL import Image as _PI
                import io as _io2
                _im = _PI.open(_p).convert("RGBA")
                _w0, _h0 = _im.size
                _sc2 = 64.0 / max(_w0, _h0)
                _im = _im.resize((max(int(_w0 * _sc2), 4),
                                  max(int(_h0 * _sc2), 4)),
                                 _PI.LANCZOS)
                _bg2 = _PI.new("RGBA", _im.size, (24, 27, 36, 255))
                _bg2.alpha_composite(_im)
                _b2 = _io2.BytesIO()
                _bg2.convert("RGB").save(_b2, format="PNG")
                _data2 = _b2.getvalue()
            except Exception as _e2:
                return self._send(500, {"error": str(_e2)[:60]})
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_data2)))
            self.send_header("Cache-Control", "max-age=600")
            self.end_headers()
            self.wfile.write(_data2)
            return

        if u.path == "/api/real/batch":
            batch = (q.get("batch") or [""])[0]
            if not batch or "/" in batch or ".." in batch:
                return self._send(400, {"error": "bad batch"})
            # BATCH-MULTI-ROOT: 同时支持 real_clips 和 extra
            bp = None
            for _r in (REAL_ROOT, "/data/test_data/real_clips_extra"):
                _p = os.path.join(_r, batch)
                if os.path.isdir(_p):
                    bp = _p
                    break
            if bp is None:
                return self._send(404, {"error": "not found"})
            # BATCH-FILTER-OUT: 只返回有 obj_good 产出的 clip
            import glob as _g3
            _prod_stems = set()
            for _f3 in _g3.glob("/data/render/obj_good/annotations/*.json"):
                try:
                    with open(_f3, encoding="utf-8") as _fh3:
                        _d3 = json.load(_fh3)
                    if _d3.get("batch") == batch:
                        _prod_stems.add(_d3.get("frame"))
                except Exception:
                    pass
            items = []
            # CLIP-LEVEL-OUTGATE: 只返回真正跑过仿真的 clip
            _done_clips = set()
            import glob as _gc
            for _cf in _gc.glob("/data/render_hybrid/*/config.json"):
                try:
                    with open(_cf, encoding="utf-8") as _fh:
                        _cd = json.load(_fh) or {}
                    if _cd.get("batch") == batch and _cd.get("clip"):
                        _done_clips.add(_cd["clip"])
                except Exception:
                    pass
            # BATCH-MULTICAM-FIX: 支持多种相机目录
            _BCAMS = ("cam_st_x2f_front", "cam_ld_x8d_front",
                      "cam_st_x8d_front", "cam_hk_x8b_front", "images")
            for d in sorted(os.listdir(bp)):
                if d not in _done_clips:
                    continue
                cd = None
                for _c in _BCAMS:
                    _cp = os.path.join(bp, d, _c)
                    if os.path.isdir(_cp):
                        cd = _cp
                        break
                if cd is None:
                    continue
                _stems = [os.path.splitext(f)[0]
                          for f in os.listdir(cd)
                          if f.lower().endswith(".jpg")]
                if _prod_stems:
                    _stems = [x for x in _stems if x in _prod_stems]
                if not _stems:
                    continue
                items.append({"clip": d, "frames": len(_stems)})
            return self._send(200, {"batch": batch, "clips": items})

        # 列出某 clip 的帧
        if u.path == "/api/real/frames":
            batch = (q.get("batch") or [""])[0]
            clip = (q.get("clip") or [""])[0]
            if not batch or not clip or "/" in clip or ".." in clip:
                return self._send(400, {"error": "bad param"})
            return self._send(200, {"batch": batch, "clip": clip,
                                    "frames": _real_clip_frames(clip, batch)})

        # 取真实原图
        if u.path == "/api/real/image":
            batch = (q.get("batch") or [""])[0]
            clip = (q.get("clip") or [""])[0]
            stem = (q.get("stem") or [""])[0]
            if (not batch or not clip or not stem or "/" in clip or
                    ".." in clip or "/" in stem or ".." in stem):
                return self._send(400, {"error": "bad param"})
            # REAL-IMAGE-MULTICAM: 支持多种相机目录
            fp = None
            _CAMS3 = ("cam_st_x2f_front", "cam_ld_x8d_front",
                      "cam_st_x8d_front", "cam_hk_x8b_front", "images")
            for _r in (REAL_ROOT, "/data/test_data/real_clips_extra"):
                for _cam in _CAMS3:
                    _c = os.path.join(_r, batch, clip, _cam,
                                      stem + ".jpg")
                    if os.path.isfile(_c):
                        fp = _c
                        break
                if fp:
                    break
            if fp is None:
                return self._send(404, {"error": "not found"})
            # IMG-RESIZE-MAX: 支持 max= 缩放，避免 3840 大图卡死
            _mx = 0
            try:
                _mx = int((q.get("max") or ["0"])[0] or 0)
            except Exception:
                _mx = 0
            data = None
            if _mx and _mx > 0:
                try:
                    import cv2 as _cv2
                    _im = _cv2.imread(fp)
                    if _im is not None and _im.shape[1] > _mx:
                        _sc = _mx / float(_im.shape[1])
                        _im = _cv2.resize(
                            _im, (int(_im.shape[1] * _sc),
                                 int(_im.shape[0] * _sc)),
                            interpolation=_cv2.INTER_AREA)
                        _ok, _buf = _cv2.imencode(
                            ".jpg", _im, [int(_cv2.IMWRITE_JPEG_QUALITY), 88])
                        if _ok:
                            data = _buf.tobytes()
                except Exception:
                    data = None
            if data is None:
                with open(fp, "rb") as f:
                    data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        # 取真实 3D 标注（含投影到 2D 的框）
        if u.path == "/api/real/labels":
            batch = (q.get("batch") or [""])[0]
            clip = (q.get("clip") or [""])[0]
            stem = (q.get("stem") or [""])[0]
            if not batch or not clip or not stem:
                return self._send(400, {"error": "bad param"})
            cp = os.path.join(REAL_ROOT, batch, clip)
            try:
                import sys as _sys
                if TOOLS_DIR not in _sys.path:
                    _sys.path.insert(0, TOOLS_DIR)
                from label_gate import load_frame_labels

                lab = load_frame_labels(cp, xml_path=stem + ".xml")
                if lab is None:
                    return self._send(404, {"error": "无标注"})

                objs = []
                for o in lab.get("objects") or []:
                    if o.get("delete"):
                        continue
                    objs.append({
                        "kind": o.get("kind"),
                        "tracking_id": o.get("tracking_id"),
                        "bbox2d": o.get("bbox2d"),
                        "in_front": bool(o.get("in_front")),
                        "depth_m": o.get("depth_m"),
                        "world": {"x": o.get("Wx"), "y": o.get("Wy"),
                                  "z": o.get("Wz")},
                        "size": {"L": o.get("L"), "W": o.get("W"),
                                 "H": o.get("H")},
                        "pts": o.get("pts"),
                    })
                cam = lab.get("camera") or {}
                return self._send(200, {
                    "clip": clip, "stem": stem,
                    "width": 1920, "height": 1080,
                    "camera": {k: cam.get(k) for k in
                               ("name", "fx", "fy", "cx", "cy")},
                    "objects": objs,
                    "n_total": len(objs),
                    "n_front": len([o for o in objs if o["in_front"]]),
                })
            except Exception as e:
                import traceback
                return self._send(500, {"error": str(e),
                                        "trace": traceback.format_exc()[-500:]})

        # 批量验证结果（已跑过的产出）
        if u.path == "/api/real/verify":
            vd = "/data/render/verify_batch"
            sp = os.path.join(vd, "summary.json")
            if not os.path.isfile(sp):
                return self._send(404, {"error": "还没跑过验证"})
            with open(sp, encoding="utf-8") as f:
                data = json.load(f)
            # 附上产出图列表
            imgs = []
            for dp, _, fs in os.walk(vd):
                for f in fs:
                    if f.endswith("_composed.jpg"):
                        rel = os.path.relpath(os.path.join(dp, f), vd)
                        imgs.append(rel)
            return self._send(200, {"summary": data, "images": sorted(imgs)[:80]})

        # 读取验证产出图
        if u.path == "/api/real/verify_image":
            rel = (q.get("path") or [""])[0]
            if not rel or ".." in rel:
                return self._send(400, {"error": "bad path"})
            fp = os.path.join("/data/render/verify_batch", rel)
            if not os.path.isfile(fp):
                return self._send(404, {"error": "not found"})
            with open(fp, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        # [REAL-PLACE] 读取点选放置产出图
        if u.path == "/api/real/place_image":
            job = (q.get("job") or [""])[0]
            name = (q.get("name") or [""])[0]
            if (not job or not name or "/" in job or ".." in job or
                    "/" in name or ".." in name):
                return self._send(400, {"error": "bad param"})
            sub = (q.get("sub") or [""])[0]
            if sub and ("/" in sub or ".." in sub):
                return self._send(400, {"error": "bad sub"})
            # RENDER-JOB-HYBRID-B: 兼容两个根目录
            fp = None
            for base in (os.path.join("/data/render", job),
                         os.path.join("/data/render_hybrid", job)):
                cand0 = (os.path.join(base, sub, name) if sub
                         else os.path.join(base, name))
                if os.path.isfile(cand0):
                    fp = cand0
                    break
                c2 = os.path.join(base, "frames", name)
                if os.path.isfile(c2):
                    fp = c2
                    break
            if not fp:
                return self._send(404, {"error": "not found"})
            with open(fp, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        # [CLIP-PLAY] 列出某 clip 的可播放帧（原图 + 仿真图成对）
        if u.path == "/api/real/clip_play":
            batch = (q.get("batch") or [""])[0]
            clip = (q.get("clip") or [""])[0]
            module = (q.get("module") or ["obj"])[0]
            if not batch or "/" in batch or ".." in batch:
                return self._send(400, {"error": "bad batch"})
            REAL_ROOT = "/data/test_data/real_clips"
            VD = "/data/render/verify_batch"

            # CLIPPLAY-MULTI-ROOT: 支持 real_clips + extra
            _ROOTS = (REAL_ROOT, "/data/test_data/real_clips_extra")
            _bp = None
            for _r in _ROOTS:
                _p = os.path.join(_r, batch)
                if os.path.isdir(_p):
                    _bp = _p
                    break
            if _bp is None:
                return self._send(404, {"error": "batch 不存在"})

            # CLIPPLAY-MULTICAM: 支持多种相机
            _CAMS5 = ("cam_st_x2f_front", "cam_ld_x8d_front",
                      "cam_st_x8d_front", "cam_hk_x8b_front", "images")
            def _cam_dir(base, cn):
                for _c in _CAMS5:
                    _p = os.path.join(base, cn, _c)
                    if os.path.isdir(_p):
                        return _p
                return None
            if clip:
                cd = _cam_dir(_bp, clip)
                if cd is None:
                    return self._send(404, {"error": "clip 不存在"})
            else:
                subs = []
                for d in os.listdir(_bp):
                    if os.path.isdir(os.path.join(_bp, d)) and _cam_dir(_bp, d):
                        subs.append(d)
                if not subs:
                    return self._send(404, {"error": "无 clip"})
                clip = sorted(subs)[0]
                cd = _cam_dir(_bp, clip)

            stems = sorted(os.path.splitext(f)[0]
                           for f in os.listdir(cd) if f.lower().endswith(".jpg"))

            # CLIP-PAIR-STRICT: 只保留有 obj_good 产出的帧
            _prod_frames = set()
            try:
                import glob as _g2
                for _f2 in _g2.glob("/data/render/obj_good/frames/*_sim.jpg"):
                    _prod_frames.add(os.path.basename(_f2)[:-8])  # 去掉 _sim.jpg
            except Exception:
                pass
            if _prod_frames:
                stems = [x for x in stems if x in _prod_frames]
                if not stems:
                    return self._send(404, {
                        "error": "该 clip 无仿真产出",
                        "batch": batch, "clip": clip})

            # OBJ-NEW-SOURCE: 多源查找（新流水线优先，回退旧数据）
            # OBJ-GOOD-SRC: 只读优质场景产出
            # V2-SOURCE: v2 优先
            _obj_dirs = []
            _ann_dirs = []
            for _cand in ("/data/render/obj_v3_10",
                          "/data/render/obj_good_v2",
                          "/data/render/obj_good"):
                if os.path.isdir(_cand + "/frames") and \
                        len(os.listdir(_cand + "/frames")) > 0:
                    _obj_dirs.append(_cand + "/frames")
                    _ann_dirs.append(_cand + "/annotations")
            # CLIPPLAY-HYBRID-SCAN: 纳入 render_hybrid 的 objtask
            import glob as _gh
            _hybrid = []
            for _cf in _gh.glob("/data/render_hybrid/*/config.json"):
                try:
                    with open(_cf, encoding="utf-8") as _fh:
                        _cd = json.load(_fh) or {}
                except Exception:
                    continue
                if _cd.get("batch") != batch:
                    continue
                _jobdir = os.path.dirname(_cf)
                # 只纳入当前 clip（或未指定 clip）
                if _cd.get("clip") and _cd.get("clip") != clip:
                    continue
                _hybrid.append(_jobdir)
            # hybrid 目录直接用 _obj.jpg，标注在 annotations/
            NEW_DIRS = _hybrid + _obj_dirs + [
                "/data/render/pipe_pc/frames",
                "/data/render/pipe3d/frames",
                "/data/render/pipe_obj/frames"]
            ANN_DIRS = [os.path.join(h, "annotations") for h in _hybrid] + \
                _ann_dirs + ["/data/render/pipe_pc/annotations",
                             "/data/render/pipe3d/annotations"]

            frames = []
            for st in stems:
                sim_p = None
                src_dir = None
                # 1) 新流水线
                for d in NEW_DIRS:
                    for suf in ("_obj.jpg", "_sim.jpg", "_composed.jpg",
                                "_gnd.jpg"):
                        cand = os.path.join(d, st + suf)
                        if os.path.isfile(cand):
                            sim_p = cand
                            src_dir = d
                            break
                    if sim_p:
                        break
                # 2) 回退旧 verify_batch
                if sim_p is None:
                    for suf in ("_composed.jpg", "_gnd.jpg", "_sim.jpg"):
                        cand = os.path.join(VD, batch, st + suf)
                        if os.path.isfile(cand):
                            sim_p = cand
                            break
                # SIM-FALLBACK: 无仿真产出时用原图（避免黑屏）
                if sim_p is None:
                    sim_url = "/api/real/image?batch=" + batch + \
                              "&clip=" + clip + "&stem=" + st
                    frames.append({
                        "stem": st,
                        "orig": sim_url,
                        "sim": sim_url,
                        "objects": [],
                        "n3d": 0,
                        "src": "orig_only",
                    })
                    continue

                # 3D 标注
                ann = None
                for ad in ANN_DIRS:
                    ap = os.path.join(ad, st + ".json")
                    if os.path.isfile(ap):
                        try:
                            with open(ap, encoding="utf-8") as f:
                                ann = json.load(f)
                        except Exception:
                            ann = None
                        break

                if src_dir and "render_hybrid" in src_dir:
                    # 直接在 job 目录里
                    job = os.path.basename(src_dir)
                    sim_url = ("/api/render_job_image?job=" + job +
                               "&name=" + os.path.basename(sim_p))
                elif src_dir and "render" in src_dir:
                    job = os.path.basename(os.path.dirname(src_dir))
                    sim_url = ("/api/render_job_image?job=" + job +
                               "&name=" + os.path.basename(sim_p))
                else:
                    sim_url = ("/api/real/verify_image?path=" +
                               batch + "/" + os.path.basename(sim_p))

                objs = (ann or {}).get("objects", [])
                frames.append({
                    "stem": st,
                    "orig": "/api/real/image?batch=" + batch +
                            "&clip=" + clip + "&stem=" + st,
                    "sim": sim_url,
                    "objects": objs,
                    "n3d": len(objs),
                    "src": "new" if src_dir else "old",
                })

            return self._send(200, {"batch": batch, "clip": clip,
                                    "module": module,
                                    "frames": frames,
                                    "n": len(frames)})

        # [OBJ-BBOX] 参数化任务逐帧合成标注
        if u.path == "/api/render_job_annotation":
            job = (q.get("job") or [""])[0]
            stem = (q.get("stem") or [""])[0]
            if (not job or not stem or "/" in job or ".." in job or
                    "/" in stem or ".." in stem):
                return self._send(400, {"error": "bad param"})
            # RENDER-JOB-HYBRID-B: 标注接口也兼容两个根目录
            ap = None
            for base in (os.path.join("/data/render", job),
                         os.path.join("/data/render_hybrid", job)):
                for cand in (
                        os.path.join(base, "annotations", stem + ".json"),
                        os.path.join(base, "frames", "annotations",
                                     stem + ".json")):
                    if os.path.isfile(cand):
                        ap = cand
                        break
                if ap:
                    break
            if ap is None:
                return self._send(404, {"error": "annotation not found"})
            try:
                with open(ap, encoding="utf-8") as f:
                    ann = json.load(f)
                # OBJ task images are undistorted; project real boxes separately
                # onto the original distorted image using this frame's calibration.
                if job.startswith(('objtask_', 'obj3d_')) and ann.get('frame') == stem:
                    try:
                        cfg_path = os.path.join(os.path.dirname(os.path.dirname(ap)), 'config.json')
                        with open(cfg_path, encoding='utf-8') as cf: cfg = json.load(cf)
                        if TOOLS_DIR not in sys.path: sys.path.insert(0, TOOLS_DIR)
                        from real_bbox_original import enrich
                        ann = enrich(ann, str(cfg.get('batch') or ''), str(cfg.get('clip') or ''), stem)
                    except Exception as exc:
                        ann['original_geometry'] = 'unavailable'
                        ann['original_geometry_error'] = str(exc)[:200]
                return self._send(200, ann)
            except Exception as e:
                return self._send(500, {"error": str(e)})

        # [PARAMS] 读取参数化产出图
        if u.path == "/api/render_job_image":
            job = (q.get("job") or [""])[0]
            name = (q.get("name") or [""])[0]
            if (not job or not name or "/" in job or ".." in job or
                    "/" in name or ".." in name):
                return self._send(400, {"error": "bad param"})
            sub = (q.get("sub") or [""])[0]
            if sub and ("/" in sub or ".." in sub):
                return self._send(400, {"error": "bad sub"})
            # RENDER-JOB-HYBRID-ROOT: 兼容 /data/render 与 /data/render_hybrid
            _bases = [os.path.join("/data/render", job),
                      os.path.join("/data/render_hybrid", job)]
            fp = None
            for base in _bases:
                cand = (os.path.join(base, sub, name) if sub
                        else os.path.join(base, name))
                if os.path.isfile(cand):
                    fp = cand
                    break
                if not sub:
                    _c = os.path.join(base, "frames", name)
                    if os.path.isfile(_c):
                        fp = _c
                        break
            if not fp:
                return self._send(404, {"error": "not found"})
            # IMG-RESIZE-MAX: 同样支持 max= 缩放
            _mx2 = 0
            try:
                _mx2 = int((q.get("max") or ["0"])[0] or 0)
            except Exception:
                _mx2 = 0
            data = None
            if _mx2 and _mx2 > 0:
                try:
                    import cv2 as _cv2b
                    _im2 = _cv2b.imread(fp)
                    if _im2 is not None and _im2.shape[1] > _mx2:
                        _sc2 = _mx2 / float(_im2.shape[1])
                        _im2 = _cv2b.resize(
                            _im2, (int(_im2.shape[1] * _sc2),
                                   int(_im2.shape[0] * _sc2)),
                            interpolation=_cv2b.INTER_AREA)
                        _ok2, _buf2 = _cv2b.imencode(
                            ".jpg", _im2, [int(_cv2b.IMWRITE_JPEG_QUALITY), 88])
                        if _ok2:
                            data = _buf2.tobytes()
                except Exception:
                    data = None
            if data is None:
                with open(fp, "rb") as f:
                    data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        # [CALIB] 列出底图（供标定面板选图）
        if u.path == "/api/tsr/bg_list":
            try:
                import glob as _g
                fs = sorted(_g.glob("/data/render_hybrid/bgs/*.jpg"))
                return self._send(200, {
                    "images": [{"idx": i, "name": os.path.basename(f),
                                "path": f} for i, f in enumerate(fs)]
                })
            except Exception as e:
                return self._send(500, {"error": str(e)})

        # [CALIB] 读取某张底图（供标定面板显示）
        if u.path == "/api/tsr/bg_image":
            idx = (q.get("id") or ["0"])[0]
            try:
                fs = sorted(__import__("glob").glob("/data/render_hybrid/bgs/*.jpg"))
                i = int(idx)
                if not (0 <= i < len(fs)):
                    return self._send(404, {"error": "index out of range"})
                with open(fs[i], "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as e:
                return self._send(500, {"error": str(e)})

        # [DETECT-JOBS] 列出所有检测任务
        if u.path == "/api/tsr/prescreen_result":
            tid = (q.get("task") or [""])[0]
            jf = "/tmp/prescreen_%s.json" % tid
            if not os.path.isfile(jf):
                return self._send(200, {"results": [], "total": 0, "hits": 0})
            with open(jf, "r", encoding="utf-8") as f:
                return self._send(200, json.load(f))

        if u.path == "/api/tsr/detect_jobs":
            jobs = []
            try:
                import glob as _g
                for fp in _g.glob("/tmp/prescreen_*.json"):
                    name = os.path.basename(fp)
                    if "prog" in name:
                        continue
                    jid = name[len("prescreen_"):-len(".json")]
                    try:
                        d = json.load(open(fp, encoding="utf-8"))
                    except Exception:
                        continue
                    jobs.append({
                        "job": jid,
                        "dir": d.get("dir", ""),
                        "total": d.get("total", 0),
                        "hits": d.get("hits", 0),
                        "usable": d.get("usable_hits", 0),
                        "mtime": int(os.path.getmtime(fp)),
                    })
                jobs.sort(key=lambda x: -x["mtime"])
                return self._send(200, {"jobs": jobs})
            except Exception as e:
                return self._send(200, {"jobs": [], "err": str(e)})

        # [JOB-DETAIL] 单个任务的完整信息
        if u.path == "/api/tsr/job_detail":
            job = (q.get("job") or [""])[0]
            if not job:
                return self._send(400, {"error": "缺少 job"})

            # 1) 检测任务：读 /tmp/prescreen_<id>.json
            pj = "/tmp/prescreen_%s.json" % job
            if os.path.isfile(pj):
                try:
                    with open(pj, "r", encoding="utf-8") as f:
                        pd = json.load(f)
                    # 进度文件补参数
                    pf0 = "/tmp/prescreen_prog_%s.json" % job
                    root0 = ""
                    if os.path.isfile(pf0):
                        try:
                            pg0 = json.load(open(pf0, encoding="utf-8"))
                        except Exception:
                            pg0 = {}
                    else:
                        pg0 = {}
                    return self._send(200, {
                        "job": job,
                        "type": "detect",
                        "module": "TSR",
                        "country": "",
                        "source_root": pd.get("dir", "") or root0,
                        "n": pd.get("usable_hits", 0),
                        "total": pd.get("total", 0),
                        "hits": pd.get("hits", 0),
                        "done": pd.get("total", 0),
                    })
                except Exception as e:
                    pass

            # 2) 仿真/批量任务：找目录（含 OBJ/GND）
            cands = [
                os.path.join(TSR_OUT_DIR, "walk_" + job),
                os.path.join(TSR_OUT_DIR, job),
                os.path.join("/mnt/Data_Platform/zhj_datasim", job),
                os.path.join("/data/render_hybrid", job),   # OBJ / GND
            ]
            dp = next((c for c in cands if os.path.isdir(c)), "")
            if not dp:
                return self._send(404, {"error": "任务不存在: " + job})

            info = {"job": job, "dir": dp}
            ad = os.path.join(dp, "annotations")
            if os.path.isdir(ad):
                info["n"] = len([f for f in os.listdir(ad) if f.endswith(".json")])

            # OBJ/GND: 从 summary.json 读模块与源信息
            sf0 = os.path.join(dp, "summary.json")
            if os.path.isfile(sf0):
                try:
                    with open(sf0, "r", encoding="utf-8") as _f:
                        _sm = json.load(_f)
                    for k in ("module", "type", "bg", "layout", "lines",
                              "frames", "source_root", "country"):
                        if k in _sm:
                            info[k] = _sm[k]
                    if _sm.get("source_root"):
                        info["root"] = _sm["source_root"]
                except Exception:
                    pass

            # 统计产出图
            if not info.get("n"):
                _jpgs = [f for f in os.listdir(dp) if f.endswith(".jpg")
                         and not f.startswith("contact")]
                info["n"] = len(_jpgs)

            # 从 summary.json 读参数
            sf = os.path.join(dp, "summary.json")
            if os.path.isfile(sf):
                try:
                    with open(sf, "r", encoding="utf-8") as f:
                        sd = json.load(f)
                    for k in ("module", "country", "source_root", "pin_class", "pin_material",
                              "min_sign", "conf", "limit", "type", "from_detect"):
                        if k in sd:
                            info[k] = sd[k]
                    if "source_root" in sd:
                        info["root"] = sd["source_root"]
                except Exception:
                    pass

            # 从任一标注读 source_path
            if os.path.isdir(ad):
                js = [x for x in os.listdir(ad) if x.endswith(".json")]
                if js:
                    try:
                        with open(os.path.join(ad, js[0]), "r", encoding="utf-8") as f:
                            a = json.load(f)
                        info["source_path"] = a.get("source_path", "")
                        if not info.get("country"):
                            # 从任务名推断
                            if "_" in job:
                                info["country"] = job.split("_")[0]
                        if a.get("objects"):
                            ob = a["objects"][0]
                            info.setdefault("pin_class", ob.get("class") or "")
                            info.setdefault("renderer", ob.get("renderer") or "")
                    except Exception:
                        pass

            # 从 progress 读参数
            pf = os.path.join(dp, "progress.json")
            if not os.path.isfile(pf):
                pf = os.path.join(TSR_OUT_DIR, "walk_%s_progress.json" % job)
            if os.path.isfile(pf):
                try:
                    with open(pf, "r", encoding="utf-8") as f:
                        pd = json.load(f)
                    info["total"] = pd.get("total")
                    info["done"] = pd.get("done")
                    info["ok"] = pd.get("ok")
                except Exception:
                    pass

            return self._send(200, info)

        if u.path == "/api/tsr/job_list":
            jobs = []
            try:
                for name in os.listdir(TSR_OUT_DIR):
                    dp = os.path.join(TSR_OUT_DIR, name)
                    ad = os.path.join(dp, "annotations")
                    if os.path.isdir(ad):
                        n = len(os.listdir(ad))
                        if n:
                            tag = name[5:] if name.startswith("walk_") else name
                            _mod = ""
                            _sf = os.path.join(dp, "summary.json")
                            if os.path.isfile(_sf):
                                try:
                                    _mod = json.load(open(_sf, encoding="utf-8")).get("module", "")
                                except Exception:
                                    pass
                            jobs.append({"job": tag, "id": tag, "label": tag,
                                         "module": _mod or "TSR",
                                         "n": n, "mtime": int(os.path.getmtime(dp))})
                # 也扫共享导出目录
                ed = "/mnt/Data_Platform/zhj_datasim"
                if os.path.isdir(ed):
                    for name in os.listdir(ed):
                        dp = os.path.join(ed, name)
                        ad = os.path.join(dp, "annotations")
                        if os.path.isdir(ad):
                            n = len(os.listdir(ad))
                            if n:
                                jobs.append({"job": name, "id": name, "label": name,
                                             "n": n, "mtime": int(os.path.getmtime(dp)),
                                             "export": True, "path": dp})
                jobs.sort(key=lambda x: -x["mtime"])
                return self._send(200, {"jobs": jobs})
            except Exception as e:
                return self._send(200, {"jobs": [], "err": str(e)})

        if u.path == "/api/tsr/latest_job":
            try:
                cands = []
                for name in os.listdir(TSR_OUT_DIR):
                    if not name.startswith("walk_"):
                        continue
                    dp = os.path.join(TSR_OUT_DIR, name)
                    ad = os.path.join(dp, "annotations")
                    if os.path.isdir(ad) and os.listdir(ad):
                        cands.append((os.path.getmtime(dp), name[5:]))
                cands.sort(reverse=True)
                if not cands:
                    return self._send(200, {"job": "", "n": 0})
                job = cands[0][1]
                ad = os.path.join(TSR_OUT_DIR, "walk_" + job, "annotations")
                return self._send(200, {"job": job, "n": len(os.listdir(ad))})
            except Exception as e:
                return self._send(200, {"job": "", "n": 0, "err": str(e)})

        if u.path == "/api/tsr/prescreen_thumb":
            tid = (q.get("task") or [""])[0]
            name = os.path.basename((q.get("name") or [""])[0])
            fp = "/tmp/prescreen_thumbs_%s/%s" % (tid, name)
            if not os.path.isfile(fp):
                return self._send(404, {"error": "not found"})
            with open(fp, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=600")
            self.end_headers()
            self.wfile.write(data)
            return

        if u.path == "/api/tsr/walk_jobs":
            import glob as _g
            # 进程存活状态
            _alive = False
            try:
                _r = subprocess.run(["pgrep", "-f", "batch_walk.py"],
                                    capture_output=True, text=True, timeout=3)
                _alive = bool((_r.stdout or "").strip())
            except Exception:
                _alive = True
            pat = os.path.join(TSR_OUT_DIR, "walk_*_progress.json")
            jobs = []
            for fp in sorted(_g.glob(pat),
                             key=lambda x: os.path.getmtime(x), reverse=True)[:20]:
                tag = os.path.basename(fp).replace("walk_", "").replace("_progress.json", "")
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        d = json.load(f)
                except Exception:
                    d = {}
                d["job"] = tag
                d["mtime"] = int(os.path.getmtime(fp))
                d["alive"] = _alive
                jobs.append(d)
            return self._send(200, {"jobs": jobs})

        # [LIBRARY-UPLOAD] 素材上传（三类：tsr / obj / gnd）
        if u.path == "/api/library/upload":
            import cgi, uuid
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("multipart/form-data"):
                return self._send(400, {"error": "需要 multipart/form-data"})
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                    environ={"REQUEST_METHOD": "POST",
                                             "CONTENT_TYPE": ctype})
            if "file" not in form:
                return self._send(400, {"error": "缺少 file 字段"})
            item = form["file"]
            fn = os.path.basename(item.filename or "unknown.png")
            cat = (form.getvalue("category") or "tsr").strip().lower()
            if cat not in ("tsr", "obj", "gnd"):
                return self._send(400, {"error": "category 必须是 tsr/obj/gnd"})
            if not fn.lower().endswith(".png"):
                return self._send(400, {"error": "只接受 PNG"})
            blob = item.file.read()
            if len(blob) > 600 * 1024:
                return self._send(400, {"error": "文件超过 600KB"})
            # 校验是否为合法 PNG 且带 alpha
            if not blob.startswith(b"\x89PNG"):
                return self._send(400, {"error": "不是合法 PNG"})
            base = {"tsr": "/data/tsr_library_v2/_uploaded",
                    "obj": "/data/obj_library",
                    "gnd": "/data/gnd_library"}[cat]
            os.makedirs(base, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fa5]", "_", fn)
            dest = os.path.join(base, safe)
            n = 1
            while os.path.exists(dest):
                stem, ext = os.path.splitext(safe)
                dest = os.path.join(base, "%s_%d%s" % (stem, n, ext))
                n += 1
            with open(dest, "wb") as f:
                f.write(blob)
            rel = os.path.relpath(dest, base)
            return self._send(200, {"ok": True, "file": dest,
                                    "url": "/api/library/file?cat=%s&name=%s"
                                           % (cat, quote(safe))})

        # [LIBRARY-FILE] 取已上传素材
        if u.path == "/api/library/file":
            cat = (q.get("cat") or ["tsr"])[0].lower()
            name = (q.get("name") or [""])[0]
            if cat not in ("tsr", "obj", "gnd"):
                return self._send(400, {"error": "category 错误"})
            if not name or "/" in name or "\\" in name or ".." in name:
                return self._send(400, {"error": "文件名错误"})
            base = {"tsr": "/data/tsr_library_v2/_uploaded",
                    "obj": "/data/obj_library",
                    "gnd": "/data/gnd_library"}[cat]
            fp = os.path.join(base, name)
            if not os.path.isfile(fp):
                return self._send(404, {"error": "文件不存在"})
            with open(fp, "rb") as f:
                return self._send(200, f.read(), "image/png")

        # [STATIC] 前端依赖（tailwind / chart.js / tabler-icons）
        if u.path.startswith("/static/"):
            import mimetypes
            name = os.path.basename(u.path[len("/static/"):])
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
                return self._send(403, {"error": "forbidden"})
            fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "static", name)
            if not os.path.isfile(fp):
                return self._send(404, {"error": "not found: " + name})
            ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
            with open(fp, "rb") as f:
                blob = f.read()
            return self._send(200, blob, ctype)
        if u.path in ("/obj", "/hybrid", "/bev"):
            fp = os.path.join(BASE, "obj_new.html")
            try:
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(404, {"error": "obj_new.html 缺失"})
        if u.path == "/gnd":
            with open(os.path.join(BASE, "obj.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        status = re.fullmatch(r"/api/(obj|gnd)/status", u.path)
        if status:
            return self._send(200, smoke_status(status.group(1)))
        result = re.fullmatch(r"/api/(obj|gnd)/result/([^/]+)/(sheet|annotations|scene)", u.path)
        if result:
            mode, run, kind = result.groups()
            artifact = smoke_artifact(run, kind) if run.startswith(mode + "_") else None
            if artifact:
                with open(artifact, "rb") as f:
                    return self._send(200, f.read(), "image/jpeg" if kind == "sheet" else "application/json")
            return self._send(404, {"error": "结果文件不存在"})
        if u.path in ("/", "/index.html"):
            # 每次读最新文件 + 禁止缓存，避免改了页面还要手动加 ?v=
            try:
                with open(os.path.join(BASE, "index.html"), encoding="utf-8") as _f:
                    _page = _f.read()
            except Exception:
                _page = HTML_PAGE
            body = _page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)
            return
        # [TSR-API] 提供 tsr.html
        if u.path in ("/tsr", "/tsr.html"):
            fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tsr.html")
            try:
                with open(fp, "rb") as f:
                    blob = f.read()
            except OSError:
                return self._send(404, {"error": "tsr.html 缺失"})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(blob)
            return
        # [HYBRID-SQLITE] 任务状态接口
        if u.path == "/hybrid":
            fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hybrid.html")
            try:
                with open(fp, "rb") as f:
                    blob = f.read()
            except OSError:
                return self._send(404, {"error": "hybrid.html 缺失"})
            return self._send(200, blob, "text/html; charset=utf-8")
        if u.path == "/api/hybrid/jobs":
            try:
                limit = max(1, min(200, int((q.get("limit") or ["50"])[0])))
                offset = max(0, int((q.get("offset") or ["0"])[0]))
            except ValueError:
                return self._send(400, {"error": "invalid pagination"})
            return self._send(200, hybrid_jobs(limit, offset))
        if u.path == "/api/hybrid/job":
            jid = (q.get("job_id") or [""])[0]
            if not jid:
                return self._send(400, {"error": "job_id 必填"})
            return self._send(200, hybrid_job(jid))
        if u.path == "/api/hybrid/frames":
            jid = (q.get("job_id") or [""])[0]
            if not jid:
                return self._send(400, {"error": "job_id 必填"})
            st = (q.get("status") or [""])[0]
            try:
                limit = max(1, min(500, int((q.get("limit") or ["100"])[0])))
                offset = max(0, int((q.get("offset") or ["0"])[0]))
            except ValueError:
                return self._send(400, {"error": "invalid pagination"})
            return self._send(200, hybrid_frames(jid, st, limit, offset))
        if u.path == "/api/hybrid/frame":
            jid = (q.get("job_id") or [""])[0]
            fid = (q.get("frame_id") or [""])[0]
            if not jid or not fid:
                return self._send(400, {"error": "job_id / frame_id 必填"})
            return self._send(200, hybrid_frame(jid, fid))
        if u.path == "/api/hybrid/artifact":
            p = (q.get("path") or [""])[0]
            blob, kind = hybrid_artifact(p)
            if blob is None:
                return self._send(404, {"error": kind})
            ct = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                  "png": "image/png", "json": "application/json"}.get(
                      kind.strip("."), "application/octet-stream")
            return self._send(200, blob, ct)

        if u.path == "/api/status":
            return self._send(200, probe_status())
        if u.path == "/api/config":
            return self._send(200, load_config())
        # [SYSINFO] 实时 GPU / 内存 / CPU 状态
        if u.path == "/api/sysinfo":
            info = {"ok": True}
            # GPU
            try:
                r = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                line = (r.stdout or "").strip().split("\n")[0]
                if line:
                    parts = [x.strip() for x in line.split(",")]
                    info["gpu"] = {
                        "name": parts[0],
                        "used_mb": float(parts[1]),
                        "total_mb": float(parts[2]),
                        "utilization": float(parts[3]),
                        "temp": float(parts[4]) if len(parts) > 4 else None,
                    }
            except Exception as e:
                info["gpu"] = None
                info["gpu_error"] = str(e)

            # 内存
            try:
                with open("/proc/meminfo") as f:
                    mem = {}
                    for ln in f:
                        k, v = ln.split(":", 1)
                        mem[k.strip()] = int(v.strip().split()[0])
                total = mem.get("MemTotal", 0) / 1024.0
                avail = mem.get("MemAvailable", 0) / 1024.0
                info["mem"] = {
                    "total_mb": total,
                    "used_mb": max(0.0, total - avail),
                    "avail_mb": avail,
                }
            except Exception as e:
                info["mem"] = None

            # CPU：型号 + 负载 + 利用率
            try:
                info["load"] = [round(x, 2) for x in os.getloadavg()]
                info["cpu_count"] = os.cpu_count()
                # CPU 型号
                cpu_name = ""
                try:
                    with open("/proc/cpuinfo") as f:
                        for ln in f:
                            if ln.startswith("model name"):
                                cpu_name = ln.split(":", 1)[1].strip()
                                break
                except Exception:
                    pass
                info["cpu"] = {
                    "name": cpu_name,
                    "count": os.cpu_count(),
                    "load": info["load"],
                    "usage": round(min(100.0, info["load"][0] /
                                       max(1, os.cpu_count()) * 100), 1),
                }
            except Exception:
                info["cpu"] = None

            # 内存条型号（DMdecode）
            try:
                r2 = subprocess.run(
                    ["dmidecode", "-t", "17"], capture_output=True, text=True, timeout=5)
                names = []
                for ln in (r2.stdout or "").split("\n"):
                    ln = ln.strip()
                    if ln.startswith("Size:") and "MB" in ln or ln.startswith("Size:") and "GB" in ln:
                        pass
                # 简化：只取第一条型号
                for ln in (r2.stdout or "").split("\n"):
                    if "Part Number:" in ln:
                        pn = ln.split(":", 1)[1].strip()
                        if pn and pn != "Unknown" and pn not in names:
                            names.append(pn)
                info["mem_parts"] = names[:4]
            except Exception:
                info["mem_parts"] = []

            return self._send(200, info)

        if u.path == "/api/metrics":
            try:
                minutes = max(1, min(1440, int((q.get("minutes") or ["10"])[0])))
            except ValueError:
                return self._send(400, {"error": "invalid minutes"})
            return self._send(200, read_metrics(minutes))
        if u.path == "/api/dataset_stats":
            return self._send(200, read_dataset_stats())
        if u.path == "/api/assets":
            content = os.path.join(load_config()["project_dir"], "Content")
            return self._send(200, {"images": list_images(),
                                    "map": "/Game/Maps/codex_validation",
                                    "map_exists": os.path.isfile(os.path.join(content, "Maps", "codex_validation.umap")),
                                    "materials": [
                                        {"path": "/Game/Validation/M_LinearDepth",
                                         "exists": os.path.isfile(os.path.join(content, "Validation", "M_LinearDepth.uasset"))},
                                        {"path": "/Game/Validation/M_SemanticMask",
                                         "exists": os.path.isfile(os.path.join(content, "Validation", "M_SemanticMask.uasset"))}]})
        if u.path == "/api/preview/image":
            path = os.path.join(load_config()["project_dir"], "Saved", "web_preview.png")
            try:
                with open(path, "rb") as f:
                    return self._send(200, f.read(), "image/png")
            except OSError:
                return self._send(404, {"error": "尚无预览图"})
        if u.path == "/api/asset/image":
            name = (q.get("name") or [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:png|jpg|jpeg)", name, re.I):
                return self._send(400, {"error": "invalid image name"})
            try:
                with open(os.path.join(image_inbox(), name), "rb") as f:
                    blob = f.read()
            except OSError:
                return self._send(404, {"error": "image not found"})
            return self._send(200, blob, "image/png" if name.lower().endswith(".png") else "image/jpeg")

        # [TSR-API] 素材库概况
        if u.path == "/api/tsr/library":
            root = load_config().get("tsr_library") or DEFAULT_TSR_LIBRARY
            items = []
            if os.path.isdir(root):
                for name in sorted(os.listdir(root)):
                    d = os.path.join(root, name)
                    if not os.path.isdir(d) or name.startswith("_"):
                        continue
                    files = [f for f in os.listdir(d)
                             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    if not files:
                        continue
                    classes = sorted(set(os.path.splitext(f)[0].split("_v")[0]
                                         for f in files))
                    items.append({"country": name, "count": len(files),
                                  "class_count": len(classes),
                                  "classes": classes})
            return self._send(200, {"root": root, "countries": items,
                                    "total_countries": len(items),
                                    "total_assets": sum(i["count"] for i in items)})

        # [TSR-API] 素材缩略图
        if u.path == "/api/tsr/library/thumb":
            country = (q.get("country") or [""])[0]
            cls = (q.get("class") or [""])[0]
            root = load_config().get("tsr_library") or DEFAULT_TSR_LIBRARY
            d = os.path.join(root, _tsr_slug(country))
            target = None
            if os.path.isdir(d) and cls:
                for f in sorted(os.listdir(d)):
                    if os.path.splitext(f)[0].split("_v")[0] == cls and \
                            f.lower().endswith((".png", ".jpg", ".jpeg")):
                        target = os.path.join(d, f)
                        break
            if not target or not os.path.isfile(target):
                return self._send(404, {"error": "asset not found"})
            return self._send_image(max_w=140)(target)

        # [TSR-API] 某类别的全部素材（小图列表用）
        if u.path == "/api/tsr/library/class":
            country = (q.get("country") or [""])[0]
            cls = (q.get("class") or [""])[0]
            root = load_config().get("tsr_library") or DEFAULT_TSR_LIBRARY
            d = os.path.join(root, _tsr_slug(country))
            names = []
            if os.path.isdir(d) and cls:
                names = sorted(f for f in os.listdir(d)
                               if os.path.splitext(f)[0].split("_v")[0] == cls
                               and f.lower().endswith((".png", ".jpg", ".jpeg")))
            return self._send(200, {"country": country, "class": cls, "files": names})

        # [TSR-API] 实拍图清单
        if u.path == "/api/tsr/images":
            d = normalize_net_path((q.get("dir") or [""])[0])
            limit = int((q.get("limit") or ["30"])[0])
            offset = int((q.get("offset") or ["0"])[0])
            if not d or not os.path.isdir(d):
                return self._send(400, {"error": "目录不存在", "dir": d})
            files = sorted(f for f in os.listdir(d)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            return self._send(200, {"dir": d, "total": len(files), "offset": offset,
                                    "images": files[offset:offset + limit]})

        # [TSR-API] 单张实拍图
        if u.path == "/api/tsr/src":
            # 直接按绝对路径发原图（检测结果预览用）
            pabs = (q.get("path") or [""])[0]
            if pabs:
                pabs = normalize_net_path(pabs.strip())
                if not os.path.isfile(pabs):
                    return self._send(404, {"error": "图片不存在: " + pabs})
                self._send_image(int((q.get("max") or ["1600"])[0]))(pabs)
                return
            # 批量产出的原图：按 annotations 里记的 source_path 读取
            d = (q.get("dir") or [""])[0]
            stem = (q.get("stem") or [""])[0]
            maxw = int((q.get("max") or ["900"])[0])
            ap = os.path.join(d, "annotations", stem + ".json")
            if not os.path.isfile(ap):
                return self._send(404, {"error": "标注不存在"})
            with open(ap, "r", encoding="utf-8") as f:
                a = json.load(f)
            # bbox=1：只返回标注信息（前端用来定位标牌），不发图片
            if (q.get("bbox") or [""])[0] == "1":
                return self._send(200, {
                    "objects": a.get("objects", []),
                    "width": a.get("width", 1920),
                    "height": a.get("height", 1080),
                    "source_image": a.get("source_image", ""),
                })
            sp = a.get("source_path", "")
            # SRC-PATH-VALIDATE: source_path 可能指向已删除的临时目录
            # （batch 跑完 /tmp 被清），必须校验存在性
            if sp and not os.path.isfile(sp):
                sp = ""
            if not sp:
                # TSR-SRC-FALLBACK: 只有 source_image（文件名），反查真实目录
                si = a.get("source_image", "") or ""
                if si:
                    # 格式 <批次>__<帧名>.jpg 或直接 <帧名>.jpg
                    frame = si.split("__")[-1] if "__" in si else si
                    # ① 同目录下找
                    _c1 = os.path.join(d, si)
                    if os.path.isfile(_c1):
                        sp = _c1
                    if not sp:
                        _c2 = os.path.join(d, frame)
                        if os.path.isfile(_c2):
                            sp = _c2
                    # ② 到真实数据目录找
                    if not sp:
                        import glob as _gg
                        for _root in ("/data/test_data/real_clips_extra",
                                      "/data/test_data/real_clips",
                                      "/data/test_data/gnd_train"):
                            _hits = _gg.glob(
                                os.path.join(_root, "*", "*",
                                             "cam_st_x2f_front", frame))
                            if _hits:
                                sp = _hits[0]
                                break
            if not sp or not os.path.isfile(sp):
                return self._send(404, {"error": "原图不存在"})
            return self._send_image(max_w=maxw)(sp)

        if u.path == "/api/tsr/image":
            d = (q.get("dir") or [""])[0]
            f = (q.get("file") or [""])[0]
            if not d or not f or "/" in f or "\\" in f or ".." in f:
                return self._send(400, {"error": "参数错误"})
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                return self._send(404, {"error": "not found"})
            maxw = int((q.get("max") or ["900"])[0])
            return self._send_image(max_w=maxw)(p)

        # [PRESCREEN] 查进度 / 取结果
        if u.path == "/api/tsr/walk_progress":
            tag = (q.get("job") or [""])[0]
            # 三种布局：根目录 / 子目录 / 共享导出目录
            cands = [
                os.path.join(TSR_OUT_DIR, "walk_%s_progress.json" % tag),
                os.path.join(TSR_OUT_DIR, "walk_" + tag, "walk_%s_progress.json" % tag),
            ]
            if "_" in tag:
                cc = tag.split("_")[0]
                for ed in (str((q.get("export") or [""])[0]),
                           "/mnt/Data_Platform/zhj_datasim"):
                    if ed:
                        cands.append(os.path.join(normalize_net_path(ed), tag, "progress.json"))
            prog = next((c for c in cands if os.path.isfile(c)), "")
            if not prog:
                return self._send(200, {"phase": "unknown", "alive": False})
            with open(prog, "r", encoding="utf-8") as f:
                d = json.load(f)
            # 判断进程是否还活着
            alive = False
            try:
                r = subprocess.run(["pgrep", "-f", "batch_walk.py|render_selected.py"],
                                   capture_output=True, text=True, timeout=3)
                alive = bool((r.stdout or "").strip())
            except Exception:
                alive = True
            d["alive"] = alive
            logf = os.path.join(TSR_OUT_DIR, "walk_%s.log" % tag)
            if os.path.isfile(logf):
                try:
                    with open(logf, "r", encoding="utf-8", errors="replace") as f:
                        d["tail"] = "".join(f.readlines()[-6:])
                except Exception:
                    pass
            return self._send(200, d)

        if u.path == "/api/tsr/prescreen_progress":
            tid = (q.get("task") or [""])[0]
            if not re.fullmatch(r"[a-f0-9]{10}", tid):
                return self._send(400, {"error": "bad task"})
            prog_json = "/tmp/prescreen_prog_%s.json" % tid
            out_json = "/tmp/prescreen_%s.json" % tid
            info = {"ok": True, "finished": False, "done": 0, "total": 0}
            if os.path.isfile(prog_json):
                try:
                    with open(prog_json, "r", encoding="utf-8") as fh:
                        info.update(json.load(fh))
                except Exception:
                    pass
            if info.get("finished") and os.path.isfile(out_json):
                try:
                    with open(out_json, "r", encoding="utf-8") as fh:
                        full = json.load(fh)
                    info["results"] = [r for r in full.get("results", [])
                                       if r.get("usable", 0) > 0]
                    info["total_files"] = full.get("total", 0)
                except Exception as e:
                    info["error"] = str(e)[:150]
            return self._send(200, info)

        # [TSR-API] 替换结果图
        if u.path == "/api/tsr/gallery":
            # 批量产出的相册：返回文件名 + 原图 + 合成图 URL
            d = (q.get("dir") or [""])[0]
            limit = int((q.get("limit") or ["400"])[0])
            offset = max(0, int((q.get("offset") or ["0"])[0]))
            if not d:
                # 在多个产出根目录下找带 annotations 的目录，按修改时间取最新。
                # 注意：批量的 --out 可以直接指定 /data/render/xxx，不一定在 TSR_OUT_DIR 下，
                # 所以两个根目录都要扫。
                cands = []
                for root in (TSR_OUT_DIR, "/data/render"):
                    if not os.path.isdir(root):
                        continue
                    for x in os.listdir(root):
                        full = os.path.join(root, x)
                        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "annotations")):
                            cands.append((os.path.getmtime(full), full))
                cands.sort()
                walks = [c for c in cands if "/walk_" in c[1] + "/"]
                d = (walks or cands)[-1][1] if (walks or cands) else ""
            if not d or not os.path.isdir(d):
                return self._send(404, {"error": "目录不存在", "dir": d})
            ann_dir = os.path.join(d, "annotations")
            all_files = []
            if os.path.isdir(ann_dir):
                all_files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".json"))
            total_all = len(all_files)
            page_files = all_files[offset:offset + limit]
            items = []
            for fn in page_files:
                try:
                    with open(os.path.join(ann_dir, fn), "r", encoding="utf-8") as f:
                        a = json.load(f)
                except Exception:
                    continue
                items.append({
                    "stem": fn[:-5],
                    "composed": a.get("image", ""),
                    "source_name": a.get("source_image", ""),
                    "width": a.get("width"), "height": a.get("height"),
                    "country": "",
                    "objects": a.get("objects", []),
                })
            return self._send(200, {"dir": d, "count": len(items),
                                    "total": total_all, "offset": offset,
                                    "items": items})

        if u.path == "/api/tsr/result":
            f = (q.get("file") or [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.(?:jpg|png)", f, re.I):
                return self._send(400, {"error": "invalid"})
            selected_dir = (q.get("dir") or [""])[0]
            if selected_dir:
                root = os.path.realpath("/data/render")
                selected = os.path.realpath(selected_dir)
                if os.path.commonpath([root, selected]) != root:
                    return self._send(400, {"error": "目录不在渲染目录内"})
                p = os.path.realpath(os.path.join(selected, f))
                if os.path.dirname(p) != selected or not os.path.isfile(p):
                    return self._send(404, {"error": "结果文件不存在"})
            else:
                p = os.path.join(TSR_OUT_DIR, f)
            if not selected_dir and not os.path.isfile(p):
                # 批量产出可能直接落在 /data/render/<目录>/ 下，也要能取到
                alt = None
                rroot = "/data/render"
                if os.path.isdir(rroot):
                    for x in os.listdir(rroot):
                        cand = os.path.join(rroot, x, f)
                        if os.path.isfile(cand):
                            alt = cand
                            break
                if alt:
                    p = alt
                else:
                    return self._send(404, {"error": "not found"})
            with open(p, "rb") as fh:
                blob = fh.read()
            # ⚠️ 必须 no-store：预览图文件名固定（xxx_preview.jpg），
            #    浏览器会缓存旧图，导致改了参数重新生成后看到的还是旧结果
            self.send_response(200)
            self.send_header("Content-Type", "image/png" if p.endswith(".png") else "image/jpeg")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(blob)
            return
        if u.path == "/api/job":
            jid = (q.get("id") or [""])[0]
            with JOBS_LOCK:
                rec = JOBS.get(jid)
            if not rec:
                return self._send(404, {"error": "no job"})
            return self._send(200, dict(rec, out=tail(rec["log"], 300)))
        if u.path == "/api/log":
            jid = (q.get("job") or [""])[0]
            if jid:
                with JOBS_LOCK:
                    rec = JOBS.get(jid)
                if not rec:
                    return self._send(404, {"error": "no job"})
                try:
                    offset = max(0, int((q.get("offset") or ["0"])[0]))
                except ValueError:
                    return self._send(400, {"error": "invalid offset"})
                try:
                    with open(rec["log"], "rb") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        offset = min(offset, size)
                        f.seek(offset)
                        blob = f.read(65536)
                except OSError:
                    blob = b""
                return self._send(200, {"job": jid, "out": blob.decode("utf-8", "replace"),
                                        "offset": offset + len(blob)})
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
        if u.path == "/api/overlay":
            # 只允许读配置目录下这一张固定文件，不接受任意路径
            p = os.path.join(load_config()["overlay_dir"], "contact_sheet.jpg")
            if not os.path.exists(p):
                return self._send(404, {"error": "还没有抽检图"})
            try:
                with open(p, "rb") as f:
                    blob = f.read()
            except Exception as e:
                return self._send(500, {"error": "读取失败: %s" % e})
            return self._send(200, blob, "image/jpeg")
        # -------- GND DATA API --------
        if u.path == "/api/gnd/clips":
            root = "/data/test_data/gnd_train"
            out = []
            if os.path.isdir(root):
                for vd in sorted(os.listdir(root)):
                    vp = os.path.join(root, vd)
                    if not os.path.isdir(vp):
                        continue
                    for cl in sorted(os.listdir(vp)):
                        cp = os.path.join(vp, cl)
                        imgd = os.path.join(cp, "img_ori")
                        if not os.path.isdir(imgd):
                            continue
                        nf = len([f for f in os.listdir(imgd)
                                  if f.endswith(".jpg")])
                        out.append({"batch": vd, "clip": cl,
                                    "n_frames": nf})
            return self._send(200, {"ok": True, "clips": out,
                                    "count": len(out)})

        if u.path == "/api/gnd/frames":
            batch = (q.get("batch") or [""])[0]
            clip = (q.get("clip") or [""])[0]
            if not (batch and clip):
                return self._send(400, {"ok": False, "error": "missing"})
            imgd = os.path.join("/data/test_data/gnd_train", batch, clip,
                                "img_ori")
            if not os.path.isdir(imgd):
                return self._send(404, {"ok": False, "error": "no clip"})
            # GND-MULTI-SRC: 候选目录列表（逐帧查找，不锁死一个目录）
            _src_dirs = []
            # GND-PRIORITY-V3: gnd_clean_v3（4A全量）优先
            for _cand in ("/data/render/gnd_clean_v3",
                          "/data/render/gnd_v3_10",
                          "/data/render/gnd_clean_v2",
                          "/data/render/gnd_clean"):
                if os.path.isdir(_cand + "/frames"):
                    _src_dirs.append((_cand + "/frames",
                                      os.path.basename(_cand)))
            if not _src_dirs:
                _src_dirs = [("/data/render/gnd_clean/frames", "gnd_clean")]
            prod = _src_dirs[0][0]
            job_name = _src_dirs[0][1]

            def _find_src(stem, suffix):
                """在全部候选目录（含 clean/ 子目录）里找 stem+suffix。"""
                for _d, _j in _src_dirs:
                    # 主目录
                    if os.path.isfile(os.path.join(_d, stem + suffix)):
                        return _j, stem + suffix
                    # clean/ 子目录
                    _sub = os.path.join(os.path.dirname(_d), "clean",
                                        stem + suffix)
                    if os.path.isfile(_sub):
                        return _j, "clean/" + stem + suffix
                return None, None
            frames = []
            for f in sorted(os.listdir(imgd)):
                if not f.endswith(".jpg"):
                    continue
                stem = os.path.splitext(f)[0]
                # GND-FRAME-FIND: 逐帧在所有候选目录里查
                _j_sim, _n_sim = _find_src(stem, "_sim.jpg")
                _j_seg, _n_seg = _find_src(stem, "_seg.png")
                _j_gt, _n_gt = _find_src(stem, "_gt.png")
                _j_clean, _n_clean = _find_src(stem, "_clean.jpg")
                frames.append({
                    "stem": stem,
                    "orig": "/api/gnd/image?batch=%s&clip=%s&name=%s"
                            % (batch, clip, f),
                    "sim": ("/api/render_job_image?job=%s&name=%s"
                            % (_j_sim, _n_sim)) if _j_sim else None,
                    "seg": ("/api/render_job_image?job=%s&name=%s"
                            % (_j_seg, _n_seg)) if _j_seg else None,
                    "gt": ("/api/render_job_image?job=%s&name=%s"
                           % (_j_gt, _n_gt)) if _j_gt else None,
                    "clean": ("/api/render_job_image?job=%s&name=%s"
                              % (_j_clean, (_n_clean[6:] if _n_clean and _n_clean.startswith("clean/") else _n_clean))
                              + ("&sub=clean" if _n_clean and _n_clean.startswith("clean/") else "")
                              ) if _j_clean else None,
                })
            return self._send(200, {"ok": True, "batch": batch,
                                    "clip": clip, "frames": frames,
                                    "n": len(frames)})

        if u.path == "/api/gnd/image":
            batch = (q.get("batch") or [""])[0]
            clip = (q.get("clip") or [""])[0]
            name = (q.get("name") or [""])[0]
            if (not batch or not clip or not name or "/" in name
                    or ".." in name or "/" in clip or ".." in clip):
                return self._send(400, {"error": "bad param"})
            fp = os.path.join("/data/test_data/gnd_train", batch, clip,
                              "img_ori", name)
            if not os.path.isfile(fp):
                return self._send(404, {"error": "not found"})
            with open(fp, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        # -------- ASSETS LIB --------
        if u.path == "/api/assets/list":
            module = (q.get("module") or ["all"])[0]
            out = {"obj": [], "gnd": [], "tsr": []}

            def scan_obj():
                items = []
                # ASSETS-GLOBAL: 新 3D 素材库（按国家/类别分层）
                _groot = "/data/assets_global/OBJ_3D"
                # CANON-ASSETS: 优先读规范化车辆 + 障碍物索引
                try:
                    _cidx = os.path.join(_groot, "asset_index_obj3d.json")
                    if os.path.isfile(_cidx):
                        _ci = json.load(open(_cidx, encoding="utf-8"))
                        _bycls = {}
                        for _a in (_ci.get("assets") or []):
                            _uid = _a.get("uid")
                            _cls = _a.get("true_class") or "car"
                            _fp = _a.get("path") or ""
                            if not (_uid and os.path.isfile(_fp)):
                                continue
                            _bycls.setdefault(_cls, []).append(
                                {"uid": _uid, "path": _fp,
                                 "lwh": _a.get("lwh")})
                        for _cls, _its in _bycls.items():
                            items.append({
                                "key": "CANON/" + _cls,
                                "kind": "canon3d",
                                "root": "global",
                                "n": len(_its),
                                "assets": _its,
                                "files": [_x["uid"] + ".glb"
                                          for _x in _its],
                            })
                except Exception:
                    pass
                # 障碍物
                try:
                    _oroot = os.path.join(_groot, "_obs_dec")
                    if os.path.isdir(_oroot):
                        for _oc in sorted(os.listdir(_oroot)):
                            _op = os.path.join(_oroot, _oc)
                            if not os.path.isdir(_op):
                                continue
                            _ol = []
                            for _m in sorted(os.listdir(_op)):
                                _mp = os.path.join(_op, _m)
                                _mf = os.path.join(_mp, "mesh.glb")
                                _tv = os.path.join(_mp, "view_000.png")
                                if os.path.isfile(_mf):
                                    _ol.append(_m)
                            if _ol:
                                items.append({
                                    "key": "OBS/" + _oc,
                                    "kind": "obs3d",
                                    "root": "global",
                                    "n": len(_ol),
                                    "files": _ol,
                                })
                except Exception:
                    pass
                if os.path.isdir(_groot):
                    # SKIP-DUP-DIRS: 这些已由上面的专用分支处理
                    _SKIP = {"SELECTED", "_obs_dec", "_raw_obs", "canonical"}
                    for country in sorted(os.listdir(_groot)):
                        if country in _SKIP:
                            continue
                        cd = os.path.join(_groot, country)
                        if not os.path.isdir(cd):
                            continue
                        for cls in sorted(os.listdir(cd)):
                            d = os.path.join(cd, cls)
                            if not os.path.isdir(d):
                                continue
                            # 每个模型一个子目录，取其 view_000.png 作为缩略图
                            files = []
                            for md in sorted(os.listdir(d)):
                                md_p = os.path.join(d, md)
                                if not os.path.isdir(md_p):
                                    continue
                                for vn in ("view_000.png", "view_001.png"):
                                    if os.path.isfile(os.path.join(md_p, vn)):
                                        files.append(md + "/" + vn)
                                        break
                            if files:
                                items.append({
                                    "key": country + "/" + cls,
                                    "kind": "glb3d",
                                    "root": "global",
                                    "n": len(files),
                                    # NO-FILES-CAP: 全量返回，前端分页
                                    "files": files,
                                })
                for root, tag in (("/data/obj_assets_pc", "pc"),
                                  ("/data/obj_assets_real", "real"),
                                  ("/data/obj_assets", "proc")):
                    if not os.path.isdir(root):
                        continue
                    for cls in sorted(os.listdir(root)):
                        d = os.path.join(root, cls)
                        if not os.path.isdir(d):
                            continue
                        fs = [f for f in sorted(os.listdir(d))
                              if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                        if not fs:
                            continue
                        items.append({
                            "key": cls, "kind": tag,
                            "root": "real" if tag == "real" else "proc",
                            "n": len(fs),
                            "files": fs,      # NO-FILES-CAP
                        })
                return items

            def scan_gnd():
                root = "/data/gnd_assets"
                items = []
                # ASSETS-GLOBAL: 新矢量素材库（按国家分层）
                _gsvg = "/data/assets_global/GND_SVG"
                if os.path.isdir(_gsvg):
                    for country in sorted(os.listdir(_gsvg)):
                        cd = os.path.join(_gsvg, country)
                        pd = os.path.join(cd, "png")
                        if not os.path.isdir(pd):
                            continue
                        fs = [f for f in sorted(os.listdir(pd))
                              if f.lower().endswith(".png")]
                        if fs:
                            items.append({
                                "key": country, "kind": "svg",
                                "root": "global", "n": len(fs),
                                "files": fs,
                            })
                if not os.path.isdir(root):
                    return items
                for key in sorted(os.listdir(root)):
                    d = os.path.join(root, key)
                    if not os.path.isdir(d):
                        continue
                    fs = [f for f in sorted(os.listdir(d))
                          if f.lower().endswith(".png")]
                    if not fs:
                        continue
                    meta = {}
                    mp = os.path.join(d, "meta.json")
                    if os.path.isfile(mp):
                        try:
                            with open(mp, encoding="utf-8") as f:
                                meta = json.load(f)
                        except Exception:
                            pass
                    items.append({"key": key,
                                  "label": meta.get("label", key),
                                  "n": len(fs), "files": fs,
                                  "size": meta.get("size")})
                return items

            if module in ("all", "obj"):
                out["obj"] = scan_obj()
            if module in ("all", "gnd"):
                out["gnd"] = scan_gnd()
            if module == "all":
                tr = "/data/tsr_library_v2"
                if os.path.isdir(tr):
                    n = sum(len([f for f in os.listdir(os.path.join(tr, c))
                                 if f.lower().endswith(".png")])
                            for c in os.listdir(tr)
                            if os.path.isdir(os.path.join(tr, c)))
                    out["tsr"] = [{"key": "tsr_library", "n": n}]
            return self._send(200, {"ok": True, "module": module,
                                    "assets": out})

        if u.path == "/api/assets/image":
            module = (q.get("module") or ["gnd"])[0]
            key = (q.get("key") or [""])[0]
            fn = (q.get("file") or ["tex.png"])[0]
            kind = (q.get("kind") or ["real"])[0]
            # VAL-GLOBAL-KEY: glb3d/svg 允许 key 含一层国家前缀（GLOBAL/car）
            #   fn 也允许 <uid>/view_xxx.png 一层
            #   仍然禁止 ".." 防止目录穿越
            # CANON-KEY-OK: canon3d/obs3d 的 key 也含一层前缀（CANON/car）
            _allow_slash = kind in ("glb3d", "svg", "canon3d", "obs3d")
            if (not key or ".." in key or ".." in fn
                    or (not _allow_slash and "/" in key)
                    or (kind != "glb3d" and "/" in fn)):
                return self._send(400, {"error": "bad param"})
            if module == "gnd":
                # GLOBAL-IMG-ROUTE: kind=svg -> assets_global/GND_SVG/<国家>/png
                if kind == "svg":
                    fp = os.path.join("/data/assets_global/GND_SVG", key,
                                      "png", fn)
                else:
                    fp = os.path.join("/data/gnd_assets", key, fn)
            elif module == "obj":
                # THUMB-FOR-CANON: canonical / obstacle 的缩略图
                if kind == "canon3d":
                    # key="CANON/car", fn="00_car_xxx.glb" -> 找同名 png
                    u = os.path.splitext(fn)[0]
                    fp = os.path.join(
                        "/data/assets_global/OBJ_3D", "canonical",
                        u + ".png")
                    if not os.path.isfile(fp):
                        # 回退到 SELECTED 里的 view_000.png
                        cls = key.split("/")[-1]
                        cand = glob.glob(
                            "/data/assets_global/OBJ_3D/SELECTED/%s/"
                            "%s*/view_000.png" % (cls, u.split("_")[0] + "_"))
                        if cand:
                            fp = cand[0]
                elif kind == "obs3d":
                    cls = key.split("/")[-1]
                    fp = os.path.join(
                        "/data/assets_global/OBJ_3D/_obs_dec", cls,
                        os.path.splitext(fn)[0], "view_000.png")
                elif kind == "glb3d":
                    # key 形如 "GLOBAL/car"，fn 形如 "<uid>/view_000.png"
                    fp = os.path.join("/data/assets_global/OBJ_3D", key, fn)
                elif kind == "pc":
                    root = "/data/obj_assets_pc"
                    fp = os.path.join(root, key, fn)
                elif kind == "real":
                    root = "/data/obj_assets_real"
                    fp = os.path.join(root, key, fn)
                else:
                    root = "/data/obj_assets"
                    fp = os.path.join(root, key, fn)
            else:
                return self._send(400, {"error": "bad module"})
            if not os.path.isfile(fp):
                return self._send(404, {"error": "not found"})
            with open(fp, "rb") as f:
                data = f.read()
            ext = os.path.splitext(fp)[1].lower()
            ctype = ("image/png" if ext == ".png"
                     else "image/jpeg" if ext in (".jpg", ".jpeg")
                     else "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        # RENDER-ASYNC: 轮询重绘任务
        if u.path == "/api/render/job":
            jid = (q.get("id") or [""])[0]
            _r = _render_job_status(jid)
            if _r is None:
                return self._send(404, {"ok": False, "error": "no job"})
            return self._send(200, _r)

        return self._send(404, {"error": "not found"})
    def _delete_canonical(self, key, fn):
        """DEL-CANON-SUPPORT: 删除 canonical/obs3d 素材 + 索引条目。"""
        try:
            import glob as _g2
            uid = os.path.splitext(os.path.basename(fn))[0]
            if ".." in uid or "/" in uid:
                return self._send(400, {"ok": False, "error": "bad uid"})
            deleted = []
            cls = key.split("/")[-1] if "/" in key else key
            if key.startswith("CANON/"):
                base = os.path.join(CANON_DIR, uid)
                idx_fp = OBJ3D_INDEX
            else:
                base = os.path.join(
                    "/data/assets_global/OBJ_3D/_obs_dec", cls, uid,
                    "mesh")
                idx_fp = None
            # 删 glb + png + 同 stem 所有文件
            for p in _g2.glob(base + ".*"):
                try:
                    os.remove(p)
                    deleted.append(p)
                except Exception:
                    pass
            # 删 raw_external 里的原始上传
            raw = os.path.join(RAW_EXTERNAL, uid)
            if os.path.isdir(raw):
                try:
                    import shutil as _sh
                    _sh.rmtree(raw)
                    deleted.append(raw)
                except Exception:
                    pass
            # 从索引移除
            if idx_fp and os.path.isfile(idx_fp):
                try:
                    import json as _j4
                    _d = _j4.load(open(idx_fp, encoding="utf-8"))
                    _before = len(_d.get("assets") or [])
                    _d["assets"] = [a for a in (_d.get("assets") or [])
                                    if a.get("uid") != uid]
                    if len(_d["assets"]) != _before:
                        with open(idx_fp, "w", encoding="utf-8") as _f4:
                            _j4.dump(_d, _f4, indent=1,
                                     ensure_ascii=False)
                        deleted.append("index:%s" % uid)
                except Exception:
                    pass
            if deleted:
                return self._send(200, {"ok": True, "deleted": deleted})
            return self._send(404, {"ok": False, "error": "not found"})
        except Exception as e:
            return self._send(500, {"ok": False, "error": str(e)[:200]})

    def do_POST(self):
        # POST-U-DEF: 解析 URL（do_POST 里必须重新定义）
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/api/light/save":
            try:
                body = self._body()
                assets = (body or {}).get("assets") or {}
                if not isinstance(assets, dict):
                    return self._send(400, {"ok": False, "error": "bad payload"})
                LIGHT_IDX = "/data/assets_global/OBJ_3D/light_index.json"
                # 备份
                if os.path.isfile(LIGHT_IDX):
                    try:
                        import shutil as _sh
                        _sh.copy2(LIGHT_IDX, LIGHT_IDX + ".bak")
                    except Exception:
                        pass
                n = 0
                for uid, a in assets.items():
                    g = a.get("groups") or {}
                    a["n_light_mats"] = sum(len(g.get(k) or []) for k in
                                            ("head", "tail", "turn"))
                    a["supported"] = a["n_light_mats"] >= 2
                    if a["supported"]:
                        n += 1
                payload = {"version": 1, "total": len(assets),
                           "supported": n, "assets": assets}
                tmp = LIGHT_IDX + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=1)
                os.replace(tmp, LIGHT_IDX)
                return self._send(200, {"ok": True, "n": n,
                                        "total": len(assets)})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)[:300]})

        if u.path in ('/api/taskclip/inspect','/api/taskclip/start'):
            try:
                import clip_task_api as _tc
                body=self._body()
                if not isinstance(body,dict): return self._send(400, {'ok':False,'error':'参数无效'})
                result=_tc.inspect(body.get('path'),body.get('module') or 'obj') if u.path.endswith('/inspect') else _tc.start(body)
                return self._send(200, result)
            except (ValueError, OSError, KeyError) as e: return self._send(400, {'ok':False,'error':str(e)[:300]})
            except Exception as e: return self._send(500, {'ok':False,'error':str(e)[:300]})

        if u.path == '/api/studio/v2/image3d-submit':
            try:
                if TOOLS_DIR not in sys.path: sys.path.insert(0, TOOLS_DIR)
                import studio_image3d
                n=int(self.headers.get('Content-Length') or 0)
                if n <= 0 or n > 20*1024*1024: raise ValueError('图片为空或超过 20 MB')
                return self._send(200, studio_image3d.submit(self.rfile.read(n)))
            except Exception as e:
                return self._send(400, {'ok':False, 'error':str(e)[:500]})

        if u.path in ('/api/studio/v2/2d-upload','/api/studio/v2/2d-publish'):
            try:
                import sys as _studio_sys
                if TOOLS_DIR not in _studio_sys.path: _studio_sys.path.insert(0, TOOLS_DIR)
                from studio_2d import upload as _upload2d, publish as _publish2d
                n=int(self.headers.get('Content-Length') or 0)
                if n<=0 or n>30*1024*1024: return self._send(400,{'ok':False,'error':'文件为空或超过 30 MB'})
                raw=self.rfile.read(n)
                if u.path.endswith('/2d-upload'):
                    result=_upload2d(raw,(q.get('module') or [''])[0])
                else:
                    result=_publish2d(json.loads(raw.decode('utf-8')))
                return self._send(200,result)
            except Exception as e:
                return self._send(400,{'ok':False,'error':str(e)[:800]})

        if u.path in ('/api/studio/v2/upload','/api/studio/v2/preview','/api/studio/v2/publish'):
            try:
                import sys as _studio_sys
                if TOOLS_DIR not in _studio_sys.path: _studio_sys.path.insert(0, TOOLS_DIR)
                from studio_safe import upload, preview, publish
                n=int(self.headers.get('Content-Length') or 0)
                if n <= 0 or n > 300*1024*1024:
                    return self._send(400,{'ok':False,'error':'文件为空或超过 300 MB'})
                raw=self.rfile.read(n)
                if u.path.endswith('/upload'):
                    result=upload(raw)
                else:
                    body=json.loads(raw.decode('utf-8'))
                    result=preview(body) if u.path.endswith('/preview') else publish(body)
                return self._send(200,result)
            except Exception as e:
                return self._send(400,{'ok':False,'error':str(e)[:800]})

        # -------- TO-LABEL-EXPORT-API --------
        if u.path == "/api/to_label/export":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8")) \
                    if n else {}
            except Exception as e:
                return self._send(400, {"ok": False,
                                        "error": "bad json: %s" % e})

            mod = str(body.get("module") or "").lower()
            items = body.get("items") or []
            clip = str(body.get("clip") or "").strip()
            ts = str(body.get("timestamp") or "").strip() or "sim"
            extra = body.get("extra") or {}
            root = "/mnt/Data_Platform/zhj_datasim"
            if not os.path.isdir(root):
                return self._send(500, {"ok": False,
                                        "error": "网盘未挂载: %s" % root})

            try:
                sys.path.insert(0, "/opt/ue5_synth/tools")
                if mod == "obj":
                    from export_to_label import (build_xml, write_pretty,
                                                 build_objects_from_items)
                    out = os.path.join(root, "OBJ", clip or "sim_clip")
                    os.makedirs(out, exist_ok=True)
                    objs = build_objects_from_items({"objects": items})
                    ego = extra.get("ego") or {
                        "wx": 0.0, "wy": 0.0, "wz": 0.0, "roll": 0.0,
                        "pitch": 0.0, "yaw": 0.0, "qx": 0.0,
                        "qy": 0.0, "qz": 0.0, "qw": 1.0}
                    K = extra.get("K") or [1319.71, 0.0, 953.25,
                                           0.0, 1319.36, 540.01,
                                           0.0, 0.0, 1.0]
                    el = build_xml(ego, K, [0.0] * 8,
                                   [1.5707963, 0.0, 0.0],
                                   [0.0, 1.5, 1.2], objs)
                    fp = os.path.join(out, "%s.xml" % ts)
                    write_pretty(el, fp)
                    open(os.path.join(out, "%s_ctrl.txt" % ts),
                         "w").close()
                    return self._send(200, {"ok": True, "path": fp,
                                            "n": len(objs),
                                            "module": "obj"})

                if mod == "tsr":
                    from export_tsr_gnd import export_tsr
                    out = os.path.join(root, "TSR", clip or "sim_clip")
                    os.makedirs(out, exist_ok=True)
                    tmp = "/tmp/_tl_tsr_%s.json" % ts
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump({"objects": items}, f,
                                  ensure_ascii=False)
                    fp = export_tsr(tmp, out, ts)
                    return self._send(200, {"ok": True, "path": fp,
                                            "n": len(items),
                                            "module": "tsr"})

                if mod == "gnd":
                    from export_tsr_gnd import (export_gnd,
                                                validate_seg_lane)
                    out = os.path.join(root, "GND")
                    os.makedirs(out, exist_ok=True)
                    rd = export_gnd(extra.get("sim_jpg"),
                                    extra.get("orig_jpg"),
                                    extra.get("seg_png"),
                                    out, ts, extra.get("yaml_src"))
                    pok = None
                    if extra.get("seg_png"):
                        ok, _ = validate_seg_lane(
                            os.path.join(rd, "seg_lane", "%s.png" % ts))
                        pok = ok
                    return self._send(200, {"ok": True, "path": rd,
                                            "palette_ok": pok,
                                            "module": "gnd"})

                return self._send(400, {"ok": False,
                                        "error": "unknown module: %s" % mod})
            except Exception as e:
                import traceback
                return self._send(500, {
                    "ok": False, "error": str(e),
                    "trace": traceback.format_exc()[:800]})

        # -------- TSR-RENDER-API --------
        if u.path == "/api/tsr/render":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8")) \
                    if n else {}
            except Exception as e:
                return self._send(400, {"ok": False,
                                        "error": "bad json: %s" % e})
            batch = str(body.get("batch") or "")
            clip = str(body.get("clip") or "")
            params = body.get("params") or {}
            spec = body.get("spec") or {}
            if not (batch and clip):
                return self._send(400, {"ok": False,
                                        "error": "missing batch/clip"})
            # 把指定牌信息合并进 params
            if spec.get("cls"):
                params["spec_sign"] = spec
                params["mode"] = "spec"
            try:
                jid = _start_render_job("tsr", batch, clip, params)
                return self._send(202, {"ok": True, "job": jid})
            except Exception as e:
                return self._send(500, {"ok": False,
                                        "error": str(e)[:200]})

        # -------- STUDIO-POST-API: 素材工坊 POST 接口 --------
        if u.path == "/api/studio/upload":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                if not raw:
                    return self._send(400, {"ok": False, "error": "empty body"})
                qs = parse_qs(u.query or "")
                uid = (qs.get("uid") or [""])[0].strip()
                if not uid:
                    import time as _t
                    uid = "ext_%d" % int(_t.time())
                safe = "".join(c for c in uid
                               if c.isalnum() or c in "_-")[:64]
                if not safe:
                    return self._send(400, {"ok": False, "error": "bad uid"})
                dst_dir = os.path.join(RAW_EXTERNAL, safe)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, "mesh.glb")
                with open(dst, "wb") as f:
                    f.write(raw)
                return self._send(200, {"ok": True, "uid": safe,
                                        "path": dst, "bytes": len(raw)})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)[:200]})

        if u.path == "/api/studio/canonicalize":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                uid = str(body.get("uid") or "").strip()
                cls = str(body.get("cls") or "car").strip()
                yaw = float(body.get("yaw") or 0.0)
                length = body.get("length")
                write_index = bool(body.get("write_index"))
                if not uid:
                    return self._send(400, {"ok": False, "error": "missing uid"})
                src = os.path.join(RAW_EXTERNAL, uid, "mesh.glb")
                if not os.path.isfile(src):
                    return self._send(404, {"ok": False, "error": "glb not uploaded"})
                out_fp = os.path.join(CANON_DIR, uid + ".glb")
                meta_fp = os.path.join(RAW_EXTERNAL, uid, "meta.json")
                cmd = [BLENDER, "--background", "--python",
                       os.path.join(TOOLS_DIR, "merge_ingest.py"), "--",
                       "--src", src, "--out", out_fp, "--meta", meta_fp,
                       "--uid", uid, "--class", cls, "--yaw", str(yaw)]
                if length is not None:
                    cmd += ["--length", str(float(length))]
                # INDEX-PASS-FIX: 传 --index 才会真写索引
                if not write_index:
                    cmd += ["--dry"]
                else:
                    cmd += ["--index", OBJ3D_INDEX]
                import subprocess as _sp
                r = _sp.run(cmd, capture_output=True, text=True, timeout=600)
                out = (r.stdout or "") + (r.stderr or "")
                lwh, min_z = None, None
                for ln in out.split(chr(10)):
                    if "LWH=" in ln:
                        try:
                            seg = ln.split("LWH=")[1]
                            nums = seg.split(")")[0].strip("(")
                            a, b, c = [float(x) for x in nums.split(",")]
                            lwh = [a, b, c]
                        except Exception:
                            pass
                    if "min_z=" in ln:
                        try:
                            min_z = float(ln.split("min_z=")[1].split()[0])
                        except Exception:
                            pass
                ok = r.returncode == 0 and lwh is not None

                # STUDIO-FULL-PIPE: 入库时渲染缩略图 + 写完整元数据
                thumb = None
                if ok:
                    try:
                        thumb = os.path.join(CANON_DIR, uid + ".png")
                        # THUMB-ALWAYS-RENDER: 每次都重渲染（预览要反映新朝向）
                        if True:
                            import subprocess as _sp2
                            _sp2.run(
                                [BLENDER, "--background", "--python",
                                 os.path.join(TOOLS_DIR, "render_thumb.py"),
                                 "--", out_fp, thumb, "400"],
                                capture_output=True, text=True, timeout=300)
                        if not os.path.isfile(thumb):
                            thumb = None
                    except Exception:
                        thumb = None

                # 写详细元数据（无论 dry 与否都记录）
                meta_full = None
                if ok:
                    try:
                        import json as _j2
                        import time as _t2
                        _mf = {}
                        if os.path.isfile(meta_fp):
                            _mf = _j2.load(open(meta_fp, encoding="utf-8"))
                        _mf.update({
                            "uid": uid,
                            "true_class": cls,
                            "source": str(body.get("source") or ""),
                            "license": str(body.get("license") or ""),
                            "author": str(body.get("author") or ""),
                            "note": str(body.get("note") or ""),
                            "thumb": thumb,
                            "indexed": bool(write_index),
                            "updated_at": _t2.strftime(
                                "%Y-%m-%d %H:%M:%S"),
                        })
                        # 几何精度信息
                        try:
                            # TRIMESH-VENV: 服务器 python3 无 trimesh，用 venv
                            try:
                                import sys as _sy
                                import subprocess as _sp3
                            except Exception:
                                _sp3 = None
                            _code = (
                                "import trimesh,json,sys;"
                                "m=trimesh.load(sys.argv[1]);"
                                "g=getattr(m,'geometry',None) or {};"
                                "nv=sum(len(x.vertices) for x in g.values()) if g else len(m.vertices);"
                                "nf=sum(len(x.faces) for x in g.values()) if g else len(m.faces);"
                                "print(json.dumps({'v':int(nv),'f':int(nf)}))")
                            _r3 = _sp3.run(
                                ["/opt/ad_mining/venv/bin/python", "-c",
                                 _code, out_fp],
                                capture_output=True, text=True, timeout=120)
                            import json as _j3
                            _st = _j3.loads((_r3.stdout or "{}").strip()
                                            .split(chr(10))[-1] or "{}")
                            if _st.get("v"):
                                _mf["n_vertices"] = _st["v"]
                                _mf["n_faces"] = _st["f"]
                        except Exception as _e3:
                            _mf["mesh_stat_err"] = str(_e3)[:80]
                        # TRY-RESTORE: 已用 venv 子进程统计，删除旧块
                        _mf["file_mb"] = round(
                            os.path.getsize(out_fp) / 1048576.0, 2)
                        os.makedirs(os.path.dirname(meta_fp),
                                    exist_ok=True)
                        with open(meta_fp, "w", encoding="utf-8") as _f:
                            _j2.dump(_mf, _f, indent=2, ensure_ascii=False)
                        meta_full = _mf
                    except Exception:
                        meta_full = None

                return self._send(200 if ok else 500, {
                    "ok": ok, "uid": uid, "cls": cls,
                    "lwh": lwh, "min_z": min_z,
                    "out": out_fp if ok else None,
                    "thumb": thumb,
                    "meta": meta_full,
                    "dry": not write_index,
                    "log": out[-800:] if not ok else "",
                })
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)[:300]})

        # -------- ASSET UPLOAD --------
        if u.path in ("/api/assets/upload", "/api/assets/delete"):
            try:
                ctype = self.headers.get("Content-Type") or ""
                if u.path == "/api/assets/delete":
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
                    module = body.get("module") or "gnd"
                    key = body.get("key") or ""
                    fn = body.get("file") or ""
                    kind = body.get("kind") or "real"
                    if not (key and fn):
                        return self._send(400, {"ok": False, "error": "bad"})

                    # DEL-CANON-SUPPORT: canonical / obs3d 走独立分支
                    if kind in ("canon3d", "obs3d") or key.startswith("CANON/") \
                            or key.startswith("OBS/"):
                        return self._delete_canonical(key, fn)

                    if "/" in key or ".." in key or "/" in fn or ".." in fn:
                        return self._send(400, {"ok": False, "error": "bad path"})
                    if module == "gnd":
                        fp = os.path.join("/data/gnd_assets", key, fn)
                    else:
                        # DEL-KIND-FIX: 与 /api/assets/image 的映射保持一致
                        # pc → obj_assets_pc / real → obj_assets_real
                        # proc → obj_assets_v2 / 其他 → obj_assets
                        if kind == "pc":
                            root = "/data/obj_assets_pc"
                        elif kind == "real":
                            root = "/data/obj_assets_real"
                        elif kind == "proc":
                            root = "/data/obj_assets_v2"
                        else:
                            root = "/data/obj_assets"
                        fp = os.path.join(root, key, fn)
                    # DEL-PAIR-FIX: 素材是 <stem>.png + <stem>.json 成对存在
                    # 前端只传一个文件名，需删除同 stem 的所有关联文件
                    import glob as _g
                    _stem, _ext = os.path.splitext(fn)
                    _base = os.path.join(os.path.dirname(fp), _stem)
                    _hits = _g.glob(_base + ".*")
                    _deleted = []
                    for _p2 in _hits:
                        try:
                            os.remove(_p2)
                            _deleted.append(_p2)
                        except Exception:
                            pass
                    if _deleted:
                        return self._send(200, {"ok": True,
                                                "deleted": _deleted})
                    return self._send(404, {"ok": False, "error": "not found"})

                # multipart 上传
                if "multipart/form-data" not in ctype:
                    return self._send(400, {"ok": False, "error": "need multipart"})
                import cgi
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                from email.parser import BytesParser
                from email.policy import default as epolicy
                hdr = ("Content-Type: " + ctype + "\r\n\r\n").encode()
                msg = BytesParser(policy=epolicy).parsebytes(hdr + raw)
                fields = {}
                files = []
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    fn = part.get_filename()
                    data = part.get_payload(decode=True) or b""
                    if fn:
                        files.append((name, fn, data))
                    else:
                        fields[name] = data.decode("utf-8", "replace")

                module = fields.get("module") or "gnd"
                key = fields.get("key") or ""
                kind = fields.get("kind") or "real"
                if not key or "/" in key or ".." in key:
                    return self._send(400, {"ok": False, "error": "bad key"})

                if module == "gnd":
                    d = os.path.join("/data/gnd_assets", key)
                else:
                    if kind == "pc":
                        root = "/data/obj_assets_pc"
                    elif kind == "real":
                        root = "/data/obj_assets_real"
                    else:
                        root = "/data/obj_assets"
                    d = os.path.join(root, key)
                os.makedirs(d, exist_ok=True)

                saved = []
                for name, fn, data in files:
                    if not data:
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
                        continue
                    safe = "upload_" + str(int(time.time() * 1000)) + "_" + \
                        "".join(c for c in os.path.splitext(fn)[0][:20]
                                if c.isalnum() or c in "-_") + ext
                    fp = os.path.join(d, safe)
                    with open(fp, "wb") as f:
                        f.write(data)
                    saved.append({"file": safe, "size": len(data)})
                return self._send(200, {"ok": True, "saved": saved,
                                        "dir": d})
            except Exception as e:
                import traceback
                return self._send(500, {"ok": False, "error": str(e),
                                        "trace": traceback.format_exc()[-600:]})

        u = urlparse(self.path)
        q = parse_qs(u.query)

        # RENDER-ASYNC: GND 参数重绘（异步）
        if u.path == "/api/gnd/render":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                batch = str(body.get("batch") or "")
                clip = str(body.get("clip") or "")
                params = body.get("params") or {}
                if not (batch and clip):
                    return self._send(400, {"ok": False, "error": "missing"})
                source_job=str(body.get('source_job') or '')
                cp = os.path.join("/data/test_data/gnd_train", batch, clip)
                if re.fullmatch(r'gndtask_\d{8}_\d{6}_[0-9a-f]{6}',source_job):
                    mp=os.path.join('/data/render_hybrid',source_job,'task.json')
                    with open(mp,encoding='utf8') as f:meta=json.load(f)
                    with open(os.path.join(os.path.dirname(mp),'config.json'),encoding='utf8') as f:old=json.load(f)
                    if old.get('batch')!=batch or old.get('clip')!=clip:raise ValueError('原任务与当前片段不匹配')
                    cp=meta.get('source') or '';params=dict(params);params['_world_source']=cp
                elif source_job.startswith('manual_gnd_'):
                    previous=_manual_load(source_job[len('manual_gnd_'):])
                    if not previous or previous.get('batch')!=batch or previous.get('clip')!=clip:raise ValueError('原任务与当前片段不匹配')
                    if previous.get('world_source'):
                        cp=previous['world_source'];params=dict(params);params['_world_source']=cp
                if not params.get('_world_source'):
                    return self._send(400, {'ok':False,'error':'请选择支持自车世界轨迹的 GND 片段任务，当前来源不能进行世界轨迹重绘'})
                if not os.path.isdir(cp):
                    return self._send(404, {"ok": False,
                                            "error": "clip not found"})
                jid = _start_render_job("gnd", batch, clip, params)
                return self._send(202, {"ok": True, "job": jid})
            except Exception as e:
                return self._send(500, {"ok": False,
                                        "error": str(e)[:300]})

        u = urlparse(self.path)

        smoke = re.fullmatch(r"/api/(obj|gnd)/run", u.path)
        if smoke:
            payload = self._body()
            if not isinstance(payload, dict):
                return self._send(400, {"error": "小样参数须为 JSON 对象"})
            with SMOKE_LAUNCH_LOCK:
                rec, error = start_job(smoke.group(1) + "_smoke", payload)
            if error:
                return self._send(409, {"error": error})
            return self._send(202, {"id": rec["id"]})

        # [JOB-LIST] 列出所有有产出的任务（工作台下拉用）
        # [DERIVE-SIM] 从检测任务派生仿真任务（只渲染检测到的帧）
        # [JOB-DELETE] 删除任务记录（检测 json 或产出目录）
        # [DISPATCH] OBJ 派发（后台跑 UE，写统一任务注册表）
        if u.path == "/api/obj/dispatch":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                layout = str(body.get("layout") or "cone_line")
                bg = str(body.get("bg") or "cand_03.jpg")
                frames = int(body.get("frames") or 4)
                frames = max(1, min(frames, 200))

                job = "obj_" + time.strftime("%Y%m%d_%H%M%S")
                out_dir = "/data/render_hybrid/" + job
                os.makedirs(out_dir, exist_ok=True)
                os.chmod(out_dir, 0o777)

                # 写任务元信息（供任务记录展示）
                with open(os.path.join(out_dir, "summary.json"), "w",
                          encoding="utf-8") as sf:
                    json.dump({
                        "module": "OBJ",
                        "type": "render",
                        "layout": layout,
                        "bg": bg,
                        "frames": frames,
                        "source_root": "/data/render_hybrid/bgs/" + bg,
                        "created": time.strftime("%Y%m%d_%H%M%S"),
                    }, sf, ensure_ascii=False, indent=1)

                cmd = ["su", "algo", "-c",
                       "cd /data/avsynth/HybridPipeline/scripts && "
                       "DEMO_BG_NAME=%s "
                       "python3 obj_batch_worker.py run "
                       "--frames %d --out %s --chunk %d --seed %d "
                       "--layout %s --bg /data/render_hybrid/bgs/%s"
                       % (bg, frames, out_dir, min(frames, 4),
                          int(time.time()) % 100000, layout, bg)]

                lf = open(os.path.join(out_dir, "run.log"), "w")
                subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)

                return self._send(200, {
                    "ok": True, "job": job, "out": out_dir, "frames": frames,
                })
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)})

        # [PLACE] OBJ 按点选位置合成（像素 → 世界坐标 → 同一 H 投影回像素）
        # 不再走 UE 实时相机：UE 的 fov+pitch 复现不了人工标定 H，会系统性偏移。
        if u.path == "/api/obj/place":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                bg = str(body.get("bg") or "cand_04.jpg")
                picks = body.get("picks") or []
                if not picks:
                    return self._send(400, {"ok": False, "error": "没有点选位置"})

                import sys as _sys
                if TOOLS_DIR not in _sys.path:
                    _sys.path.insert(0, TOOLS_DIR)
                from pixel2world import load_cam, pixel_to_ue_cm

                cam = load_cam(bg)
                if cam is None:
                    return self._send(400, {"ok": False,
                                            "error": "该底图无相机参数，请先做地面标定"})

                placed = []
                for pt in picks[:20]:
                    uu = float(pt.get("u") or 0)
                    vv = float(pt.get("v") or 0)
                    cls = str(pt.get("cls") or "cone")
                    w = pixel_to_ue_cm(cam, uu, vv)
                    if w is None:
                        continue
                    placed.append({"cls": cls,
                                   "Y": round(w["y_cm"] / 100.0, 4),
                                   "Z": round(w["x_cm"] / 100.0, 4)})
                if not placed:
                    return self._send(400, {
                        "ok": False,
                        "error": "所有点都落在地平线附近或太远，请点在画面下方的路面"})

                job = "obj_" + time.strftime("%Y%m%d_%H%M%S")
                out_dir = "/data/render_hybrid/" + job
                os.makedirs(out_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(bg))[0]
                out_img = os.path.join(out_dir, stem + "_obj_composed.jpg")

                cmd = [ALGO_PYTHON,
                       os.path.join(TOOLS_DIR, "obj_sprite.py"),
                       "--bg", bg, "--out", out_img,
                       "--place", json.dumps(placed, ensure_ascii=False)]
                env = dict(os.environ)
                env["PYTHONPATH"] = TOOLS_DIR
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=180, env=env)
                if r.returncode != 0:
                    return self._send(500, {"ok": False,
                                            "error": (r.stderr or r.stdout or "")[-1200:]})

                with open(os.path.join(out_dir, "summary.json"), "w",
                          encoding="utf-8") as sf:
                    json.dump({"module": "OBJ", "type": "place",
                               "bg": bg, "placed": placed,
                               "source_root": "/data/render_hybrid/bgs/" + bg,
                               "created": time.strftime("%Y%m%d_%H%M%S")},
                              sf, ensure_ascii=False, indent=1)

                return self._send(200, {
                    "ok": True, "job": job, "out": out_dir,
                    "image": os.path.basename(out_img),
                    "placed": placed, "count": len(placed),
                })
            except Exception as e:
                import traceback
                return self._send(500, {"ok": False, "error": str(e),
                                        "trace": traceback.format_exc()[-600:]})

        # [REAL-PLACE] 真实数据点选放置（用 XML 真实标定）
        if u.path == "/api/real/place":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                batch = str(body.get("batch") or "")
                clip = str(body.get("clip") or "")
                stem = str(body.get("stem") or "")
                picks = body.get("picks") or []
                if not (batch and clip and stem and picks):
                    return self._send(400, {"ok": False,
                                            "error": "缺少 batch/clip/stem/picks"})

                import sys as _sys
                if TOOLS_DIR not in _sys.path:
                    _sys.path.insert(0, TOOLS_DIR)
                from real_calib import (body_to_pixel_real, load_cam_real,
                                        pixel_to_ue_cm_real)

                cam = load_cam_real(batch, clip, xml_name=stem + ".xml")
                if cam is None:
                    return self._send(400, {"ok": False,
                                            "error": "该帧无标定"})

                placed = []
                rejected = []
                for pt in picks[:20]:
                    uu = float(pt.get("u") or 0)
                    vv = float(pt.get("v") or 0)
                    cls = str(pt.get("cls") or "cone")
                    r = pixel_to_ue_cm_real(cam, uu, vv)
                    if r is None:
                        rejected.append({"u": uu, "v": vv,
                                         "why": "地平线附近或太远"})
                        continue
                    placed.append({"cls": cls,
                                   "x_m": round(r["x_fwd_m"], 3),
                                   "y_m": round(r["y_left_m"], 3),
                                   "px": [uu, vv]})
                if not placed:
                    return self._send(400, {"ok": False,
                                            "error": "所有点都无法反投影",
                                            "rejected": rejected})

                job = "robj_" + time.strftime("%Y%m%d_%H%M%S")
                out_dir = "/data/render/" + job
                os.makedirs(out_dir, exist_ok=True)

                # 渲染：读原图 → 画锥桶 → 存
                import cv2
                import numpy as _np
                REAL_ROOT = "/data/test_data/real_clips"
                img_p = os.path.join(REAL_ROOT, batch, clip,
                                     "cam_st_x2f_front", stem + ".jpg")
                bg = cv2.imread(img_p)
                if bg is None:
                    return self._send(404, {"ok": False,
                                            "error": "原图不存在"})
                h, w = bg.shape[:2]
                out = bg.copy()

                # 前置遮挡掩码（可选）
                try:
                    from ground_masks import (foreground_mask, hood_mask,
                                              road_mask)
                    _road = road_mask(bg)
                    _fg = foreground_mask(bg, _road)
                    _hood = hood_mask(bg, frac=0.18)
                    protect = _np.clip(_np.maximum(_fg, _hood), 0, 1)
                except ImportError:
                    protect = _np.zeros((h, w), _np.float32)

                SIZE = {"cone": (0.28, 0.70), "car": (1.80, 1.50),
                        "suv": (1.90, 1.75), "truck": (2.50, 3.20),
                        "pedestrian": (0.55, 1.70)}

                def _sprite(kind, sh=256):
                    sw = max(int(sh * 0.62), 8)
                    im = _np.zeros((sh, sw, 4), _np.uint8)
                    if kind == "cone":
                        pts = _np.array([[sw // 2, 8], [sw - 6, sh - 6],
                                         [6, sh - 6]], _np.int32)
                        cv2.fillPoly(im, [pts], (30, 130, 240, 255))
                        cv2.rectangle(im, (3, sh - 12), (sw - 3, sh - 3),
                                      (25, 25, 25, 255), -1)
                    else:
                        cv2.rectangle(im, (4, int(sh * 0.32)),
                                      (sw - 4, sh - 6), (150, 150, 150, 255), -1)
                    return im

                objs = []
                for p in sorted(placed, key=lambda q: -q["x_m"]):
                    kind = p["cls"]
                    w_m, h_m = SIZE.get(kind, (0.5, 0.8))
                    # 四角
                    quad = []
                    for yy in (p["y_m"] + w_m / 2, p["y_m"] - w_m / 2):
                        uv, z, ok = body_to_pixel_real(
                            cam, [[p["x_m"], yy, 0.0]])
                        if not ok[0] or not _np.isfinite(uv[0]).all():
                            quad = None
                            break
                        quad.append(uv[0])
                    if not quad:
                        continue
                    uv_l, _, _ = body_to_pixel_real(
                        cam, [[p["x_m"], p["y_m"] + 0.5, 0.0]])
                    uv_r, _, _ = body_to_pixel_real(
                        cam, [[p["x_m"], p["y_m"] - 0.5, 0.0]])
                    px_per_m = abs(uv_l[0][0] - uv_r[0][0])
                    top_dy = h_m * px_per_m

                    left, right = quad[0], quad[1]
                    q4 = _np.array([(left[0], left[1] - top_dy),
                                    (right[0], right[1] - top_dy),
                                    right, left], _np.float32)

                    sp = _sprite(kind)
                    sh, sw = sp.shape[:2]
                    src = _np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1],
                                     [0, sh - 1]], _np.float32)
                    Pm = cv2.getPerspectiveTransform(src, q4)

                    a = sp[..., 3].astype(_np.float32) / 255.0
                    rgb = sp[..., :3].astype(_np.float32)
                    prem = rgb * a[..., None]
                    a_w = cv2.warpPerspective(a, Pm, (w, h),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=0)
                    p_w = cv2.warpPerspective(prem, Pm, (w, h),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0, 0, 0))
                    a_w = _np.clip(a_w, 0, 1)
                    straight = _np.divide(p_w, _np.maximum(a_w[..., None], 1e-6))
                    straight[a_w <= 1e-4] = 0
                    keep = a_w * (1.0 - protect)
                    out = (out.astype(_np.float32) * (1 - keep[..., None])
                           + straight * keep[..., None])
                    out = _np.clip(out, 0, 255).astype(_np.uint8)

                    xs, ys = q4[:, 0], q4[:, 1]
                    objs.append({"class": kind,
                                 "bbox": [float(xs.min()), float(ys.min()),
                                          float(xs.max()), float(ys.max())],
                                 "world": {"x_m": p["x_m"], "y_m": p["y_m"]},
                                 "size_m": [w_m, h_m]})

                out_p = os.path.join(out_dir, stem + "_placed.jpg")
                cv2.imwrite(out_p, out, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 标注
                ann = {
                    "image": os.path.basename(out_p),
                    "batch": batch, "clip": clip, "stem": stem,
                    "width": w, "height": h,
                    "module": "OBJ",
                    "calibration": {"source": "xml",
                                    "fx": cam["fx"], "fy": cam["fy"],
                                    "cx": cam["cx"], "cy": cam["cy"]},
                    "objects": objs,
                    "rejected": rejected,
                    "training_eligible": False,
                    "training_blockers": ["appearance_unverified",
                                          "multi_camera_missing"],
                }
                ad = os.path.join(out_dir, "annotations")
                os.makedirs(ad, exist_ok=True)
                with open(os.path.join(ad, stem + ".json"), "w",
                          encoding="utf-8") as f:
                    json.dump(ann, f, ensure_ascii=False, indent=1)

                return self._send(200, {
                    "ok": True, "job": job, "out": out_dir,
                    "image_url": "/api/real/place_image?job=" + job +
                                 "&name=" + stem + "_placed.jpg",
                    "placed": placed, "objects": objs,
                    "rejected": rejected, "count": len(objs),
                })
            except Exception as e:
                import traceback
                return self._send(500, {"ok": False, "error": str(e),
                                        "trace": traceback.format_exc()[-700:]})

        # [REAL-DISPATCH] 真实数据批量派发
        if u.path == "/api/real/dispatch":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                batch = str(body.get("batch") or "")
                nclip = int(body.get("clips") or 2)
                nfrm = int(body.get("frames") or 8)
                nclip = max(1, min(nclip, 20))
                nfrm = max(1, min(nfrm, 50))
                if not batch or "/" in batch or ".." in batch:
                    return self._send(400, {"ok": False, "error": "bad batch"})

                REAL_ROOT = "/data/test_data/real_clips"
                bp = os.path.join(REAL_ROOT, batch)
                if not os.path.isdir(bp):
                    return self._send(404, {"ok": False,
                                            "error": "数据包不存在"})

                job = "real_" + time.strftime("%Y%m%d_%H%M%S")
                out_dir = "/data/render/" + job
                os.makedirs(out_dir, exist_ok=True)

                cmd = [ALGO_PYTHON,
                       os.path.join(TOOLS_DIR, "verify_batch.py"),
                       "--per-clip", str(nfrm),
                       "--out", out_dir]
                env = dict(os.environ)
                env["PYTHONPATH"] = TOOLS_DIR
                env["REAL_ONLY_BATCH"] = batch
                lf = open(os.path.join(out_dir, "run.log"), "w")
                subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)

                with open(os.path.join(out_dir, "summary_meta.json"), "w",
                          encoding="utf-8") as sf:
                    json.dump({"module": "REAL", "type": "batch",
                               "batch": batch, "clips": nclip,
                               "frames": nfrm,
                               "created": time.strftime("%Y%m%d_%H%M%S")},
                              sf, ensure_ascii=False, indent=1)

                return self._send(200, {"ok": True, "job": job,
                                        "out": out_dir,
                                        "n_clips": nclip,
                                        "n_frames": nfrm})
            except Exception as e:
                import traceback
                return self._send(500, {"ok": False, "error": str(e),
                                        "trace": traceback.format_exc()[-500:]})

        # [PARAMS] 按前端参数实时重绘 OBJ / GND
        # RENDER-ASYNC: OBJ 参数重绘（异步）
        if u.path == "/api/real/render_params":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                batch = str(body.get("batch") or "")
                clip = str(body.get("clip") or "")
                params = body.get("params") or {}
                if not (batch and clip):
                    return self._send(400, {"ok": False, "error": "missing"})
                if params.get('pos_on'):
                    return self._send(400, {"ok":False,"error":"世界轨迹重绘暂不支持指定参考帧位置，请关闭指定位置；不能用固定自车坐标代替"})
                jid = _start_render_job("obj", batch, clip, params)
                return self._send(202, {"ok": True, "job": jid})
            except Exception as e:
                return self._send(500, {"ok": False,
                                        "error": str(e)[:300]})

        if u.path == "/api/gnd/dispatch":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                bg = str(body.get("bg") or "cand_04.jpg")
                lines = str(body.get("lines") or "")
                picks = body.get("picks") or []

                # 点选模式：像素 → 世界坐标 → 生成标线参数
                if picks and not lines:
                    import sys as _sys
                    if TOOLS_DIR not in _sys.path:
                        _sys.path.insert(0, TOOLS_DIR)
                    from pixel2world import load_cam, pixel_to_ue_cm
                    cam = load_cam(bg)
                    if cam is None:
                        return self._send(400, {
                            "ok": False,
                            "error": "该底图无相机参数，请先做地面标定"})
                    parts = []
                    for pt in picks[:10]:
                        u = float(pt.get("u") or 0)
                        v = float(pt.get("v") or 0)
                        cls = str(pt.get("cls") or "lane_solid_white")
                        w = pixel_to_ue_cm(cam, u, v)
                        if w is None:
                            continue
                        x_m = w["x_cm"] / 100.0
                        y_m = w["y_cm"] / 100.0
                        parts.append("%s,%.2f,%.2f,10" % (cls, y_m, x_m))
                    if not parts:
                        return self._send(400, {
                            "ok": False,
                            "error": "所有点都无法反投影（可能点在天空或太近）"})
                    lines = ";".join(parts)

                job = "gnd_" + time.strftime("%Y%m%d_%H%M%S")
                out_dir = "/data/render_hybrid/" + job
                os.makedirs(out_dir, exist_ok=True)

                with open(os.path.join(out_dir, "summary.json"), "w",
                          encoding="utf-8") as sf:
                    json.dump({
                        "module": "GND",
                        "type": "project",
                        "bg": bg,
                        "lines": lines,
                        "source_root": "/data/render_hybrid/bgs/" + bg,
                        "created": time.strftime("%Y%m%d_%H%M%S"),
                    }, sf, ensure_ascii=False, indent=1)

                cmd = [ALGO_PYTHON,
                       os.path.join(TOOLS_DIR, "gnd_project.py"),
                       "--bg", bg, "--out", out_dir]
                if lines:
                    cmd += ["--lines", lines]

                env = dict(os.environ)
                env["PYTHONPATH"] = TOOLS_DIR
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=180, env=env)
                if r.returncode != 0:
                    return self._send(500, {
                        "ok": False,
                        "error": (r.stderr or r.stdout or "")[-1200:],
                    })

                stem = os.path.splitext(os.path.basename(bg))[0]
                img = os.path.join(out_dir, stem + "_gnd_composed.jpg")
                return self._send(200, {
                    "ok": True,
                    "job": job,
                    "out": out_dir,
                    "image": os.path.basename(img) if os.path.isfile(img) else "",
                })
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)})

        # [GND-PROJECT] 真实底图 + 磨损纹理 + 人工标定的 2.5D 标线投影
        if u.path == "/api/gnd/project":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                bg = str(body.get("bg") or "cand_04.jpg")
                lines = str(body.get("lines") or "")
                tag = time.strftime("%Y%m%d_%H%M%S")
                out_dir = "/data/render_hybrid/gnd_out_" + tag
                os.makedirs(out_dir, exist_ok=True)

                cmd = [ALGO_PYTHON,
                       os.path.join(TOOLS_DIR, "gnd_project.py"),
                       "--bg", bg,
                       "--out", out_dir]
                if lines:
                    cmd += ["--lines", lines]

                env = dict(os.environ)
                env["PYTHONPATH"] = TOOLS_DIR
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=180, env=env)
                if r.returncode != 0:
                    return self._send(500, {
                        "ok": False,
                        "error": (r.stderr or r.stdout or "")[-1500:],
                    })

                # 找产出
                ann_dir = os.path.join(out_dir, "annotations")
                stem = os.path.splitext(os.path.basename(bg))[0]
                img_p = os.path.join(out_dir, stem + "_gnd_composed.jpg")
                ann_p = os.path.join(ann_dir, stem + ".json")
                ann = {}
                if os.path.isfile(ann_p):
                    with open(ann_p, encoding="utf-8") as f:
                        ann = json.load(f)

                return self._send(200, {
                    "ok": True,
                    "out_dir": out_dir,
                    "image": os.path.basename(img_p) if os.path.isfile(img_p) else "",
                    "image_url": "/api/gnd/result?dir=" + os.path.basename(out_dir) +
                                 "&name=" + os.path.basename(img_p),
                    "lines": ann.get("lines", []),
                    "quality": ann.get("quality", {}),
                    "training_eligible": ann.get("training_eligible", False),
                    "log": r.stdout[-2000:],
                })
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)})


        # [CALIB] 地面四点点选标定
        if u.path == "/api/tsr/calib_save":
            body = self._body()
            if not isinstance(body, dict):
                body = {}
            try:
                name = str(body.get("image") or "custom").replace("/", "_")
                calib_dir = "/data/render_hybrid/bgs"
                os.makedirs(calib_dir, exist_ok=True)
                out = os.path.join(calib_dir, "_calib_" + os.path.splitext(name)[0] + ".json")
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(body, f, ensure_ascii=False, indent=1)
                return self._send(200, {"ok": True, "path": out})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)})


        if u.path == "/api/tsr/job_delete":
            try:
                body = self._body()
                if not isinstance(body, dict):
                    body = {}
                job = str(body.get("job") or "").strip()
                if not job:
                    return self._send(400, {"error": "缺少 job"})

                deleted = []
                # 1) 检测结果
                pj = "/tmp/prescreen_%s.json" % job
                if os.path.isfile(pj):
                    os.remove(pj)
                    deleted.append("detect")
                pj2 = "/tmp/prescreen_prog_%s.json" % job
                if os.path.isfile(pj2):
                    os.remove(pj2)
                    deleted.append("progress")
                td = "/tmp/prescreen_thumbs_%s" % job
                if os.path.isdir(td):
                    import shutil as _sh
                    _sh.rmtree(td, ignore_errors=True)
                    deleted.append("thumbs")

                # 2) 产出目录
                if body.get("delete_files"):
                    for cand in (os.path.join(TSR_OUT_DIR, "walk_" + job),
                                 os.path.join("/mnt/Data_Platform/zhj_datasim", job)):
                        if os.path.isdir(cand):
                            import shutil as _sh
                            _sh.rmtree(cand, ignore_errors=True)
                            deleted.append("dir:" + cand)

                return self._send(200, {"ok": True, "deleted": deleted})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        if u.path == "/api/tsr/derive_sim":
          try:
            body = self._body()
            if not isinstance(body, dict):
                body = {}
            det_job = str(body.get("detect_job") or "").strip()
            country = str(body.get("country") or "BR")
            pin_class = str(body.get("pin_class") or "")
            module = str(body.get("module") or "TSR").upper()

            if not det_job:
                return self._send(400, {"error": "缺少 detect_job"})

            # 读检测结果
            pj = "/tmp/prescreen_%s.json" % det_job
            if not os.path.isfile(pj):
                return self._send(404, {"error": "检测结果不存在: " + det_job})
            with open(pj, "r", encoding="utf-8") as f:
                pd = json.load(f)

            usable = [r for r in pd.get("results", []) if r.get("usable", 0) > 0]
            if not usable:
                return self._send(400, {"error": "该检测任务没有可用帧"})

            # 构造渲染清单
            picked = []
            for r in usable:
                rp = r.get("path") or ""
                if rp and os.path.isfile(rp):
                    picked.append({
                        "stem": r.get("stem") or os.path.basename(rp),
                        "path": rp,
                        "seq": r.get("seq") or ""
                    })
            if not picked:
                return self._send(400, {"error": "检测结果里的路径都不可用"})

            # 任务 ID：检测ID + _sim
            tag = det_job + "_sim"
            out_dir = os.path.join(TSR_OUT_DIR, "walk_" + tag)
            os.makedirs(out_dir, exist_ok=True)
            lst = "/tmp/render_sel_%s.json" % tag
            with open(lst, "w", encoding="utf-8") as f:
                json.dump(picked, f, ensure_ascii=False)
            prog = os.path.join(out_dir, "walk_%s_progress.json" % tag)

            # 写元信息
            try:
                with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as sf:
                    json.dump({
                        "module": module,
                        "country": country,
                        "pin_class": pin_class,
                        "type": "render",
                        "from_detect": det_job,
                        "source_root": pd.get("dir", ""),
                        "count": len(picked),
                        "created": time.strftime("%Y%m%d_%H%M%S"),
                    }, sf, ensure_ascii=False, indent=1)
            except Exception:
                pass

            cmd = [ALGO_PYTHON, os.path.join(TOOLS_DIR, "render_selected.py"),
                   "--list", lst, "--out", out_dir,
                   "--country", country,
                   "--signs-dir", load_config().get("tsr_library") or DEFAULT_TSR_LIBRARY,
                   "--jobs-progress", prog]
            if pin_class:
                cmd += ["--pin-class", pin_class]

            try:
                lf = open(os.path.join(out_dir, "render.log"), "w")
                subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            except Exception as e:
                return self._send(500, {"error": str(e)})

            return self._send(200, {"ok": True, "job": tag,
                                    "from": det_job, "count": len(picked),
                                    "out": out_dir})
          except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            try:
                with open("/tmp/derive_err.log", "a") as _lf:
                    _lf.write(_tb.format_exc() + "\n")
            except Exception:
                pass
            return self._send(500, {"error": str(_e)})


        if u.path == "/api/hybrid/submit":
            payload = self._body()
            if not isinstance(payload, dict):
                return self._send(400, {"error": "参数须为 JSON 对象"})
            try:
                n = int(payload.get("frames") or 0)
            except (TypeError, ValueError):
                return self._send(400, {"error": "frames 需为整数"})
            bg = payload.get("bg") or "/data/render_hybrid/bgs/cand_03.jpg"
            name = payload.get("name") or ""
            lay = payload.get("layout") or ""
            return self._send(200, hybrid_submit(n, bg, name, layout=lay))
        if u.path == "/api/preview":
            rec, err = start_job("preview", {})
            return self._send(400, {"error": err}) if err else self._send(200, {"id": rec["id"]})
        if u.path == "/api/assets/upload":
            name = self.headers.get("X-Filename", "")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:png|jpg|jpeg)", name, re.I):
                return self._send(400, {"error": "文件名仅允许英数字、点、下划线、连字符及 PNG/JPEG 扩展名"})
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= 20 * 1024 * 1024:
                return self._send(400, {"error": "图片须小于 20 MB"})
            blob = self.rfile.read(size)
            if not (blob.startswith(b"\x89PNG\r\n\x1a\n") if name.lower().endswith(".png")
                    else blob.startswith(b"\xff\xd8\xff")):
                return self._send(400, {"error": "图片内容与扩展名不符"})
            folder = image_inbox()
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, name), "wb") as f:
                f.write(blob)
            return self._send(200, {"name": name, "bytes": len(blob)})
        p = self._body()

        # [PRESCREEN] YOLO 预筛（异步，带进度）
        if u.path == "/api/tsr/prescreen":
            body = p
            # UNC / Windows 路径统一转成 Linux 挂载路径
            d = normalize_net_path(str(body.get("dir") or "").strip())
            conf = float(body.get("conf") or 0.30)
            min_px = int(body.get("min_px") or 24)
            limit = int(body.get("limit") or 300)
            if not d or not os.path.isdir(d):
                return self._send(400, {"error": "目录不存在: " + d, "dir": d})
            import subprocess as _sp
            tid = uuid.uuid4().hex[:10]
            out_json = "/tmp/prescreen_%s.json" % tid
            prog_json = "/tmp/prescreen_prog_%s.json" % tid
            try:
                with open(prog_json, "w", encoding="utf-8") as fh:
                    json.dump({"done": 0, "total": 0, "usable": 0, "hits": 0,
                               "finished": False}, fh)
            except Exception:
                pass
            exe = ALGO_PYTHON if os.path.isfile(ALGO_PYTHON) else "python3"
            script_p = os.path.join(TOOLS_DIR, "prescreen_yolo.py")
            pin_class = str(body.get("pin_class") or "")
            thumb_dir = "/tmp/prescreen_thumbs_%s" % tid
            cmd = [exe, script_p, d, "--out", out_json, "--conf", str(conf),
                   "--min-px", str(min_px), "--limit", str(limit),
                   "--progress", prog_json,
                   "--recursive", "--thumb-dir", thumb_dir]
            if pin_class:
                cmd += ["--pin-class", pin_class]
            _sp.Popen(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                      start_new_session=True)
            self._send(200, {"ok": True, "task": tid, "dir": d})
            return

        # [RENDER-SELECTED] 只渲染勾选的帧
        if u.path == "/api/tsr/render_selected":
            body = p
            tid = str(body.get("task") or "")
            stems = body.get("stems") or []
            country = str(body.get("country") or "BR")
            pin_class = str(body.get("pin_class") or "")
            if not tid or not stems:
                return self._send(400, {"error": "缺少 task/stems"})

            jf = "/tmp/prescreen_%s.json" % tid
            if not os.path.isfile(jf):
                return self._send(404, {"error": "检测结果不存在"})
            with open(jf, "r", encoding="utf-8") as f:
                res = json.load(f)

            want = set(stems)
            picked = []
            for r in res.get("results", []):
                key = r.get("stem") or ""
                if key in want and r.get("path"):
                    picked.append({"stem": key, "path": r["path"],
                                   "seq": r.get("seq") or ""})
            if not picked:
                return self._send(400, {"error": "没有匹配的帧"})

            tag = time.strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(TSR_OUT_DIR, "walk_" + tag)
            os.makedirs(out_dir, exist_ok=True)
            lst = "/tmp/render_sel_%s.json" % tag
            with open(lst, "w", encoding="utf-8") as f:
                json.dump(picked, f, ensure_ascii=False)
            prog = os.path.join(out_dir, "walk_%s_progress.json" % tag)

            cmd = [ALGO_PYTHON, os.path.join(TOOLS_DIR, "render_selected.py"),
                   "--list", lst, "--out", out_dir,
                   "--country", country,
                   "--signs-dir", load_config().get("tsr_library") or DEFAULT_TSR_LIBRARY,
                   "--jobs-progress", prog]
            if pin_class:
                cmd += ["--pin-class", pin_class]
            try:
                lf = open(os.path.join(out_dir, "render.log"), "w")
                subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            except Exception as e:
                return self._send(500, {"error": str(e)})
            return self._send(200, {"ok": True, "job": tag, "out": out_dir,
                                    "count": len(picked)})


        # [PRESCREEN] 读取检测结果

        # [PRESCREEN] 进度
        if u.path == "/api/tsr/prescreen_progress":
            tid = (q.get("task") or [""])[0]
            pf = "/tmp/prescreen_prog_%s.json" % tid
            if not os.path.isfile(pf):
                return self._send(200, {"done": 0, "total": 0, "usable": 0,
                                        "hits": 0, "finished": False})
            with open(pf, "r", encoding="utf-8") as f:
                return self._send(200, json.load(f))

        # [PRESCREEN] 缩略图

        # [TSR-API] 单图检测
        if u.path == "/api/tsr/detect":
            b = p
            d = str(b.get("dir") or "")
            f = str(b.get("file") or "")
            cf = float(b.get("conf") or 0.30)
            if not d or not f:
                return self._send(400, {"error": "缺少 dir/file"})
            fp = os.path.join(d, f)
            if not os.path.isfile(fp):
                return self._send(400, {"error": "图片不存在"})
            return self._send(200, _run_algo(
                [os.path.join(TOOLS_DIR, "tsr_detect_api.py"), fp, str(cf)], timeout=180))

# [PARAM-PREVIEW] 按前端参数重渲染单帧
        if u.path == "/api/tsr/param_preview":
            body = p
            stem = str(body.get("stem") or "").strip()
            d = normalize_net_path(str(body.get("dir") or "").strip())
            country = str(body.get("country") or "BR")
            cls = str(body.get("target_class") or "")
            bbox = body.get("bbox") or []
            params = body.get("params") or {}

            if not stem or not d:
                return self._send(400, {"error": "缺少 stem/dir"})

            # 从标注里拿原图路径
            ap = os.path.join(d, "annotations", stem + ".json")
            if not os.path.isfile(ap):
                return self._send(404, {"error": "标注不存在: " + ap})
            with open(ap, "r", encoding="utf-8") as f:
                ann = json.load(f)
            src = ann.get("source_path", "")
            if not src or not os.path.isfile(src):
                return self._send(404, {"error": "原图不存在: " + src})

            os.makedirs(TSR_OUT_DIR, exist_ok=True)
            # 文件名带参数哈希：参数一变就是新文件，浏览器不会命中旧缓存
            import hashlib as _hl
            _ph = _hl.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
            out = os.path.join(TSR_OUT_DIR,
                               "param_%s_%s.jpg" % (_ph, stem[-24:]))

            # Reuse the selected task's country and curated library; a material
            # filename can refer to a different sign in the default library.
            library=load_config().get('tsr_library') or DEFAULT_TSR_LIBRARY
            try:
                with open(os.path.join(d,'config.json'),encoding='utf8') as f:taskcfg=json.load(f)
                taskcountry=str(taskcfg.get('country') or '').upper()
                if re.fullmatch(r'[A-Z]{2}',taskcountry):country=taskcountry
                candidate=os.path.realpath(str(taskcfg.get('signs_dir') or ''))
                allowed={'/data/tsr_library_v2','/data/tsr_curated_20261125_v1'}
                if candidate in allowed and os.path.isdir(candidate):library=candidate
            except (OSError,ValueError,TypeError):pass
            args = [ALGO_PYTHON, os.path.join(TOOLS_DIR, "tsr_param_preview.py"),
                    src, out,
                    "--library", library,
                    "--country", country]
            if cls:
                args += ["--target-class", cls]
            # 标注里记了素材文件名，直接传，避免猜错
            mat0 = str(body.get("material") or "")
            if not mat0:
                objs = ann.get("objects") or []
                if objs:
                    mat0 = str(objs[0].get("material") or "")
                    if not cls:
                        cls = str(objs[0].get("class") or "")
                        args += ["--target-class", cls]
            if mat0:
                args += ["--material", mat0]
            if bbox and len(bbox) == 4:
                args += ["--bbox", ",".join(str(round(float(v), 1)) for v in bbox)]
            args += ["--params", json.dumps(params, ensure_ascii=False)]

            jid='tsr_%d_%s'%(int(time.time()),_uuid.uuid4().hex[:6])
            manual={'id':jid,'kind':'tsr','state':'running','manual':True,'source':src,'country':country,'t0':time.time(),'n':0,'parent_dir':d,'stem':stem,'params':params,'material':mat0,'library':library,'versions':_manual_versions()}
            _manual_save(jid,manual)
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=120)
                last = (r.stdout or "").strip().split("\n")[-1]
                info = json.loads(last) if last.startswith("{") else {"ok": False, "error": last}
                if info.get("ok"):
                    info["url"] = ("/api/tsr/result?dir=" + quote(TSR_OUT_DIR) +
                                   "&file=" + quote(os.path.basename(out)))
                manual.update(state='done' if info.get('ok') else 'failed',n=1 if info.get('ok') else 0,err='' if info.get('ok') else str(info.get('error') or '单帧预览失败'),t1=time.time(),url=info.get('url',''))
                _manual_save(jid,manual);info['job']=jid
                return self._send(200 if info.get("ok") else 500, info)
            except Exception as e:
                manual.update(state='failed',err='渲染异常: '+str(e),t1=time.time());_manual_save(jid,manual)
                return self._send(500, {"error": "渲染异常: " + str(e),"job":jid})

        # [TSR-API] 单图替换预览
        if u.path == "/api/tsr/replace":
            body = p
            d = str(body.get("dir") or "")
            f = str(body.get("file") or "")
            country = str(body.get("country") or "")
            cls = str(body.get("target_class") or "")
            bbox = body.get("bbox") or []
            if not d or not f:
                return self._send(400, {"error": "缺少 dir/file"})
            os.makedirs(TSR_OUT_DIR, exist_ok=True)
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                return self._send(400, {"error": "图片不存在"})
            out = os.path.join(TSR_OUT_DIR, _tsr_slug(os.path.splitext(f)[0]) + "_preview.jpg")
            args = [os.path.join(TOOLS_DIR, "tsr_preview_api.py"), p, out,
                    "--library", load_config().get("tsr_library") or DEFAULT_TSR_LIBRARY,
                    "--country", country]
            if cls:
                args += ["--target-class", cls]
            shp = str(body.get("shape") or "")
            if shp in ("circle", "triangle", "rect"):
                args += ["--shape", shp]
            # 仿真参数（面板滑块）
            for k in ("bright_gain", "haze_gain", "contrast_gain",
                      "satur_keep", "feather_ratio"):
                v = body.get(k)
                if v is not None:
                    try:
                        args += ["--" + k.replace("_", "-"), str(float(v))]
                    except (TypeError, ValueError):
                        pass
            if bbox and len(bbox) == 4:
                args += ["--bbox", ",".join(str(float(v)) for v in bbox)]
            res = _run_algo(args, timeout=300)
            if res.get("ok"):
                res["url"] = "/api/tsr/result?file=" + os.path.basename(out)
            return self._send(200, res)

        # [TSR-API] 批量替换
        if u.path == "/api/tsr/walk":
            try:
                import clip_task_api as _tc
                if not isinstance(p,dict):return self._send(400,{'ok':False,'error':'参数无效'})
                root=normalize_net_path(str(p.get('root') or '').strip())
                return self._send(200,_tc.start_tsr(p,root))
            except (ValueError,OSError,KeyError) as e:return self._send(400,{'ok':False,'error':str(e)[:300]})
            except Exception as e:return self._send(500,{'ok':False,'error':str(e)[:300]})

        if u.path == "/api/tsr/batch":
            body = p
            d = str(body.get("dir") or "")
            country = str(body.get("country") or "")
            cls = str(body.get("target_class") or "")
            limit = int(body.get("limit") or 20)
            if not d or not os.path.isdir(d):
                return self._send(400, {"error": "实拍目录不存在"})
            os.makedirs(TSR_OUT_DIR, exist_ok=True)
            cmd = [os.path.join(TOOLS_DIR, "batch_replace.py"),
                   "--images", d, "--signs-dir", cfg.get("tsr_library") or DEFAULT_TSR_LIBRARY,
                   "--out", TSR_OUT_DIR, "--country", country, "--limit", str(limit)]
            if cls:
                cmd += ["--target-class", cls]
            shp2 = str(body.get("shape") or "")
            if shp2 in ("circle", "triangle", "rect"):
                cmd += ["--shape", shp2]
            for k in ("bright_gain", "haze_gain", "contrast_gain",
                      "satur_keep", "feather_ratio"):
                v = body.get(k)
                if v is not None:
                    try:
                        cmd += ["--" + k.replace("_", "-"), str(float(v))]
                    except (TypeError, ValueError):
                        pass
            if body.get("vary_class"):
                cmd += ["--vary-class"]
            rec, err = start_job("tsr_batch", {"cmd": cmd, "out": TSR_OUT_DIR})
            if err:
                return self._send(400, {"error": err})
            return self._send(self._json_code if hasattr(self, "_json_code") else 200,
                              {"id": rec["id"], "out": TSR_OUT_DIR})
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
            body = p or {}
            # 按进程名杀（前端「停止遍历」用）
            pat = str(body.get("pattern") or "").strip()
            if pat:
                try:
                    r = subprocess.run(["pkill", "-f", pat],
                                       capture_output=True, text=True, timeout=5)
                    return self._send(200, {"killed_by_pattern": pat,
                                            "rc": r.returncode})
                except Exception as e:
                    return self._send(500, {"error": str(e)})
            return self._send(200, {"killed": kill_job(body.get("id", ""))})
        if u.path == "/api/config":
            return self._send(200, save_config(p))
        return self._send(404, {"error": "not found"})

def acquire_single_instance_lock():
    """只允许一个面板进程：否则后台自动化（显卡记录 + 自动处理链）会被起两份。
    返回锁文件句柄；Windows（无 fcntl）或拿不到锁返回 None，由调用方决定是否退出。"""
    if fcntl is None:
        return None
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        f = open(os.path.join(LOG_DIR, ".panel.lock"), "w")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except Exception:
        return None

def main():
    port = int(os.environ.get("UE5_PANEL_PORT") or (sys_argv_port() or 8010))
    os.makedirs(JOBS_DIR, exist_ok=True)
    lock = acquire_single_instance_lock()
    if fcntl is not None and lock is None:
        log_auto("已有面板进程在跑（锁被占用），本进程退出")
        sys.exit(3)
    if AUTO_POST or AUTO_GPU_WATCH:
        threading.Thread(target=auto_loop, daemon=True).start()
        log_auto("后台自动化已启用：显卡记录=%s，渲染结束自动处理=%s（轮询 %ds）"
                 % (AUTO_GPU_WATCH, AUTO_POST, int(AUTO_POLL_SEC)))
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print("[panel] http://0.0.0.0:%d  (Ctrl-C 停)" % port)
    srv.serve_forever()

def sys_argv_port():
    return sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else None

if __name__ == "__main__":
    main()
