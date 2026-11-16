# -*- coding: utf-8 -*-
# server.py — UE5 合成流水线可视化操作台（零依赖单文件，服务器系统 python3 直接跑）
# 干什么：网页上做 环境核查/生成任务/启动停止渲染/进度/后处理/日志查看，只暴露白名单动作。
# 怎么跑：python3 server.py [端口]  → 浏览器开 http://<服务器IP>:8010
# 成功判据：/api/status 出 JSON；点"环境核查"出 PASS/FAIL 列表；渲染按钮对应 tmux 会话起停。
# ⚠️ 只调白名单脚本、参数走 argv 不过 shell；不碰 ad_mining 与其 venv。
# ⚠️ 后台自动化：显卡记录随面板常驻；渲染结束（或停止）后自动处理+打包+抽检，
#    失败重试与跳过已完成都在这层做，界面上不再提供对应按钮（见 AUTO_* 常量）。
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
