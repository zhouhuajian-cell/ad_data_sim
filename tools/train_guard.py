#!/usr/bin/env python3
"""训练守护进程：崩溃自动重启续训，绝不放弃。

用法：
  python train_guard.py --config guard_config.json

行为：
  - 启动训练
  - 进程退出（崩溃/OOM/被杀）→ 等待后从 last.pt 续训
  - 最多重试 N 次
  - 记录所有崩溃到 crashes.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import time

PY = "/opt/ad_mining/venv/bin/python"
TOOLS = "/opt/ue5_synth/tools"


def load_cfg(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_resume(project, name):
    """找可续训的 last.pt。"""
    for f in ("last.pt", "best.pt"):
        p = os.path.join(project, name, "weights", f)
        if os.path.isfile(p) and os.path.getsize(p) > 1e6:
            return p
    return None


def run_once(cfg, resume=None):
    """跑一次训练。"""
    script = os.path.join(TOOLS, cfg["script"])
    cmd = [PY, script] + cfg["args"]
    if resume:
        cmd += ["--model", resume]

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["OMP_NUM_THREADS"] = "4"

    log = cfg["log"]
    print("[guard] 启动: %s" % " ".join(cmd[-6:]))
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n===== [guard] start %s resume=%s =====\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"), resume))
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           env=env)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    project = cfg.get("project", "/data/train/runs")
    name = cfg.get("name", "obj_v2_pr")
    max_retry = cfg.get("max_retry", 10)
    wait = cfg.get("retry_wait", 60)
    crash_log = cfg.get("crash_log", "/data/train/crashes.jsonl")

    os.makedirs(os.path.dirname(crash_log), exist_ok=True)

    for attempt in range(1, max_retry + 1):
        resume = find_resume(project, name) if attempt > 1 else None
        if attempt > 1 and resume is None:
            print("[guard] 无可续训权重，从头开始")

        t0 = time.time()
        code = run_once(cfg, resume)
        dt = time.time() - t0

        rec = {"attempt": attempt, "returncode": code,
               "elapsed_s": round(dt, 1),
               "resume": resume,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(crash_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if code == 0:
            print("[guard] ✅ 训练正常完成，用时 %.1f 分钟" % (dt / 60))
            return 0

        print("[guard] ⚠️ 第 %d 次异常退出 code=%s，%.0f 分钟后续训"
              % (attempt, code, wait))
        # 清理显存
        subprocess.run(["bash", "-c",
                        "pkill -f 'train_pr|train_yolo|train_gnd' || true"],
                       capture_output=True)
        time.sleep(wait)

    print("[guard] ❌ 重试 %d 次仍失败" % max_retry)
    return 1


if __name__ == "__main__":
    sys.exit(main())
