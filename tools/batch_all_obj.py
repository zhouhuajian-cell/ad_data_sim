#!/usr/bin/env python3
"""批量跑所有 clip 的 OBJ 仿真，产出统一的 final_obj3。"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "/opt/ue5_synth/tools")

OUT = "/data/render/final_obj3"
PY = "/opt/ad_mining/venv/bin/python"
TOOLS = "/opt/ue5_synth/tools"

# 每类取几帧
FRAMES = 6
CONFIG = '{"count":3,"cls":"car","dist_min":15,"dist_max":32,"lat":3.0,"scale":1.0,"shadow":0.35,"occlusion":true}'


def list_clips():
    roots = ["/data/test_data/real_clips",
             "/data/test_data/real_clips_extra"]
    out = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for b in sorted(glob.glob(os.path.join(r, "*"))):
            if not os.path.isdir(b):
                continue
            for c in sorted(glob.glob(os.path.join(b, "*"))):
                if os.path.isdir(os.path.join(c, "cam_st_x2f_front")):
                    out.append((os.path.basename(b), os.path.basename(c), c))
    return out


def merge_dir(src, dst):
    """把 src 的 frames/annotations 合并到 dst。"""
    for sub in ("frames", "annotations"):
        s = os.path.join(src, sub)
        d = os.path.join(dst, sub)
        if not os.path.isdir(s):
            continue
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(s):
            sp = os.path.join(s, f)
            dp = os.path.join(d, f)
            # 加批次前缀避免重名
            if os.path.isfile(sp):
                shutil.copy2(sp, dp)


def main():
    clips = list_clips()
    print("待处理 clip: %d" % len(clips))

    os.makedirs(OUT, exist_ok=True)
    tmp = "/tmp/obj_batch_out"

    ok = 0
    fail = 0
    t0 = time.time()

    for i, (batch, clip, cdir) in enumerate(clips):
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        cmd = [PY, os.path.join(TOOLS, "pipeline_sim.py"),
               "--batch", batch, "--clip", clip,
               "--module", "obj", "--frames", str(FRAMES),
               "--out", tmp, "--config", CONFIG]
        env = dict(os.environ)
        env["PYTHONPATH"] = TOOLS
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=600, env=env)
            n_sim = len(glob.glob(os.path.join(tmp, "frames", "*_sim.jpg")))
            if r.returncode == 0 and n_sim > 0:
                merge_dir(tmp, OUT)
                ok += 1
                print("  [%d/%d] %-30s %-24s %d 帧"
                      % (i + 1, len(clips), batch[:30], clip[:24], n_sim))
            else:
                fail += 1
                err = (r.stderr or r.stdout or "")[-100:]
                print("  [%d/%d] %-30s %-24s 失败 %s"
                      % (i + 1, len(clips), batch[:30], clip[:24],
                         err.replace("\n", " ")[:60]))
        except subprocess.TimeoutExpired:
            fail += 1
            print("  [%d/%d] %s 超时" % (i + 1, len(clips), clip[:24]))
        except Exception as e:
            fail += 1
            print("  [%d/%d] %s 异常 %s"
                  % (i + 1, len(clips), clip[:24], str(e)[:50]))

    dt = time.time() - t0
    n_f = len(glob.glob(os.path.join(OUT, "frames", "*_sim.jpg")))
    n_a = len(glob.glob(os.path.join(OUT, "annotations", "*.json")))
    print("\n=== 完成 %.1f 分钟 ===" % (dt / 60))
    print("成功 %d / 失败 %d" % (ok, fail))
    print("产出: %d 帧 / %d 标注" % (n_f, n_a))

    if os.path.isdir(tmp):
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
