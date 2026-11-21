#!/usr/bin/env python3
"""批量跑所有 GND clip 的仿真。"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

OUT = "/data/render/gnd_model"
PY = "/opt/ad_mining/venv/bin/python"
TOOLS = "/opt/ue5_synth/tools"

FRAMES = 6
CONFIG = ('{"lanes":3,"dash":true,"edge":0.20,"width":0.15,'
          '"wear":0.25,"opacity":1.0,"lane_type":"lane_dashed_white",'
          '"reflectivity":1.95,"occlusion":true}')


def list_gnd_clips():
    root = "/data/test_data/gnd_train"
    out = []
    if not os.path.isdir(root):
        return out
    for vd in sorted(os.listdir(root)):
        vp = os.path.join(root, vd)
        if not os.path.isdir(vp):
            continue
        for cl in sorted(os.listdir(vp)):
            cp = os.path.join(vp, cl)
            if os.path.isdir(os.path.join(cp, "img_ori")):
                out.append((vd, cl, cp))
    return out


def merge_dir(src, dst):
    for sub in ("frames", "annotations", ("frames", "seg")):
        pass
    for sub in ("frames", "annotations"):
        s = os.path.join(src, sub)
        d = os.path.join(dst, sub)
        if not os.path.isdir(s):
            continue
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(s):
            sp = os.path.join(s, f)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(d, f))


def main():
    clips = list_gnd_clips()
    print("GND clip: %d" % len(clips))
    if not clips:
        raise SystemExit("无数据")

    os.makedirs(OUT, exist_ok=True)
    tmp = "/tmp/gnd_batch_out"

    ok = 0
    fail = 0
    t0 = time.time()

    for i, (vd, cl, cp) in enumerate(clips):
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        cmd = [PY, os.path.join(TOOLS, "pipeline_gnd_model.py"),
               "--out", tmp, "--frames", str(FRAMES),
               "--max-clips", "1", "--config", CONFIG]
        env = dict(os.environ)
        env["PYTHONPATH"] = TOOLS
        env["GND_FORCE_CLIP"] = cp
        try:
            # 用 env 传 clip 路径（pipeline 支持不了，改用直接调用）
            sys.path.insert(0, TOOLS)
            from pipeline_gnd_model import process_clip
            cfg = json.loads(CONFIG)
            n = process_clip(cp, tmp, cfg, FRAMES)
            if n > 0:
                merge_dir(tmp, OUT)
                ok += 1
                print("  [%d/%d] %-28s %-22s %d 帧"
                      % (i + 1, len(clips), vd[:28], cl[:22], n))
            else:
                fail += 1
                print("  [%d/%d] %-28s %-22s 0 帧"
                      % (i + 1, len(clips), vd[:28], cl[:22]))
        except Exception as e:
            fail += 1
            print("  [%d/%d] %-28s %-22s 异常 %s"
                  % (i + 1, len(clips), vd[:28], cl[:22], str(e)[:50]))

    dt = time.time() - t0
    n_f = len(glob.glob(os.path.join(OUT, "frames", "*_sim.jpg")))
    print("\n=== 完成 %.1f 分钟 ===" % (dt / 60))
    print("成功 %d / 失败 %d" % (ok, fail))
    print("产出: %d 帧" % n_f)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
