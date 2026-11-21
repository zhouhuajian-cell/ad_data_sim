#!/usr/bin/env python3
"""批量跑优质场景的 OBJ 仿真（add 模式）。

只保留：空位 >40%，无大框，最近车 >12m 的场景。
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "/opt/ue5_synth/tools")

OUT = "/data/render/obj_good"
PY = "/opt/ad_mining/venv/bin/python"
TOOLS = "/opt/ue5_synth/tools"


def score_clips():
    """给所有 clip 打分，返回优质列表。"""
    import numpy as np
    import render_core as rc
    from real_calib import load_cam_real

    ann_dir = "/data/render/final_obj3/annotations"
    files = sorted(glob.glob(ann_dir + "/*.json"))
    seen = set()
    results = []

    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        key = (d.get("batch"), d.get("clip"))
        if key in seen:
            continue
        seen.add(key)
        batch, clip = key

        cam = load_cam_real(batch, clip)
        if cam is None:
            continue
        objs = [o for o in (d.get("objects") or [])
                if o.get("source") == "real"
                and (o.get("bbox2d") or [0, 0, 0, 0])[2] > 0]
        if not objs:
            continue

        H = d.get("height", 1080)
        W = d.get("width", 1920)
        boxes = [o["bbox2d"] for o in objs]

        try:
            drv = rc.build_drivable((H, W), calib=None, cam=cam)
            spawn = rc.filter_boxes_for_spawn(drv, boxes,
                                              margin=18, check=(40, 30))
        except Exception:
            continue

        def frac(m):
            return float(np.count_nonzero(m)) / m.size * 100.0

        big = sum(1 for b in boxes
                  if (b[2] - b[0]) * (b[3] - b[1]) > H * W * 0.05)
        xs = [(o.get("box3d") or {}).get("x", 999) for o in objs]
        nearest = min(xs) if xs else 999
        sp = frac(spawn)

        if sp > 40 and big == 0 and nearest > 12:
            results.append({"batch": batch, "clip": clip,
                            "n": len(objs), "spawn": sp,
                            "nearest": nearest})

    results.sort(key=lambda r: -r["spawn"])
    return results


def merge_dir(src, dst):
    for sub in ("frames", "annotations", "ignore"):
        s = os.path.join(src, sub)
        d = os.path.join(dst, sub)
        if not os.path.isdir(s):
            continue
        os.makedirs(d, exist_ok=True)
        for fn in os.listdir(s):
            sp = os.path.join(s, fn)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(d, fn))


def main():
    clips = score_clips()
    print("优质场景: %d 个" % len(clips))
    if not clips:
        raise SystemExit("无优质场景")

    os.makedirs(OUT, exist_ok=True)
    tmp = "/tmp/obj_good_one"

    ok = 0
    fail = 0
    t0 = time.time()
    total_obj = 0

    for i, c in enumerate(clips):
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)

        cfg = ('{"edit_mode":"add","count":3,"cls":"car",'
               '"dist_min":15,"dist_max":45,"lat":5.0,"shadow":0.65}')
        cmd = [PY, os.path.join(TOOLS, "pipeline_sim.py"),
               "--batch", c["batch"], "--clip", c["clip"],
               "--module", "obj", "--frames", "0",
               "--out", tmp, "--config", cfg]
        env = dict(os.environ)
        env["PYTHONPATH"] = TOOLS
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=1800, env=env)
            n_sim = len(glob.glob(os.path.join(tmp, "frames", "*_sim.jpg")))
            if r.returncode == 0 and n_sim > 0:
                merge_dir(tmp, OUT)
                # 统计合成数
                anns = sorted(glob.glob(
                    os.path.join(tmp, "annotations", "*.json")))
                n_syn = 0
                if anns:
                    try:
                        a = json.load(open(anns[0], encoding="utf-8"))
                        n_syn = (a.get("meta") or {}).get("n_synth", 0)
                    except Exception:
                        pass
                total_obj += n_syn * n_sim
                ok += 1
                print("  [%d/%d] %-32s %-22s %3d帧 %d个/帧 空位%.0f%%"
                      % (i + 1, len(clips), c["batch"][:32],
                         c["clip"][:22], n_sim, n_syn, c["spawn"]))
            else:
                fail += 1
                err = (r.stderr or r.stdout or "")[-120:]
                print("  [%d/%d] %-32s %-22s 失败 %s"
                      % (i + 1, len(clips), c["batch"][:32],
                         c["clip"][:22], err.replace("\n", " ")[:60]))
        except subprocess.TimeoutExpired:
            fail += 1
            print("  [%d/%d] %s 超时" % (i + 1, len(clips), c["clip"][:22]))
        except Exception as e:
            fail += 1
            print("  [%d/%d] %s 异常 %s"
                  % (i + 1, len(clips), c["clip"][:22], str(e)[:50]))

    dt = time.time() - t0
    n_f = len(glob.glob(os.path.join(OUT, "frames", "*_sim.jpg")))
    n_a = len(glob.glob(os.path.join(OUT, "annotations", "*.json")))
    n_i = len(glob.glob(os.path.join(OUT, "ignore", "*.png")))

    print("\n=== 完成 %.1f 分钟 ===" % (dt / 60))
    print("成功 %d / 失败 %d" % (ok, fail))
    print("产出: %d 帧 / %d 标注 / %d ignore 掩码" % (n_f, n_a, n_i))
    print("合成物体总数: 约 %d" % total_obj)

    if os.path.isdir(tmp):
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
