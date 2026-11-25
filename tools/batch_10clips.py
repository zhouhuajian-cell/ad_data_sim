#!/usr/bin/env python3
"""三模块各抽 10 个 clip 批量出图。"""
import json
import os
import random
import shutil
import subprocess
import sys
import time
import glob

sys.path.insert(0, "/opt/ue5_synth/tools")

VENV = "/opt/ad_mining/venv/bin/python"
TOOLS = "/opt/ue5_synth/tools"
random.seed(20261122)


def _env():
    e = dict(os.environ)
    e["PYTHONPATH"] = TOOLS
    e["HF_ENDPOINT"] = "https://hf-mirror.com"
    e["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return e


def pick_gnd(n=10):
    root = "/data/test_data/gnd_train"
    clips = []
    for b in sorted(os.listdir(root)):
        bd = os.path.join(root, b)
        if not os.path.isdir(bd):
            continue
        for c in sorted(os.listdir(bd)):
            cd = os.path.join(bd, c)
            if os.path.isdir(os.path.join(cd, "img_ori")):
                clips.append((b, c))
    random.shuffle(clips)
    return clips[:n]


_YOLO = None


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO("/opt/ad_mining/tsr_sign.pt")
    return _YOLO


def _clip_has_signs(batch, clip):
    """PICK-WITH-YOLO: 用 tsr_sign.pt 真实预检（和 batch_replace 同模型）。"""
    import glob as _g
    import cv2 as _cv
    for root in ("/data/test_data/real_clips",
                 "/data/test_data/real_clips_extra"):
        src = os.path.join(root, batch, clip, "cam_st_x2f_front")
        if not os.path.isdir(src):
            continue
        imgs = sorted(_g.glob(os.path.join(src, "*.jpg")))[:3]
        if not imgs:
            return False
        try:
            m = _get_yolo()
            for p in imgs:
                im = _cv.imread(p)
                if im is None:
                    continue
                r = m(im, conf=0.25, verbose=False)[0]
                if len(r.boxes) > 0:
                    return True
        except Exception:
            return False
        return False
    return False


def pick_tsr_signs(n=10):
    """专给 TSR 用：只挑有标牌的 clip。"""
    allc = pick_obj_tsr(999)
    out = []
    for b, c in allc:
        if _clip_has_signs(b, c):
            out.append((b, c))
            print("   有标牌: %s / %s" % (b[:34], c[:22]))
        if len(out) >= n:
            break
    return out


def pick_obj_tsr(n=10):
    clips = []
    for root in ("/data/test_data/real_clips",
                 "/data/test_data/real_clips_extra"):
        if not os.path.isdir(root):
            continue
        for b in sorted(os.listdir(root)):
            bd = os.path.join(root, b)
            if not os.path.isdir(bd):
                continue
            for c in sorted(os.listdir(bd)):
                cd = os.path.join(bd, c)
                if os.path.isdir(os.path.join(cd, "cam_st_x2f_front")):
                    clips.append((b, c))
    random.shuffle(clips)
    return clips[:n]


def run_gnd(clips, out):
    os.makedirs(out, exist_ok=True)
    tmp = "/tmp/b10_gnd"
    ok = fail = 0
    t0 = time.time()
    line_assets = ["lane_solid_white", "lane_dashed_white",
                   "lane_solid_yellow", "lane_dashed_yellow",
                   "road_edge_white", "double_yellow"]
    for i, (b, c) in enumerate(clips):
        lk = random.choice(line_assets)
        ck = random.choice(["double_yellow", "lane_dashed_yellow",
                            "lane_solid_white"])
        items = [
            {"asset": lk, "x": 5, "y": -5.25, "len_m": 45, "width_m": 0.20},
            {"asset": lk, "x": 5, "y": 5.25, "len_m": 45, "width_m": 0.20},
            {"asset": ck, "x": 5, "y": 0, "len_m": 45, "width_m": 0.30},
            {"asset": lk, "x": 5, "y": -1.75, "len_m": 45, "width_m": 0.15},
            {"asset": lk, "x": 5, "y": 1.75, "len_m": 45, "width_m": 0.15},
        ]
        cfg = {"batch_mode": True, "batch": b, "clip": c, "out": tmp,
               "frames": 20,
               "params": {"gnd_mode": "clean", "lanes": 3, "dash": True,
                          "edge": 0.2, "width": 0.15,
                          "wear": round(random.uniform(0.25, 0.45), 2),
                          "dirt": round(random.uniform(0.15, 0.35), 2),
                          "optical_blur": round(random.uniform(0.8, 1.5), 1),
                          "gnd_spec_on": True, "gnd_items": items}}
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        cf = "/tmp/b10_gnd_cfg.json"
        with open(cf, "w") as f:
            json.dump(cfg, f)
        r = subprocess.run([VENV, os.path.join(TOOLS, "gnd_render_one.py"),
                            "--config", cf],
                           capture_output=True, text=True,
                           env=_env(), timeout=3600)
        n = len(glob.glob(os.path.join(tmp, "frames", "*_sim.jpg")))
        if n > 0:
            ok += 1
            print("  [%d/%d] %-26s %-22s %3d帧 %s"
                  % (i + 1, len(clips), b[:26], c[:22], n, lk[:14]))
            # 合并
            for sub in ("frames", "clean", "annotations"):
                sd = os.path.join(tmp, sub)
                dd = os.path.join(out, sub)
                if os.path.isdir(sd):
                    os.makedirs(dd, exist_ok=True)
                    for f2 in glob.glob(os.path.join(sd, "*")):
                        shutil.copy2(f2, dd)
        else:
            fail += 1
            print("  [%d/%d] %-26s 失败 %s"
                  % (i + 1, len(clips), c[:22],
                     (r.stderr or "")[-80:].replace("\n", " ")))
    return ok, fail, time.time() - t0


def run_obj(clips, out):
    os.makedirs(out, exist_ok=True)
    tmp = "/tmp/b10_obj"
    ok = fail = 0
    t0 = time.time()
    pool = glob.glob("/data/obj_assets_real/car/*.png") + \
           glob.glob("/data/obj_assets_real/car/*.jpg")
    for i, (b, c) in enumerate(clips):
        cfg = {"module": "obj", "frames": 0}
        if pool:
            cfg["sprite_path"] = random.sample(pool, min(3, len(pool)))
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        cf = "/tmp/b10_obj_cfg.json"
        with open(cf, "w") as f:
            json.dump(cfg, f)
        cmd = [VENV, os.path.join(TOOLS, "pipeline_sim.py"),
               "--batch", b, "--clip", c, "--module", "obj",
               "--frames", "20", "--out", tmp,
               "--config", json.dumps(cfg)]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env=_env(), timeout=3600)
        n = len(glob.glob(os.path.join(tmp, "frames", "*_sim.jpg")))
        if n > 0:
            ok += 1
            print("  [%d/%d] %-26s %-22s %3d帧"
                  % (i + 1, len(clips), b[:26], c[:22], n))
            for sub in ("frames", "annotations", "ignore"):
                sd = os.path.join(tmp, sub)
                dd = os.path.join(out, sub)
                if os.path.isdir(sd):
                    os.makedirs(dd, exist_ok=True)
                    for f2 in glob.glob(os.path.join(sd, "*")):
                        shutil.copy2(f2, dd)
        else:
            fail += 1
            print("  [%d/%d] %-26s 失败 %s"
                  % (i + 1, len(clips), c[:22],
                     (r.stderr or "")[-80:].replace("\n", " ")))
    return ok, fail, time.time() - t0


def run_tsr(clips, out):
    os.makedirs(out, exist_ok=True)
    ok = fail = 0
    t0 = time.time()
    countries = sorted([d for d in os.listdir("/data/tsr_library_v2")
                        if os.path.isdir("/data/tsr_library_v2/" + d)])
    for i, (b, c) in enumerate(clips):
        # PATH-FIX: 两个根目录都要找
        src = None
        for root in ("/data/test_data/real_clips",
                     "/data/test_data/real_clips_extra"):
            cand = os.path.join(root, b, c, "cam_st_x2f_front")
            if os.path.isdir(cand):
                src = cand
                break
        if src is None:
            fail += 1
            print("  [%d/%d] %-26s 目录不存在" % (i + 1, len(clips), c[:22]))
            continue
        country = random.choice(countries)
        tmpin = "/tmp/b10_tsr_in"
        if os.path.isdir(tmpin):
            shutil.rmtree(tmpin)
        os.makedirs(tmpin, exist_ok=True)
        imgs = sorted(glob.glob(os.path.join(src, "*.jpg")))[:20]
        for f2 in imgs:
            shutil.copy2(f2, tmpin)
        cmd = [VENV, os.path.join(TOOLS, "batch_replace.py"),
               "--images", tmpin,
               "--signs-dir", "/data/tsr_library_v2",
               "--country", country,
               "--all-signs", "--vary-class", "--v6",
               "--limit", "20",
               "--out", out]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env=_env(), timeout=3600)
        n = len(glob.glob(os.path.join(out, "*_composed.jpg")))
        if r.returncode == 0:
            ok += 1
            tail = [x for x in (r.stdout or "").strip().split("\n")
                    if "完成" in x]
            print("  [%d/%d] %-26s %-18s %s"
                  % (i + 1, len(clips), b[:26], country,
                     tail[-1] if tail else ""))
        else:
            fail += 1
            print("  [%d/%d] %-26s 失败 %s"
                  % (i + 1, len(clips), c[:22],
                     (r.stderr or "")[-80:].replace("\n", " ")))
    return ok, fail, time.time() - t0


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("gnd", "all"):
        print("=" * 60)
        print("GND 10 clips")
        cl = pick_gnd(10)
        o, f, t = run_gnd(cl, "/data/render/gnd_v3_10")
        print("GND: 成功 %d / 失败 %d  %.1f 分钟" % (o, f, t / 60))
    if which in ("obj", "all"):
        print("=" * 60)
        print("OBJ 10 clips")
        cl = pick_obj_tsr(10)
        o, f, t = run_obj(cl, "/data/render/obj_v3_10")
        print("OBJ: 成功 %d / 失败 %d  %.1f 分钟" % (o, f, t / 60))
    if which in ("tsr", "all"):
        print("=" * 60)
        print("TSR 10 clips（只选有标牌的）")
        cl = pick_tsr_signs(10)
        o, f, t = run_tsr(cl, "/data/render/tsr_v3_10")
        print("TSR: 成功 %d / 失败 %d  %.1f 分钟" % (o, f, t / 60))
