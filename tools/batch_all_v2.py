#!/usr/bin/env python3
"""三模块全量重跑：OBJ / GND / TSR，随机素材。

用法:
  python3 batch_all_v2.py obj   # 只跑 OBJ
  python3 batch_all_v2.py gnd
  python3 batch_all_v2.py all
"""
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "/opt/ue5_synth/tools")

PY = "/opt/ad_mining/venv/bin/python"
TOOLS = "/opt/ue5_synth/tools"
ASSET_REAL = "/data/obj_assets_real"
GND_ASSETS = "/data/gnd_assets"

random.seed(20261121)


def _list_sprites(cls):
    """某类别的所有 sprite 路径。"""
    out = []
    for root in (ASSET_REAL, "/data/obj_assets_pc", "/data/obj_assets"):
        d = os.path.join(root, cls)
        if os.path.isdir(d):
            out.extend(sorted(glob.glob(os.path.join(d, "*.png"))))
    return out


def _pick_sprites(cls, n):
    """随机抽 n 个不重复的 sprite。"""
    pool = _list_sprites(cls)
    if not pool:
        return []
    random.shuffle(pool)
    return pool[:n]


def _list_gnd_clips():
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
                out.append((vd, cl))
    return out


def _obj_clips_from_existing():
    """从 obj_good 现有标注里提取 clip 列表（场景筛选结果）。"""
    seen = {}
    for f in glob.glob("/data/render/obj_good/annotations/*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            k = (d.get("batch"), d.get("clip"))
            seen[k] = seen.get(k, 0) + 1
        except Exception:
            pass
    return [{"batch": k[0], "clip": k[1], "n": v}
            for k, v in sorted(seen.items()) if k[0] and k[1]]


def _merge_dir(src, dst):
    for sub in ("frames", "annotations", "ignore", "clean"):
        s = os.path.join(src, sub)
        d = os.path.join(dst, sub)
        if not os.path.isdir(s):
            continue
        os.makedirs(d, exist_ok=True)
        for fn in os.listdir(s):
            sp = os.path.join(s, fn)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(d, fn))


def _run(cmd, timeout=3600):
    env = dict(os.environ)
    env["PYTHONPATH"] = TOOLS
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env)


# ---------------- OBJ ----------------

def run_obj():
    OUT = "/data/render/obj_good_v2"
    tmp = "/tmp/obj_v2_one"
    clips = _obj_clips_from_existing()
    print("=" * 60)
    print("OBJ 重跑: %d 个 clip（随机素材）" % len(clips))
    print("=" * 60)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    ok = fail = total_syn = 0
    t0 = time.time()

    for i, c in enumerate(clips):
        # 随机挑 3 个 car sprite
        sprites = _pick_sprites("car", 3)
        if not sprites:
            print("  [%d/%d] 无 car 素材，跳过" % (i + 1, len(clips)))
            fail += 1
            continue

        # 每个物体用不同 sprite
        items = []
        for k, sp in enumerate(sprites):
            items.append({"sprite_path": sp})

        cfg = {
            "edit_mode": "add",
            "count": len(sprites),
            "cls": "car",
            "dist_min": 15, "dist_max": 45, "lat": 5.0,
            "shadow": 0.65,
            "sprite_path": sprites[0],   # 主素材
            "random_sprites": sprites,   # 随机素材池
        }

        if os.path.isdir(tmp):
            shutil.rmtree(tmp)

        cmd = [PY, os.path.join(TOOLS, "pipeline_sim.py"),
               "--batch", c["batch"], "--clip", c["clip"],
               "--module", "obj", "--frames", "0",
               "--out", tmp,
               "--config", json.dumps(cfg, ensure_ascii=False)]

        try:
            r = _run(cmd)
            n_sim = len(glob.glob(os.path.join(tmp, "frames", "*_sim.jpg")))
            if r.returncode == 0 and n_sim > 0:
                _merge_dir(tmp, OUT)
                anns = sorted(glob.glob(
                    os.path.join(tmp, "annotations", "*.json")))
                n_syn = 0
                if anns:
                    try:
                        with open(anns[0], encoding="utf-8") as fh:
                            a = json.load(fh)
                        n_syn = (a.get("meta") or {}).get("n_synth", 0)
                    except Exception:
                        pass
                total_syn += n_syn * n_sim
                ok += 1
                print("  [%d/%d] %-40s %-22s %3d帧 %d个/帧 素材=%s"
                      % (i + 1, len(clips), c["batch"][:40],
                         c["clip"][:22], n_sim, n_syn,
                         ",".join(os.path.basename(s)[:14]
                                  for s in sprites[:2])))
            else:
                fail += 1
                err = (r.stderr or r.stdout or "")[-140:]
                print("  [%d/%d] %-40s 失败 %s"
                      % (i + 1, len(clips), c["clip"][:22],
                         err.replace("\n", " ")[:70]))
        except Exception as e:
            fail += 1
            print("  [%d/%d] %s 异常 %s"
                  % (i + 1, len(clips), c["clip"][:22], str(e)[:60]))

    dt = (time.time() - t0) / 60
    n_f = len(glob.glob(os.path.join(OUT, "frames", "*_sim.jpg")))
    n_a = len(glob.glob(os.path.join(OUT, "annotations", "*.json")))
    n_i = len(glob.glob(os.path.join(OUT, "ignore", "*.png")))
    print("\n=== OBJ 完成 %.1f 分钟 ===" % dt)
    print("成功 %d / 失败 %d" % (ok, fail))
    print("产出: %d 帧 / %d 标注 / %d ignore" % (n_f, n_a, n_i))
    print("合成物体: 约 %d" % total_syn)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)


# ---------------- GND ----------------

def run_gnd():
    OUT = "/data/render/gnd_clean_v2"
    tmp = "/tmp/gnd_v2_one"
    clips = _list_gnd_clips()
    print("=" * 60)
    print("GND 重跑: %d 个 clip（随机素材）" % len(clips))
    print("=" * 60)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    # 可用于"指定位置"的素材池
    line_assets = ["lane_solid_white", "lane_dashed_white",
                   "lane_solid_yellow", "lane_dashed_yellow",
                   "road_edge_white", "double_yellow"]
    extra_assets = ["arrow_straight", "arrow_left", "arrow_right",
                    "stop_line", "crosswalk_zebra", "hatch_zone"]

    ok = fail = 0
    t0 = time.time()

    for i, (batch, clip) in enumerate(clips):
        # 随机选一套线型
        lane_key = random.choice(line_assets)
        center_key = random.choice(["double_yellow", "lane_dashed_yellow",
                                    "lane_solid_white"])

        items = [
            {"asset": lane_key, "x": 5.0, "y": -5.25,
             "len_m": 45, "width_m": 0.20},
            {"asset": lane_key, "x": 5.0, "y": 5.25,
             "len_m": 45, "width_m": 0.20},
            {"asset": center_key, "x": 5.0, "y": 0.0,
             "len_m": 45, "width_m": 0.30},
            {"asset": lane_key, "x": 5.0, "y": -1.75,
             "len_m": 45, "width_m": 0.15},
            {"asset": lane_key, "x": 5.0, "y": 1.75,
             "len_m": 45, "width_m": 0.15},
        ]
        # 30% 概率加一个额外标记（箭头/停止线等）
        if random.random() < 0.3:
            items.append({"asset": random.choice(extra_assets),
                          "x": round(random.uniform(15, 30), 1),
                          "y": round(random.uniform(-3, 3), 1),
                          "len_m": 4.0, "width_m": 2.0})

        cfg = {
            "gnd_mode": "clean",
            "lanes": 3, "dash": True, "edge": 0.2, "width": 0.15,
            "wear": round(random.uniform(0.25, 0.45), 2),
            "dirt": round(random.uniform(0.15, 0.35), 2),
            "optical_blur": round(random.uniform(0.8, 1.5), 1),
            "gnd_spec_on": True,
            "gnd_items": items,
        }

        if os.path.isdir(tmp):
            shutil.rmtree(tmp)

        cmd = [PY, os.path.join(TOOLS, "pipeline_sim.py"),
               "--batch", batch, "--clip", clip,
               "--module", "gnd", "--frames", "0",
               "--out", tmp,
               "--config", json.dumps(cfg, ensure_ascii=False)]

        try:
            r = _run(cmd)
            n_sim = len(glob.glob(os.path.join(tmp, "frames", "*_sim.jpg")))
            if r.returncode == 0 and n_sim > 0:
                _merge_dir(tmp, OUT)
                ok += 1
                print("  [%d/%d] %-28s %-22s %3d帧 线=%s 磨损%.2f"
                      % (i + 1, len(clips), batch[:28], clip[:22],
                         n_sim, lane_key[:14], cfg["wear"]))
            else:
                fail += 1
                err = (r.stderr or r.stdout or "")[-140:]
                print("  [%d/%d] %-28s 失败 %s"
                      % (i + 1, len(clips), clip[:22],
                         err.replace("\n", " ")[:70]))
        except Exception as e:
            fail += 1
            print("  [%d/%d] %s 异常 %s"
                  % (i + 1, len(clips), clip[:22], str(e)[:60]))

    dt = (time.time() - t0) / 60
    n_f = len(glob.glob(os.path.join(OUT, "frames", "*_sim.jpg")))
    n_c = len(glob.glob(os.path.join(OUT, "clean", "*_clean.jpg")))
    print("\n=== GND 完成 %.1f 分钟 ===" % dt)
    print("成功 %d / 失败 %d" % (ok, fail))
    print("产出: %d 帧 / %d 净面底图" % (n_f, n_c))
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)


# ---------------- TSR ----------------

def run_tsr():
    """TSR 用现有的 sign 素材随机替换。"""
    OUT = "/data/render/tsr_v2"
    print("=" * 60)
    print("TSR 重跑")
    print("=" * 60)

    # 检查 TSR 入口
    candidates = [
        os.path.join(TOOLS, "batch_replace.py"),
        os.path.join(TOOLS, "pipeline6.py"),
        os.path.join(TOOLS, "tsr_detect_api.py"),
    ]
    found = [c for c in candidates if os.path.isfile(c)]
    print("可用 TSR 脚本: %s" % [os.path.basename(c) for c in found])

    # 检查素材
    sign_dirs = [d for d in ("/data/sign_assets", "/data/assets/signs",
                             "/data/tsr_assets")
                 if os.path.isdir(d)]
    print("sign 素材目录: %s" % sign_dirs)

    # TSR 数据源
    src = "/data/test_data/tsr_train"
    if not os.path.isdir(src):
        src = "/data/render/tsr_final_1000"
    print("TSR 数据源: %s (%s)" % (src, "存在" if os.path.isdir(src) else "不存在"))

    print("\nTSR 未配置批量入口，需人工确认数据源后单独跑。")


if __name__ == "__main__":
    what = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    if what in ("obj", "all"):
        run_obj()
    if what in ("gnd", "all"):
        run_gnd()
    if what in ("tsr", "all"):
        run_tsr()
