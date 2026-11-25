#!/usr/bin/env python3
"""GND 全量重跑（PLAN-E 精简版 + 防OOM + 净面落盘）。

关键设计：
  1. 每个 clip 结束立即 sess.clear() + torch.cuda.empty_cache()   → 防 OOM
  2. 净面底图落盘到 <out>/clean/                                    → 后续毫秒级重绘
  3. 每帧处理完立刻释放大对象
"""
import gc
import json
import os
import glob
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, '/opt/ue5_synth/tools')

OUT = '/data/render/gnd_clean_v3'
V = '/opt/ad_mining/venv/bin/python'
T = '/opt/ue5_synth/tools'
random.seed(20261122)


def env():
    e = dict(os.environ)
    e['PYTHONPATH'] = T
    e['HF_ENDPOINT'] = 'https://hf-mirror.com'
    e['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    return e


def list_clips():
    root = '/data/test_data/gnd_train'
    out = []
    for b in sorted(os.listdir(root)):
        bd = os.path.join(root, b)
        if not os.path.isdir(bd):
            continue
        for c in sorted(os.listdir(bd)):
            if os.path.isdir(os.path.join(bd, c, 'img_ori')):
                out.append((b, c))
    return out


def main():
    clips = list_clips()
    os.makedirs(OUT, exist_ok=True)
    # ITEM-FIX: 按职责分离素材
    #   边线 y=±5.25  -> road_edge_white（路缘白线）
    #   分隔线 y=±1.75 -> lane_dashed_white / lane_solid_white
    #   中线 y=0      -> double_yellow / lane_solid_yellow
    # COLOR-FIX: 严格颜色语义（中线黄、车道线白、边线白）
    edge_assets = ['road_edge_white']
    lane_assets = ['lane_dashed_white', 'lane_solid_white']
    center_assets = ['double_yellow', 'lane_solid_yellow']

    # ---- PROD-GUARDS ----
    import hashlib

    # ① 配置快照
    manifest = {
        "version": "gnd_lock_v1",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modules": {
            "VEH-SLIM": "obj_mask 像素级掩码, 远景1px/近中景2px",
            "ORTHO-TOPHAT": "正交双1D顶帽 竖核(1,17)+横核(15,1), thr=15",
            "ROAD-IN-ENVELOPE": "road|veh 填车洞后 erode 6px",
            "ZONE-B2": "远景=road顶部55%, prob 0.15/0.05 分区",
            "DIL2-LOCK": "远景膨胀2px, 近中景3px",
        },
        "params": {"prob_near": 0.15, "prob_far": 0.05,
                   "thin_thr": 15, "far_ratio": 0.55,
                   "dilate_near": 3, "dilate_far": 2},
        "output": OUT,
    }
    try:
        with open(os.path.join(OUT, "pipeline_manifest.json"), "w",
                  encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        print("[MANIFEST] 已写入 pipeline_manifest.json")
    except Exception as e:
        print("[MANIFEST] 写失败:", str(e)[:60])

    # ② 断点续跑：扫描已有 annotations 判断哪些 clip 完成
    done_clips = set()
    for f in glob.glob(OUT + "/annotations/*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
            done_clips.add((d.get("batch", ""), d.get("clip", "")))
        except Exception:
            pass
    if done_clips:
        print("[RESUME] 已完成 %d 个 clip，将跳过" % len(done_clips))
    # 只保留真正全跑完的（20 帧）
    from collections import Counter as _C
    _cnt = _C()
    for f in glob.glob(OUT + "/annotations/*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
            _cnt[(d.get("batch", ""), d.get("clip", ""))] += 1
        except Exception:
            pass
    done_full = {k for k, v in _cnt.items() if v >= 18}
    print("[RESUME] 完整 clip: %d" % len(done_full))

    failed_queue = []
    ok = fail = skip = 0
    t0 = time.time()
    print("GND 全量生产: %d clip（锁版 gnd_lock_v1）" % len(clips), flush=True)

    for i, (b, c) in enumerate(clips):
        # RESUME-ADD: 断点续跑
        if (b, c) in done_full:
            skip += 1
            print("[%d/%d] %-24s %-20s 跳过(已完成)"
                  % (i + 1, len(clips), b[:24], c[:20]), flush=True)
            continue
        ek = random.choice(edge_assets)
        lk = random.choice(lane_assets)
        ck = random.choice(center_assets)
        items = [
            # 两条边线（路缘）
            {'asset': ek, 'x': 5, 'y': -5.25, 'len_m': 45, 'width_m': 0.20},
            {'asset': ek, 'x': 5, 'y': 5.25, 'len_m': 45, 'width_m': 0.20},
            # 中线（黄双实线或白实线）
            {'asset': ck, 'x': 5, 'y': 0, 'len_m': 45, 'width_m': 0.15},
            # 两条车道分隔线
            {'asset': lk, 'x': 5, 'y': -1.75, 'len_m': 45, 'width_m': 0.15},
            {'asset': lk, 'x': 5, 'y': 1.75, 'len_m': 45, 'width_m': 0.15},
        ]
        cfg = {
            'batch_mode': True, 'batch': b, 'clip': c,
            'out': '/tmp/gnd_final_one', 'frames': 0,
            'params': {
                'gnd_mode': 'clean', 'lanes': 3, 'dash': True,
                'edge': 0.2, 'width': 0.15,
                # ITEM-FIX: 磨损/污渍稍微降低，让标线更清晰
                'wear': round(random.uniform(0.18, 0.32), 2),
                'dirt': round(random.uniform(0.10, 0.25), 2),
                'optical_blur': round(random.uniform(0.8, 1.3), 1),
                'gnd_spec_on': True, 'gnd_items': items,
            },
        }
        tmp = '/tmp/gnd_final_one'
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        cf = '/tmp/gnd_final_cfg.json'
        with open(cf, 'w') as f:
            json.dump(cfg, f)

        t1 = time.time()
        try:
            r = subprocess.run([V, os.path.join(T, 'gnd_render_one.py'),
                                '--config', cf],
                               capture_output=True, text=True,
                               env=env(), timeout=3600)
        except subprocess.TimeoutExpired:
            fail += 1
            failed_queue.append({"batch": b, "clip": c,
                                 "reason": "timeout"})
            print("[%d/%d] %-24s %-20s 超时"
                  % (i + 1, len(clips), b[:24], c[:20]), flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        except Exception as e:
            fail += 1
            failed_queue.append({"batch": b, "clip": c,
                                 "reason": str(e)[:120]})
            print("[%d/%d] %-24s %-20s 异常 %s"
                  % (i + 1, len(clips), b[:24], c[:20], str(e)[:60]),
                  flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        dt = time.time() - t1
        n = len(glob.glob(os.path.join(tmp, 'frames', '*_sim.jpg')))

        if n > 0:
            # 落盘：frames / clean / annotations
            for sub in ('frames', 'clean', 'annotations'):
                sd = os.path.join(tmp, sub)
                dd = os.path.join(OUT, sub)
                if os.path.isdir(sd):
                    os.makedirs(dd, exist_ok=True)
                    for f2 in glob.glob(os.path.join(sd, '*')):
                        shutil.copy2(f2, dd)
            ok += 1
            print('[%d/%d] %-24s %-20s %3d帧 %.0fs'
                  % (i + 1, len(clips), b[:24], c[:20], n, dt), flush=True)
        else:
            fail += 1
            _err = (r.stderr or r.stdout or '')[-300:]
            failed_queue.append({"batch": b, "clip": c, "reason": _err})
            print('[%d/%d] %-24s %-20s 失败 %s'
                  % (i + 1, len(clips), b[:24], c[:20],
                     _err[-80:].replace('\n', ' ')), flush=True)
            # 写 failed_queue.json
            try:
                with open(os.path.join(OUT, "failed_queue.json"), "w",
                          encoding="utf-8") as qf:
                    json.dump(failed_queue, qf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # === 防 OOM：每个 clip 结束清理 ===
        shutil.rmtree(tmp, ignore_errors=True)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    n_all = len(glob.glob(OUT + '/frames/*_sim.jpg'))
    n_clean = len(glob.glob(OUT + '/clean/*_clean.jpg'))
    print()
    print('=== 完成 %.1f 分钟 ===' % ((time.time() - t0) / 60))
    print('成功 %d / 跳过 %d / 失败 %d' % (ok, skip, fail))
    print('总帧数: %d  净面: %d' % (n_all, n_clean))


if __name__ == '__main__':
    main()
