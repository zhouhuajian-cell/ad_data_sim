#!/usr/bin/env python3
"""补齐 gnd_clean_v3（4A）缺失的 clip。"""
import json, os, glob, shutil, subprocess, sys, time

sys.path.insert(0, '/opt/ue5_synth/tools')

OUT = '/data/render/gnd_clean_v3'
V = '/opt/ad_mining/venv/bin/python'
T = '/opt/ue5_synth/tools'

def env():
    e = dict(os.environ)
    e['PYTHONPATH'] = T
    e['HF_ENDPOINT'] = 'https://hf-mirror.com'
    e['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    return e

# 已完成
done = set()
for f in glob.glob(OUT + '/annotations/*.json'):
    try:
        d = json.load(open(f, encoding='utf-8'))
        done.add((d.get('batch',''), d.get('clip','')))
    except Exception:
        pass

# 全部
root = '/data/test_data/gnd_train'
allc = []
for b in sorted(os.listdir(root)):
    bd = os.path.join(root, b)
    if not os.path.isdir(bd): continue
    for c in sorted(os.listdir(bd)):
        if os.path.isdir(os.path.join(bd, c, 'img_ori')):
            allc.append((b, c))

todo = [x for x in allc if x not in done]
print('总 %d, 已完成 %d, 待补 %d' % (len(allc), len(done), len(todo)), flush=True)

import random
random.seed(20261122)
line_assets = ['lane_solid_white','lane_dashed_white','lane_solid_yellow',
               'lane_dashed_yellow','road_edge_white','double_yellow']

ok = 0
t0 = time.time()
for i, (b, c) in enumerate(todo):
    lk = random.choice(line_assets)
    ck = random.choice(['double_yellow','lane_dashed_yellow','lane_solid_white'])
    items = [
        {'asset': lk, 'x': 5, 'y': -5.25, 'len_m': 45, 'width_m': 0.20},
        {'asset': lk, 'x': 5, 'y': 5.25, 'len_m': 45, 'width_m': 0.20},
        {'asset': ck, 'x': 5, 'y': 0, 'len_m': 45, 'width_m': 0.30},
        {'asset': lk, 'x': 5, 'y': -1.75, 'len_m': 45, 'width_m': 0.15},
        {'asset': lk, 'x': 5, 'y': 1.75, 'len_m': 45, 'width_m': 0.15},
    ]
    cfg = {'batch_mode': True, 'batch': b, 'clip': c, 'out': '/tmp/fill_one',
           'frames': 20,
           'params': {'gnd_mode': 'clean', 'lanes': 3, 'dash': True,
                      'edge': 0.2, 'width': 0.15,
                      'wear': round(random.uniform(0.25, 0.45), 2),
                      'dirt': round(random.uniform(0.15, 0.35), 2),
                      'optical_blur': round(random.uniform(0.8, 1.5), 1),
                      'gnd_spec_on': True, 'gnd_items': items}}
    tmp = '/tmp/fill_one'
    if os.path.isdir(tmp): shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    cf = '/tmp/fill_cfg.json'
    with open(cf, 'w') as f:
        json.dump(cfg, f)
    r = subprocess.run([V, os.path.join(T, 'gnd_render_one.py'), '--config', cf],
                       capture_output=True, text=True, env=env(), timeout=3600)
    n = len(glob.glob(os.path.join(tmp, 'frames', '*_sim.jpg')))
    if n > 0:
        for sub in ('frames', 'clean', 'annotations'):
            sd = os.path.join(tmp, sub)
            dd = os.path.join(OUT, sub)
            if os.path.isdir(sd):
                os.makedirs(dd, exist_ok=True)
                for f2 in glob.glob(os.path.join(sd, '*')):
                    shutil.copy2(f2, dd)
        ok += 1
        print('[%d/%d] %-24s %-20s %d帧' % (i+1, len(todo), b[:24], c[:20], n), flush=True)
    else:
        print('[%d/%d] %-24s %-20s 失败 %s' % (i+1, len(todo), b[:24], c[:20],
              (r.stderr or '')[-80:].replace('\n',' ')), flush=True)

print()
print('=== 补齐完成: %d/%d, 耗时 %.1f 分钟 ===' % (ok, len(todo), (time.time()-t0)/60))
n_all = len(glob.glob(OUT + '/frames/*_sim.jpg'))
print('gnd_clean_v3 总帧数: %d' % n_all)
