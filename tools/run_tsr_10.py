#!/usr/bin/env python3
"""跑 10 个 TSR clip + 统计 exact/fallback 匹配占比。"""
import sys, os, glob, shutil, subprocess, random, json
sys.path.insert(0, '/opt/ue5_synth/tools')
import batch_10clips as B

random.seed(20261122)
cl = B.pick_tsr_signs(10)
V = '/opt/ad_mining/venv/bin/python'
T = '/opt/ue5_synth/tools'

def env():
    e = dict(os.environ)
    e['PYTHONPATH'] = T
    e['HF_ENDPOINT'] = 'https://hf-mirror.com'
    e['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    return e

cs = sorted([d for d in os.listdir('/data/tsr_library_v2')
             if os.path.isdir('/data/tsr_library_v2/' + d)])
out = '/data/render/tsr_v3_10'
os.makedirs(out, exist_ok=True)
total = 0
for i, (b, c) in enumerate(cl):
    src = None
    for root in ('/data/test_data/real_clips',
                 '/data/test_data/real_clips_extra'):
        cand = os.path.join(root, b, c, 'cam_st_x2f_front')
        if os.path.isdir(cand):
            src = cand
            break
    if src is None:
        continue
    co = random.choice(cs)
    tmp = '/tmp/t10f_in'
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for f in sorted(glob.glob(os.path.join(src, '*.jpg')))[:20]:
        shutil.copy2(f, tmp)
    cmd = [V, os.path.join(T, 'batch_replace.py'), '--images', tmp,
           '--signs-dir', '/data/tsr_library_v2', '--country', co,
           '--all-signs', '--vary-class', '--v6', '--limit', '20',
           '--out', out]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env=env(), timeout=3600)
    t = [x for x in (r.stdout or '').split('\n') if x.startswith('完成')]
    print('[%d/10] %-32s %-4s %s' % (i + 1, b[:32], co,
                                     t[-1] if t else '?'), flush=True)
    total += 1

n = len(glob.glob(os.path.join(out, '*_composed.jpg')))
print()
print('=== 总产出 %d 张 ===' % n)
