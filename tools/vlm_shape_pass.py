#!/usr/bin/env python3
"""TSR 形状预扫描：用 Qwen2-VL-2B 批量判定检测框内的标牌形状。

产出一个可跨任务复用的 JSON 缓存：
    {"<图片路径>|<x1,y1,x2,y2>": "circle" | "triangle" | ...}

设计要点：
  - 只在任务开始时加载一次模型，判完即卸载，避免与渲染争显存
  - 检测结果优先复用 walk_cache，没有才现场检测
  - 已判过的键直接命中缓存，不重复推理
"""
import glob
import json
import os
import sys
import time
import hashlib
import tempfile

import cv2

sys.path.insert(0, '/opt/ue5_synth/tools')

DEFAULT_CACHE = '/data/avsynth/state/tsr_shape_cache.json'
_MODEL_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_shape_vlm.py')
try:
    with open(_MODEL_SOURCE, 'rb') as _f:
        _CACHE_REVISION = hashlib.sha256(_f.read()).hexdigest()[:16]
except OSError:
    _CACHE_REVISION = 'missing-model-script'


def key_of(path, bbox):
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    try:
        st = os.stat(path)
        fingerprint = '%d:%d' % (st.st_size, st.st_mtime_ns)
    except OSError:
        fingerprint = 'missing'
    return '%s|%s|%s|%d,%d,%d,%d' % (_CACHE_REVISION, fingerprint, path, x1, y1, x2, y2)


def load_cache(path):
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_cache(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.tsr_shape_', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def detect_for(path, conf, detector_key=None):
    """优先用 walk_cache，未命中则现场检测。"""
    dets = None
    try:
        import walk_cache
        try:
            dets = walk_cache.get_scan(path, conf=conf, detector=detector_key)
        except TypeError:
            dets = walk_cache.get_scan(path)
    except Exception:
        dets = None
    if dets is None:
        from replace_sign import detect_signs
        dets = detect_signs(path, conf)
        try:
            import walk_cache as wc
            try:
                wc.put_scan(path, dets, conf=conf, detector=detector_key)
            except TypeError:
                wc.put_scan(path, dets)
        except Exception:
            pass
    return dets or []


def build_shape_map(files, min_sign=12, conf=0.15, cache_path=DEFAULT_CACHE,
                    detector_key=None, batch=1, max_new=4000, verbose=True):
    """返回 {key: label}，同时把新结果写回缓存。"""
    cache = load_cache(cache_path)
    todo = []          # (key, inner_crop)
    hit = 0
    t0 = time.time()

    seen = set()
    for p in files:
        for d in detect_for(p, conf, detector_key):
            try:
                w = float(d.get('w') or 0)
                h = float(d.get('h') or 0)
                if min(w, h) < min_sign:
                    continue
                bb = d['bbox']
            except Exception:
                continue
            k = key_of(p, bb)
            if k in cache:
                hit += 1
                continue
            if k in seen:
                continue
            seen.add(k)
            todo.append(k)
            if len(todo) >= max_new:
                break
        if len(todo) >= max_new:
            break

    if verbose:
        print('  [VLM] 待判 %d 个，缓存命中 %d 个' % (len(todo), hit), flush=True)
    if not todo:
        return cache

    import sign_shape_vlm as vlm

    # 第二遍：按 todo 顺序裁剪并推理（保持路径顺序稳定）
    by_key = {}
    for p in files:
        for d in detect_for(p, conf, detector_key):
            try:
                if min(float(d.get('w') or 0), float(d.get('h') or 0)) < min_sign:
                    continue
                bb = d['bbox']
            except Exception:
                continue
            k = key_of(p, bb)
            if k not in todo or k in by_key:
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bb]
            # VLM-USE-BETTER: 带一圈背景上下文，模型才能看清牌的边界
            _cx, _cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            _s = max(x2 - x1, y2 - y1, 6) * 1.9
            a1 = int(max(0, _cx - _s / 2)); b1 = int(max(0, _cy - _s / 2))
            a2 = int(min(img.shape[1], _cx + _s / 2)); b2 = int(min(img.shape[0], _cy + _s / 2))
            inner = img[b1:b2, a1:a2]
            if inner.size == 0:
                continue
            by_key[k] = inner

    keys = [k for k in todo if k in by_key]
    if verbose:
        print('  [VLM] 实际裁剪 %d 个' % len(keys), flush=True)

    # The detector and VLM share the GPU. Unload our detector before loading
    # the VLM; detector weights will be loaded on demand again during rendering.
    detector = sys.modules.get('replace_sign')
    if detector is not None:
        getattr(detector, '_YOLO_MODELS', {}).clear()
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print('  [VLM] GPU free %.2f/%.2f GiB' % (free/2**30,total/2**30), flush=True)
            if free < 6 * 2**30:
                raise RuntimeError('VLM skipped: insufficient free GPU memory')
        preds = vlm.classify([by_key[k] for k in keys], batch_size=max(1, min(int(batch), 1)))
    finally:
        # Also release partially initialized models after an exception/OOM.
        vlm.release()
    if len(preds) != len(keys):
        raise RuntimeError('VLM result count mismatch')
    for k, (lab, _raw) in zip(keys, preds):
        cache[k] = lab

    save_cache(cache_path, cache)
    if verbose:
        from collections import Counter
        print('  [VLM] 完成 %d 个，耗时 %.1fs，分布 %s'
              % (len(keys), time.time() - t0, dict(Counter(preds_i[0] for preds_i in preds))),
              flush=True)
    return cache


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--min-sign', type=int, default=12)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--cache', default=DEFAULT_CACHE)
    ap.add_argument('--limit', type=int, default=4000)
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(a.images, '*.jpg')))
    m = build_shape_map(fs, a.min_sign, a.conf, a.cache, max_new=a.limit)
    from collections import Counter
    print('缓存总量:', len(m), dict(Counter(m.values())))
