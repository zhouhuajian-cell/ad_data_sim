#!/usr/bin/env python3
"""用 Qwen2-VL-2B 对交通标牌小图做形状判别（离线批量）。

输出严格枚举：circle / triangle / square / diamond / hexagon / octagon / rect / unknown
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import os as _os
_os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL = ('/root/.cache/huggingface/hub/'
         'models--Qwen--Qwen2-VL-2B-Instruct/snapshots/'
         '895c3a49bc3fa70a340399125c650a463535e71c')

ENUM = ['circle', 'triangle', 'square', 'diamond',
        'hexagon', 'octagon', 'rect', 'unknown']

# VLM-USE-BETTER: 详细提示 + 更高分辨率（实测 224px 简提示 unknown 占 6 成，
# 384~512px 详提示在 26~42px 的六边形牌上 14/14 判对）
PROMPT = (
    '图里是一块交通标志牌。请只判断牌子的外轮廓形状，忽略文字内容。\n'
    '参照：\n'
    '- 限速、禁止、指示牌多为圆形\n'
    '- 警示牌多为三角形，也有菱形\n'
    '- 指路牌、信息牌多为矩形\n'
    '规则：即使招牌有透视倾斜、模糊、部分遮挡，也请选最接近的形状；'
    '只有完全看不清时才答 unknown。\n'
    '只输出一个词：circle / triangle / square / diamond / hexagon / octagon / rect / unknown\n'
    '形状=')

_MODEL = None
_PROC = None


def load_model():
    global _MODEL, _PROC
    if _MODEL is not None:
        return _MODEL, _PROC
    _PROC = AutoProcessor.from_pretrained(MODEL, local_files_only=True,
                                          min_pixels=64 * 28 * 28,
                                          max_pixels=768 * 28 * 28)
    _MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map='cuda',
        attn_implementation='sdpa').eval()
    return _MODEL, _PROC


def release():
    """卸载模型，释放显存。"""
    global _MODEL, _PROC
    _MODEL = None
    _PROC = None
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def upscale(img, size=384, pad=10):
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return np.full((size, size, 3), 255, np.uint8)
    s = float(size - 2 * pad) / max(h, w)
    img = cv2.resize(img, (max(2, int(round(w * s))),
                           max(2, int(round(h * s)))),
                     interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y = (size - img.shape[0]) // 2
    x = (size - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def classify(images_bgr, batch_size=1):
    """images_bgr: list of BGR numpy arrays → list of (label, raw)"""
    model, proc = load_model()
    out = []
    for i in range(0, len(images_bgr), batch_size):
        chunk = images_bgr[i:i + batch_size]
        texts, imgs = [], []
        for im in chunk:
            rgb = cv2.cvtColor(upscale(im), cv2.COLOR_BGR2RGB)
            msgs = [{'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': PROMPT}]}]
            texts.append(proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            imgs.append(rgb)
        inputs = proc(text=texts, images=imgs, return_tensors='pt',
                      padding=True).to('cuda')
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=6,
                                 do_sample=False,
                                 temperature=None, top_p=None)
        trimmed = gen[:, inputs['input_ids'].shape[1]:]
        raws = proc.batch_decode(trimmed, skip_special_tokens=True)
        for raw in raws:
            low = str(raw).strip().lower()
            hit = 'unknown'
            for e in ENUM:
                if e in low:
                    hit = e
                    break
            out.append((hit, low[:24]))
    return out


def crop_from_bbox(img, bbox, pad_ratio=0.45):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s = max(x2 - x1, y2 - y1, 6) * (1.0 + pad_ratio * 2)
    a1 = int(max(0, cx - s / 2)); b1 = int(max(0, cy - s / 2))
    a2 = int(min(img.shape[1], cx + s / 2)); b2 = int(min(img.shape[0], cy + s / 2))
    c = img[b1:b2, a1:a2]
    return c if c.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', nargs='*', default=[],
                    help='任务目录，从 annotations 取 source_bbox 裁剪原图')
    ap.add_argument('--limit', type=int, default=30)
    ap.add_argument('--sheet', default='/tmp/shapetest/sheet.png')
    ap.add_argument('--json-out', default='/tmp/shapetest/pred.json')
    args = ap.parse_args()

    items = []
    for job in args.jobs:
        ann = os.path.join(job, 'annotations')
        for f in sorted(glob.glob(ann + '/*.json')):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            src = j.get('source_path')
            if not (src and os.path.isfile(src)):
                continue
            img = cv2.imread(src)
            if img is None:
                continue
            for o in (j.get('objects') or []):
                bb = o.get('source_bbox') or o.get('bbox')
                if not bb:
                    continue
                c = crop_from_bbox(img, bb)
                if c is None:
                    continue
                items.append({'crop': c, 'cls': str(o.get('class')),
                              'size': round(float(min(bb[2] - bb[0],
                                                      bb[3] - bb[1])), 1),
                              'src': os.path.basename(src)})
    # 去重（同一原图同一位置）
    seen = set()
    uniq = []
    for it in items:
        k = (it['src'], it['cls'], it['size'])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    items = uniq[:args.limit]
    print('待判别 %d 个裁剪' % len(items))

    preds = classify([it['crop'] for it in items])
    os.makedirs(os.path.dirname(args.sheet), exist_ok=True)
    rows = []
    recs = []
    for it, (lab, raw) in zip(items, preds):
        recs.append({'cls': it['cls'], 'size': it['size'],
                     'pred': lab, 'raw': raw, 'src': it['src']})
        c = upscale(it['crop'], 120, 4)
        tile = np.full((150, 150, 3), 30, np.uint8)
        tile[0:120, 15:135] = c
        cv2.putText(tile, '%s %spx' % (it['cls'][:12], it['size']),
                    (2, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
        color = (0, 220, 0) if lab != 'unknown' else (0, 120, 220)
        cv2.putText(tile, lab, (2, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        rows.append(tile)
    json.dump(recs, open(args.json_out, 'w'), ensure_ascii=False, indent=1)

    if rows:
        per = 6
        grid = []
        for i in range(0, len(rows), per):
            chunk = rows[i:i + per]
            while len(chunk) < per:
                chunk.append(np.full((150, 150, 3), 30, np.uint8))
            grid.append(np.hstack(chunk))
        sheet = np.vstack(grid)
        cv2.imwrite(args.sheet, sheet)
        print('对照表:', args.sheet, sheet.shape)
    from collections import Counter
    print('预测分布:', dict(Counter(r['pred'] for r in recs)))


if __name__ == '__main__':
    main()
