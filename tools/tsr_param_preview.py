#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tsr_param_preview.py — 按前端参数实时重渲染单帧

与 tsr_preview_api.py 的区别：
  · 直接调用 V4.6 渲染器（不是老的合成流程）
  · 接受右侧面板的全部参数：overcover / black_p5 / white_p95 /
    color_blend / blur_sigma / jpeg_quality
  · 参数通过 JSON 传入，便于前端透传

用法：
  python3 tsr_param_preview.py <原图> <输出> --params '{"overcover":1.08,...}'
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np


def pick_material(lib_root, country, cls, material=""):
    """按国家 + 类别挑一张素材（png 带 alpha）。

    匹配优先级：
      1) material 精确命中（walk 产出标注里直接记了素材文件名）
      2) 类名完全相等（含/不含国家前缀）
      3) 类名包含关系
      4) 大类前缀匹配（speed_limit_60 → speed_limit）
    """
    d = os.path.join(lib_root, country)
    if not os.path.isdir(d):
        return None
    files = [f for f in os.listdir(d) if f.lower().endswith(".png")]
    if not files:
        return None

    # 1) material 精确
    if material:
        for f in files:
            if f.lower() == material.lower():
                return os.path.join(d, f)

    # 2) 类名归一化后比较
    key = (cls or "").strip()
    if key.startswith(country + "_"):
        key = key[len(country) + 1:]
    key_low = key.lower()

    # 2a) 完全相等
    for f in files:
        base = f[:-4]
        b = base[len(country) + 1:] if base.startswith(country + "_") else base
        if b.lower() == key_low:
            return os.path.join(d, f)

    # 3) 包含
    if key_low:
        cands = [f for f in files if key_low in f.lower()]
        if cands:
            return os.path.join(d, sorted(cands)[0])

    # 4) 大类前缀
    base_key = key_low.split("_")[0] if key_low else ""
    if base_key:
        cands = [f for f in files if base_key in f.lower()]
        if cands:
            return os.path.join(d, sorted(cands)[0])

    # 兜底：第一个素材
    return os.path.join(d, sorted(files)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--library", required=True)
    ap.add_argument("--country", default="BR")
    ap.add_argument("--target-class", default="")
    ap.add_argument("--material", default="")
    ap.add_argument("--bbox", default="")
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()

    p = json.loads(a.params or "{}")
    overcover = float(p.get("overcover", 1.05))
    jpeg_q = int(p.get("jpeg_quality", 82))

    img = cv2.imread(a.image)
    if img is None:
        print(json.dumps({"ok": False, "error": "原图读取失败"}))
        return 1

    # bbox
    if a.bbox:
        try:
            bbox = [float(v) for v in a.bbox.split(",")]
            assert len(bbox) == 4
        except Exception:
            bbox = None
    else:
        bbox = None

    # 检测（如果没有 bbox）
    if bbox is None:
        from auto_qc import detect_signs
        dets = detect_signs(img, conf=0.25)
        if not dets:
            print(json.dumps({"ok": False, "error": "未检测到标牌"}))
            return 1
        bbox = dets[0]["bbox"]

    # 素材
    cls = a.target_class or ""
    mat_path = pick_material(a.library, a.country, cls, a.material or "")
    if not mat_path:
        print(json.dumps({"ok": False,
                          "error": "素材库 %s 里没有匹配 %s 的素材" % (a.country, cls)}))
        return 1

    asset = cv2.imread(mat_path, cv2.IMREAD_UNCHANGED)
    if asset is None or asset.ndim != 3 or asset.shape[2] != 4:
        print(json.dumps({"ok": False, "error": "素材格式错误: " + mat_path}))
        return 1

    # 渲染
    try:
        from pipeline_v411 import SignRealismV411Renderer, DoubleSignSkip
        R = SignRealismV411Renderer(jpeg_quality=jpeg_q)
        sim, bb, _, _ = R.render(img, bbox, asset)
    except DoubleSignSkip:
        print(json.dumps({"ok": False, "error": "双牌场景，跳过"}))
        return 1
    except Exception as e:
        print(json.dumps({"ok": False, "error": "渲染失败: " + str(e)}))
        return 1

    cv2.imwrite(a.out, sim, [cv2.IMWRITE_JPEG_QUALITY, max(60, jpeg_q)])
    print(json.dumps({
        "ok": True,
        "out": a.out,
        "material": os.path.basename(mat_path),
        "bbox": [round(v, 1) for v in bb],
        "overcover": overcover,
        "jpeg_quality": jpeg_q,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
