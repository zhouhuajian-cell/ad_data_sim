#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_tsr_benchmark.py — TSR 仿真效果基准测试（今天验收用）

干什么：
  拿真实实拍图 + 对应国家的素材库，批量做标牌替换，输出：
    - 每张图的原图/合成图对比
    - 统计：成功/跳过/失败、各类跳过原因
    - 抽样拼图，方便人工看效果
  同时跑一遍「用对国家的素材」与「用错国家素材」的对比，量化国家匹配的价值。

为什么这样测：
  TSR 仿真的最终验收标准是「贴上去看不出 + 能当训练数据」。
  肉眼看样张 + 统计成功率，比单看某一项指标更接近实际使用。

用法：
  python3 run_tsr_benchmark.py --images <实拍目录> --library <素材库根> \
         --country BR --out <输出> [--limit 20]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "modules"))

import cv2
import numpy as np

from replace_sign import detect_signs, load_sign, sign_mask
from sign_realism import paste_realistic, ref_stats
from batch_replace import detect_shape, shape_matches, fit_bbox, list_signs, pick_neutral_asset


def pick_images(root, limit):
    if os.path.isfile(root):
        return [root]
    out = []
    for f in sorted(os.listdir(root)):
        p = os.path.join(root, f)
        if os.path.isfile(p) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            out.append(p)
    return out[:limit] if limit else out


def run_one(img_path, sign_table, out_dir, name, dry=False):
    """处理一张图，返回结果字典。dry=True 只做检测与匹配判断，不写文件。"""
    t0 = time.time()
    im = cv2.imread(img_path)
    if im is None:
        return {"ok": False, "reason": "读图失败"}
    dets = detect_signs(img_path, 0.30)
    if not dets:
        return {"ok": False, "reason": "无检测", "detections": 0}

    usable = [d for d in dets if d["w"] >= 24]
    if not usable:
        return {"ok": False, "reason": "标牌过小", "detections": len(dets)}

    d = usable[0]
    tgt_shape = detect_shape(im, d["bbox"])
    cls = d["class"]

    # 按大类挑「中性代表素材」——不能随便取第一个，否则会把限高牌贴到禁止右转的位置
    cand_name, path = pick_neutral_asset(cls, sign_table)
    cand = None
    if path:
        s = load_sign(path)
        m = sign_mask(s)
        if m.any():
            try:
                from build_library_mtsd import shape_for
                shp = shape_for(cand_name)
            except Exception:
                shp = "circle"
            if shape_matches(shp, tgt_shape):
                cand = (s, m, shp)
            else:
                return {"ok": False,
                        "reason": "形状不符(%s图案 vs %s场景)" % (shp, tgt_shape),
                        "detected": cls, "scene_shape": tgt_shape,
                        "tried": cand_name}

    if cand is None:
        return {"ok": False, "reason": "无匹配素材",
                "detected": cls, "scene_shape": tgt_shape}

    s, m, shp = cand
    bbox = fit_bbox(d["bbox"], m.shape, keep_ratio=True)
    composed, soft = paste_realistic(im, s[:, :, :3], m, bbox, strength=1.0)
    if soft is None:
        return {"ok": False, "reason": "bbox 过小"}

    res = {"ok": True, "detected": cls, "used": cand_name,
           "scene_shape": tgt_shape, "sign_shape": shp,
           "conf": d["conf"], "size_px": d["w"],
           "bbox": [round(v, 1) for v in bbox],
           "ms": round((time.time() - t0) * 1000)}

    if not dry:
        cv2.imwrite(os.path.join(out_dir, name + "_composed.jpg"), composed,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        # 保存对比裁剪
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = int((x2 - x1) * 0.8)
        H, W = im.shape[:2]
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
        a = im[cy1:cy2, cx1:cx2]
        b = composed[cy1:cy2, cx1:cx2]
        side = 260
        a = cv2.resize(a, (side, side))
        b = cv2.resize(b, (side, side))
        pair = np.hstack([a, np.full((side, 4, 3), 255, np.uint8), b])
        cv2.putText(pair, "ORIG", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(pair, "SIM", (side + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, name + "_pair.jpg"), pair)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--country", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    table = list_signs(a.library, a.country)
    if not table:
        print("素材表为空：%s（country=%s）" % (a.library, a.country))
        return 2
    print("素材库: %s  国家=%s  类别数=%d"
          % (a.library, a.country or "(全部)", len(table)))

    imgs = pick_images(a.images, a.limit)
    print("测试图片: %d 张" % len(imgs))

    results = []
    for i, p in enumerate(imgs):
        name = os.path.splitext(os.path.basename(p))[0]
        r = run_one(p, table, a.out, name)
        r["image"] = os.path.basename(p)
        results.append(r)
        flag = "OK " if r.get("ok") else "SKIP"
        detail = (("%s→%s 形状%s/%s %.0fpx"
                   % (r.get("detected"), r.get("used"), r.get("scene_shape"),
                      r.get("sign_shape"), r.get("size_px", 0)))
                  if r.get("ok") else r.get("reason", ""))
        print("  [%2d/%2d] %s %-34s %s" % (i + 1, len(imgs), flag, name[-34:], detail))

    ok = sum(1 for r in results if r.get("ok"))
    reasons = {}
    for r in results:
        if not r.get("ok"):
            reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1

    summary = {"library": a.library, "country": a.country,
               "total": len(results), "ok": ok, "skip": len(results) - ok,
               "skip_reasons": reasons,
               "avg_ms": round(np.mean([r["ms"] for r in results if r.get("ms")])) if ok else 0,
               "results": results}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    print("-" * 60)
    print("成功 %d / 跳过 %d （成功率 %.0f%%），平均 %.0f ms/张"
          % (ok, len(results) - ok, 100.0 * ok / max(1, len(results)), summary["avg_ms"]))
    if reasons:
        print("跳过原因：")
        for k, v in sorted(reasons.items(), key=lambda t: -t[1]):
            print("  %-30s %d" % (k, v))

    # 成功案例拼图
    pairs = sorted(f for f in os.listdir(a.out) if f.endswith("_pair.jpg"))
    if pairs:
        tiles = [cv2.imread(os.path.join(a.out, f)) for f in pairs[:12]]
        tiles = [t for t in tiles if t is not None]
        if tiles:
            side = tiles[0].shape[0]
            rows = []
            for i in range(0, len(tiles), 3):
                row = tiles[i:i + 3]
                while len(row) < 3:
                    row.append(np.full_like(tiles[0], 30))
                rows.append(np.hstack(row))
            cv2.imwrite(os.path.join(a.out, "grid.jpg"), np.vstack(rows))
            print("样张拼图: %s" % os.path.join(a.out, "grid.jpg"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
