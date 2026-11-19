#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdxl3.py — 用正确的调用方式测 SDXL

上一轮的 bug：把「已经贴好的图」当成 asset 传进 scene_fusion，
导致 step2 又贴了一次，SD 拿到的是错乱的输入，于是把标牌重绘成风景。

正确用法：
  sim.run(原图, 标牌掩码, 素材RGBA)
  → step1 LaMa 擦原牌
  → step2 把素材贴到擦干净的位置
  → step3 SD 精修
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np

SDXL_INPAINT = "/data/models/sdxl-inpaint"
CN_CANNY = "/data/models/controlnet-canny-sdxl"


def pick_samples(n=3, min_px=50):
    anns = sorted(glob.glob("/data/render/tsr_100/annotations/*.json"))
    out = []
    for a in anns:
        try:
            with open(a, encoding="utf-8") as f:
                d = json.load(f)
            o = d["objects"][0]
            src = d["source_path"]
            if not os.path.isfile(src):
                continue
            bb = o["bbox"]
            sz = min(bb[2] - bb[0], bb[3] - bb[1])
            if sz < min_px:
                continue
            out.append((src, d, o, sz))
        except Exception:
            continue
    out.sort(key=lambda x: -x[3])
    return out[:n]


def crop_view(img, bbox, pad_ratio=0.55, S=250):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = max(10, int((x2 - x1) * pad_ratio))
    H, W = img.shape[:2]
    a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
    a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
    return cv2.resize(img[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)


def main():
    samples = pick_samples(3)
    if not samples:
        print("没有样本")
        return 2
    print("样本尺寸:", [int(s[3]) for s in samples])

    from replace_sign import load_sign, sign_mask
    from scene_fusion import SceneFusionTSRSimulator

    print("加载 SDXL…")
    sim = SceneFusionTSRSimulator(inpaint_model=SDXL_INPAINT,
                                  controlnet_model=CN_CANNY)
    print("加载完成")

    rows = []
    for src, d, o, sz in samples:
        orig = cv2.imread(src)
        if orig is None:
            continue
        # 用产出时记录的素材路径
        mat = o.get("material", "")
        mat_path = os.path.join("/data/tsr_library_v2", "BR", mat)
        if not os.path.isfile(mat_path):
            print("  缺素材 %s" % mat)
            continue
        asset = load_sign(mat_path)
        if asset.shape[2] == 3:
            m = sign_mask(asset)
            asset = cv2.cvtColor(asset, cv2.COLOR_BGR2BGRA)
            asset[:, :, 3] = m

        bb = o["bbox"]
        mask = np.zeros(orig.shape[:2], np.uint8)
        x1, y1, x2, y2 = [int(v) for v in bb]
        # 掩码比检测框略大，盖住原牌的深色描边
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        hw, hh = int(w * 0.60), int(h * 0.60)
        mask[max(0, cy - hh):min(orig.shape[0], cy + hh),
             max(0, cx - hw):min(orig.shape[1], cx + hw)] = 255

        # 参照：现有纯贴图结果
        comp_path = os.path.join("/data/render/tsr_100", d["image"])
        base = cv2.imread(comp_path) if os.path.isfile(comp_path) else orig.copy()

        tiles = [crop_view(orig, bb), crop_view(base, bb)]
        cv2.putText(tiles[0], "ORIG %dpx" % sz, (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1)
        cv2.putText(tiles[1], "PASTE", (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 200, 255), 1)

        for name, st, cs in (("s0.35", 0.35, 0.9), ("s0.28", 0.28, 1.0)):
            try:
                t0 = time.time()
                out = sim.run(orig.copy(), mask, asset, strength=st, canny_scale=cs)
                dt = time.time() - t0
                tiles.append(crop_view(out, bb))
                cv2.putText(tiles[-1], "%s %.0fs" % (name, dt), (3, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
                print("  %-22s %3dpx %s %.1fs" % (o["class"][:22], sz, name, dt))
            except Exception as e:
                print("  %s 失败: %s" % (name, str(e)[:100]))

        while len(tiles) < 4:
            tiles.append(np.full((250, 250, 3), 40, np.uint8))
        row = np.hstack([t for p in tiles for t in (p, np.full((250, 5, 3), 40, np.uint8))])
        rows.append(row)

    if rows:
        mw = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], mw - r.shape[1], 3), 40, np.uint8)])
                for r in rows]
        out = np.vstack(rows)
        cv2.imwrite("/tmp/sdxl_v3.jpg", out)
        print("\n对比图 /tmp/sdxl_v3.jpg %s" % (out.shape,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
