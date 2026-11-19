#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdxl.py — 验证 SDXL + ControlNet 场景融合效果

对比三种方案：
  A. 纯贴图（现有 sign_realism2 v3）
  B. SD 1.5 精修（strength 0.24）
  C. SDXL + ControlNet Canny（strength 0.48, canny 0.72）

重点看：C 是否在保持数字正确的前提下，把光影真正融合进去。
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np

# 依赖模型目录（下载完才会存在）
SDXL_INPAINT = "/data/models/sdxl-inpaint"
CN_CANNY = "/data/models/controlnet-canny-sdxl"
SD15_INPAINT = "/data/models/sd15-inpaint"


def pick_samples(n=4):
    """从已有产出里挑几个样本，返回 (原图, 原图路径, 标注)。"""
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
            if sz < 30:
                continue
            out.append((src, d, o, sz))
        except Exception:
            continue
    out.sort(key=lambda x: -x[3])   # 优先大牌
    return out[:n]


def crop_view(img, bbox, pad_ratio=0.55, S=240):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = max(10, int((x2 - x1) * pad_ratio))
    H, W = img.shape[:2]
    a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
    a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
    return cv2.resize(img[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)


def main():
    samples = pick_samples(4)
    if not samples:
        print("没有可用样本")
        return 2
    print("选中 %d 个样本，尺寸: %s" % (len(samples), [int(s[3]) for s in samples]))

    # 先加载 SDXL（慢），失败就直接返回
    print("\n加载 SDXL + ControlNet…")
    t0 = time.time()
    try:
        from scene_fusion import SceneFusionTSRSimulator
        sim = SceneFusionTSRSimulator(
            inpaint_model=SDXL_INPAINT,
            controlnet_model=CN_CANNY,
        )
        print("加载成功 %.1fs" % (time.time() - t0))
    except Exception as e:
        print("SDXL 加载失败: %s: %s" % (type(e).__name__, str(e)[:300]))
        return 2

    # SD 1.5（可选对比）
    sd15 = None
    try:
        from sd_refine import refine as sd15_refine
        sd15 = sd15_refine
    except Exception:
        pass

    rows = []
    for src, d, o, sz in samples:
        im = cv2.imread(src)
        if im is None:
            continue
        comp_path = os.path.join("/data/render/tsr_100", d["image"])
        base = cv2.imread(comp_path) if os.path.isfile(comp_path) else im.copy()

        # 掩码用检测框
        from sd_refine import make_sign_mask
        mask = make_sign_mask(im.shape, o["bbox"])

        tiles = [crop_view(base, o["bbox"])]
        cv2.putText(tiles[0], "PASTE %dpx" % sz, (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1)

        # B. SD 1.5
        if sd15 is not None:
            try:
                t1 = time.time()
                out15, _ = sd15(base, mask, strength=0.24, sign_px=sz,
                                model_dir=SD15_INPAINT)
                tiles.append(crop_view(out15, o["bbox"]))
                cv2.putText(tiles[-1], "SD15 %.1fs" % (time.time() - t1),
                            (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
            except Exception as e:
                print("  SD1.5 失败: %s" % str(e)[:80])

        # C. SDXL + ControlNet
        try:
            t2 = time.time()
            out_xl = sim.run(im.copy(), mask,
                             cv2.cvtColor(base, cv2.COLOR_BGR2BGRA),
                             strength=0.48, canny_scale=0.72,
                             skip_lama=True)
            tiles.append(crop_view(out_xl, o["bbox"]))
            cv2.putText(tiles[-1], "SDXL %.1fs" % (time.time() - t2),
                        (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 200, 255), 1)
            print("  %s %dpx  SDXL %.1fs OK" % (o["class"][:20], sz, time.time() - t2))
        except Exception as e:
            print("  SDXL 失败: %s: %s" % (type(e).__name__, str(e)[:150]))

        while len(tiles) < 3:
            tiles.append(np.full((240, 240, 3), 40, np.uint8))
        row = np.hstack([t for pair in tiles for t in (pair, np.full((240, 6, 3), 40, np.uint8))])
        rows.append(row)

    if rows:
        mw = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], mw - r.shape[1], 3), 40, np.uint8)])
                for r in rows]
        out = np.vstack(rows)
        cv2.imwrite("/tmp/sdxl_cmp.jpg", out)
        print("\n对比图 /tmp/sdxl_cmp.jpg %s" % (out.shape,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
