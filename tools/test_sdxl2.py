#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdxl2.py — SDXL 参数搜索：找到不毁掉标牌字形的配置

上一轮 strength=0.48 / canny=0.72 把标牌整个重绘成了风景，
说明在这个掩码尺寸下 SDXL 的结构约束不够强。

这轮对比三种策略：
  A. canny 拉满 1.0，strength 降到 0.35
  B. canny 拉满 1.0，strength 降到 0.28
  C. 只 mask 边缘环带（标牌本体不重绘），strength 0.45
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


def pick_samples(n=3):
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
            if sz < 50:
                continue
            out.append((src, d, o, sz))
        except Exception:
            continue
    out.sort(key=lambda x: -x[3])
    return out[:n]


def ring_mask(shape, bbox, inner_ratio=0.88, outer_ratio=1.25):
    """只取标牌边缘一圈做掩码，标牌本体不动。

    这样 SD 只负责把"贴上去的边缘"和场景融合，
    不能把牌面内容重画掉。
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    m = np.zeros((h, w), np.uint8)
    # 外框
    ow, oh = bw * outer_ratio / 2, bh * outer_ratio / 2
    cv2.rectangle(m, (int(cx - ow), int(cy - oh)), (int(cx + ow), int(cy + oh)), 255, -1)
    # 挖掉内框（标牌本体）
    iw, ih = bw * inner_ratio / 2, bh * inner_ratio / 2
    cv2.rectangle(m, (int(cx - iw), int(cy - ih)), (int(cx + iw), int(cy + ih)), 0, -1)
    return m


def crop_view(img, bbox, pad_ratio=0.55, S=240):
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

    print("加载 SDXL…")
    from scene_fusion import SceneFusionTSRSimulator
    from sd_refine import make_sign_mask
    sim = SceneFusionTSRSimulator(
        inpaint_model=SDXL_INPAINT,
        controlnet_model=CN_CANNY,
    )
    print("加载完成")

    variants = [
        ("A s0.35 c1.0", 0.35, 1.0, "full"),
        ("B s0.28 c1.0", 0.28, 1.0, "full"),
        ("C s0.45 ring", 0.45, 0.9, "ring"),
    ]

    rows = []
    for src, d, o, sz in samples:
        im = cv2.imread(src)
        if im is None:
            continue
        comp_path = os.path.join("/data/render/tsr_100", d["image"])
        base = cv2.imread(comp_path) if os.path.isfile(comp_path) else im.copy()

        tiles = [crop_view(base, o["bbox"])]
        cv2.putText(tiles[0], "PASTE %dpx" % sz, (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1)

        for name, st, cs, mtype in variants:
            try:
                if mtype == "ring":
                    mask = ring_mask(im.shape, o["bbox"])
                else:
                    mask = make_sign_mask(im.shape, o["bbox"])
                t0 = time.time()
                out = sim.run(im.copy(), mask,
                              cv2.cvtColor(base, cv2.COLOR_BGR2BGRA),
                              strength=st, canny_scale=cs,
                              skip_lama=(mtype == "ring"))
                dt = time.time() - t0
                tiles.append(crop_view(out, o["bbox"]))
                cv2.putText(tiles[-1], "%s %.1fs" % (name, dt), (3, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
                print("  %s %dpx %s %.1fs OK" % (o["class"][:18], sz, name, dt))
            except Exception as e:
                print("  %s 失败: %s" % (name, str(e)[:100]))

        while len(tiles) < 4:
            tiles.append(np.full((240, 240, 3), 40, np.uint8))
        row = np.hstack([t for p in tiles for t in (p, np.full((240, 5, 3), 40, np.uint8))])
        rows.append(row)

    if rows:
        mw = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], mw - r.shape[1], 3), 40, np.uint8)])
                for r in rows]
        out = np.vstack(rows)
        cv2.imwrite("/tmp/sdxl_tune.jpg", out)
        print("\n对比图 /tmp/sdxl_tune.jpg %s" % (out.shape,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
