#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SD 精修效果对比：同一批图跑两次（不精修 vs 精修），生成并排对比。"""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

IMAGES = "/mnt/Data_Platform/zhj_datamining/test/projects/Oversea_南美/images"
PY = "/opt/ad_mining/venv/bin/python"
WALK = "/opt/ue5_synth/tools/batch_walk.py"
BASE_ARGS = [
    "--images", IMAGES, "--country", "BR",
    "--limit", "12", "--min-sign", "30", "--conf", "0.28",
    "--vary-class", "--grow", "1.12", "--realism", "v3",
    "--qc", "--tune", "--gate4", "--shuffle", "--max-scan", "40000",
]


def run(out_dir, extra):
    subprocess.run(["rm", "-rf", out_dir])
    cmd = [PY, "-u", WALK] + BASE_ARGS + ["--out", out_dir] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    tail = (r.stdout or "").strip().split("\n")[-6:]
    return "\n".join(tail)


def main():
    out_a = "/data/render/cmp_nosd"
    out_b = "/data/render/cmp_sd"
    print("[1/2] 不精修…")
    print(run(out_a, []))
    print()
    print("[2/2] 加 SD 精修…")
    print(run(out_b, ["--sd"]))

    # 配对：同一源文件
    def load(out):
        m = {}
        for ap in glob.glob(os.path.join(out, "annotations/*.json")):
            try:
                with open(ap, encoding="utf-8") as f:
                    d = json.load(f)
                m[d.get("source_image", "")] = (out, d)
            except Exception:
                pass
        return m

    ma, mb = load(out_a), load(out_b)
    common = sorted(set(ma) & set(mb))[:8]
    print("\n共同样本: %d" % len(common))

    tiles = []
    for k in common:
        try:
            d = ma[k][1]
            o = d["objects"][0]
            src = d["source_path"]
            ca_path = os.path.join(out_a, d["image"])
            cb_path = os.path.join(out_b, d["image"])
            src_img = cv2.imread(src)
            a_img = cv2.imread(ca_path)
            b_img = cv2.imread(cb_path)
            if src_img is None or a_img is None or b_img is None:
                continue
            bb = o["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bb]
            pad = max(10, int((x2 - x1) * 0.55))
            H, W = src_img.shape[:2]
            a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
            a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
            S = 200
            def rs(img, tag, col):
                c = cv2.resize(img[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)
                cv2.putText(c, tag, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
                return c
            row = np.hstack([
                rs(src_img, "ORIG", (0, 255, 0)), np.full((S, 2, 3), 255, np.uint8),
                rs(a_img, "NO-SD", (0, 180, 255)), np.full((S, 2, 3), 255, np.uint8),
                rs(b_img, "SD", (0, 255, 255)), np.full((S, 10, 3), 40, np.uint8),
            ])
            tiles.append(row)
        except Exception:
            continue

    if tiles:
        mw = max(t.shape[1] for t in tiles)
        tiles = [np.hstack([t, np.full((200, mw - t.shape[1], 3), 40, np.uint8)])
                 for t in tiles]
        out = np.vstack(tiles)
        cv2.imwrite("/tmp/sd_vs_nosd.jpg", out)
        print("对比图 /tmp/sd_vs_nosd.jpg %s" % (out.shape,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
