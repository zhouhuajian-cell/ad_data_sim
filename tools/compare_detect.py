#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较原图与仿真图在 YOLO 上的检出能力，判断合成是否拉低了可识别性。"""
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np

from auto_qc import gate4_model_check, qc_check
from batch_replace import std_shape


def main():
    orig_pass = orig_fail = comp_pass = comp_fail = 0
    both_ok = orig_ok_comp_fail = 0
    samples = []

    od = "/data/render/tsr_final_1000"
    for ap in sorted(glob.glob(od + "/annotations/*.json"))[:120]:
        try:
            with open(ap, encoding="utf-8") as f:
                d = json.load(f)
            o = d["objects"][0]
            src = d.get("source_path", "")
            comp = os.path.join(od, d.get("image", ""))
            if not (os.path.isfile(src) and os.path.isfile(comp)):
                continue
            im = cv2.imread(src)
            ci = cv2.imread(comp)
            if im is None or ci is None:
                continue
            sh = std_shape(im, o["bbox"])
            if sh == "unknown":
                sh = "circle"
            ok1, _ = qc_check(im, ci, o["bbox"], sh)
            if not ok1:
                continue

            ok_o, _ = gate4_model_check(im, o["bbox"], o["class"])
            ok_c, _ = gate4_model_check(ci, o["bbox"], o["class"])
            orig_pass += 1 if ok_o else 0
            orig_fail += 0 if ok_o else 1
            comp_pass += 1 if ok_c else 0
            comp_fail += 0 if ok_c else 1
            if ok_o and ok_c:
                both_ok += 1
            if ok_o and not ok_c:
                orig_ok_comp_fail += 1
                if len(samples) < 8:
                    samples.append((im, ci, o["bbox"], o["class"]))
        except Exception:
            continue

    total = max(orig_pass + orig_fail, 1)
    print("前三道通过的样本 %d 个：" % total)
    print("  原图模型检出:   通过 %d  失败 %d  (%.1f%% 通过)"
          % (orig_pass, orig_fail, 100.0 * orig_pass / total))
    print("  仿真图模型检出: 通过 %d  失败 %d  (%.1f%% 通过)"
          % (comp_pass, comp_fail, 100.0 * comp_pass / total))
    print()
    print("  两者都能认:           %d" % both_ok)
    print("  原图能认+仿真认不出:  %d   <-- 合成拉低了可识别性" % orig_ok_comp_fail)

    if samples:
        tiles = []
        for im, ci, bb, cls in samples:
            x1, y1, x2, y2 = [int(v) for v in bb]
            pad = max(10, int((x2 - x1) * 0.5))
            H, W = im.shape[:2]
            a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
            a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
            S = 180
            ca = cv2.resize(im[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)
            cb = cv2.resize(ci[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)
            cv2.putText(ca, "ORIG ok", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 0), 1)
            cv2.putText(cb, "SIM fail", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 255), 1)
            tiles.append(np.hstack([ca, np.full((S, 3, 3), 255, np.uint8),
                                    cb, np.full((S, 10, 3), 40, np.uint8)]))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        out = np.vstack(rows)
        cv2.imwrite("/tmp/degrade.jpg", out)
        print()
        print("退化样本对比已保存 /tmp/degrade.jpg  %s" % (out.shape,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
