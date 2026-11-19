import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")
sys.path.insert(0, "/opt/ue5_synth/modules")

import cv2
import numpy as np

from batch_replace import std_shape, list_signs, asset_shape, CLASS_SHAPE
from pipeline6 import run_pipeline6
from replace_sign import detect_signs, load_sign, sign_mask

D = "/mnt/Data_Platform/zhj_datamining/test/projects/Oversea_南美/images"
files = []
for root, dirs, fs in os.walk(D):
    for f in fs:
        if f.endswith(".jpg"):
            files.append(os.path.join(root, f))
    if len(files) > 300:
        break

table = list_signs("/data/tsr_library_v2", "BR")
tried = 0
for p in files[:120]:
    dets = detect_signs(p, 0.28)
    if not dets:
        continue
    im = cv2.imread(p)
    if im is None:
        continue
    d = dets[0]
    sh = std_shape(im, d["bbox"])
    if sh == "unknown":
        continue
    want = "circle" if sh == "circle" else ("triangle" if sh == "triangle" else "rect")
    cands = [(n, pp) for n, pp in table.items()
             if (n == d["class"] or n.startswith(d["class"] + "_"))
             and asset_shape(n) == want]
    if not cands:
        continue
    cls, path = cands[0]
    s = load_sign(path)
    m = sign_mask(s)
    s_rgba = np.dstack([s[:, :, :3], m]) if s.shape[2] == 3 else s

    out, info = run_pipeline6(im, d["bbox"], s_rgba, sh)
    tried += 1
    print("--- 样本 %d: %s  形状=%s  框=%s" % (tried, cls, sh, [int(v) for v in d["bbox"]]))
    print("    ok=%s reason=%s" % (info.get("ok"), info.get("reason", "")))
    for st in info.get("steps", []):
        extra = {k: v for k, v in st.items() if k not in ("step", "name")}
        print("    Step%d %-14s %s" % (st["step"], st["name"], extra))
    if tried >= 3:
        break
