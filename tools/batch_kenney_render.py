#!/usr/bin/env python3
"""批量把 Kenney GLB 渲染成 2D RGBA 贴片并入库。

输出结构（与现有管线兼容）:
    /data/assets_global/OBJ_PATCH/<category>/
        <name>_yaw000_pitch-8.png   (512x512 RGBA)
        ...
        <name>_meta.json            含真实尺寸、视角、来源

用法:
    python batch_kenney_render.py --out /data/assets_global/OBJ_PATCH
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, "/opt/ue5_synth/tools")
from render_glb_v2 import normalize_mesh, render_view  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SRC = "/data/assets_global/OBJ_3D/KENNEY"

# 类别 -> (L 长, W 宽, H 高) 米，按常见实车
CLASS_SPEC = {
    "car":        (4.50, 1.80, 1.50),
    "suv":        (4.70, 1.90, 1.75),
    "van":        (5.20, 1.95, 2.10),
    "truck":      (7.00, 2.30, 2.90),
    "bus":        (11.0, 2.50, 3.20),
    "cone":       (0.40, 0.40, 0.70),
    "barrier":    (2.00, 0.50, 0.90),
}

# 模型 -> 类别
MODEL_MAP = {
    "sedan":           "car",
    "sedan-sports":    "car",
    "hatchback-sports": "car",
    "taxi":            "car",
    "police":          "car",
    "suv":             "suv",
    "suv-luxury":      "suv",
    "delivery":        "van",
    "delivery-flat":   "van",
    "truck":           "truck",
    "tractor":         "truck",
    "tractor-shovel":  "truck",
    "garbage-truck":   "truck",
    "firetruck":       "truck",
    "ambulance":       "van",
    "cone":            "cone",
    "cone-flat":       "cone",
    "box":             "barrier",
}

YAWS = list(range(0, 360, 30))       # 12 个角度
PITCH = -8.0
RES = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/assets_global/OBJ_PATCH")
    ap.add_argument("--src", default=SRC)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index = []
    ok = fail = 0

    for model, cat in MODEL_MAP.items():
        glb = os.path.join(args.src, f"{model}.glb")
        if not os.path.isfile(glb):
            print(f"  跳过（无文件）: {model}")
            continue

        LWH = CLASS_SPEC[cat]
        cdir = os.path.join(args.out, cat)
        os.makedirs(cdir, exist_ok=True)

        try:
            mesh = normalize_mesh(glb, LWH)
        except Exception as e:
            print(f"  FAIL {model}: {e}")
            fail += 1
            continue

        views = []
        for yaw in YAWS:
            try:
                rgba = render_view(mesh, yaw, PITCH, res=RES)
            except Exception as e:
                print(f"    render FAIL yaw={yaw}: {e}")
                continue
            fn = f"{model}_yaw{yaw:03d}.png"
            cv2.imwrite(os.path.join(cdir, fn), rgba)
            views.append({"file": fn, "yaw": yaw, "pitch": PITCH})

        if not views:
            fail += 1
            continue

        meta = {
            "model": model,
            "category": cat,
            "source": "Kenney Car Kit (CC0)",
            "glb": os.path.relpath(glb, "/data"),
            "real_size_m": {"L": LWH[0], "W": LWH[1], "H": LWH[2]},
            "views": views,
            "res": RES,
        }
        with open(os.path.join(cdir, f"{model}_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        index.append({"model": model, "category": cat,
                      "n_views": len(views),
                      "dir": os.path.join(args.out, cat)})
        ok += 1
        print(f"  OK {model:20s} -> {cat:10s} {len(views)} 视角")

    with open(os.path.join(args.out, "_index.json"), "w",
              encoding="utf-8") as f:
        json.dump({"total": ok, "items": index}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n完成：{ok} 个模型，失败 {fail}")
    total_png = sum(1 for _, _, fs in os.walk(args.out)
                    for f in fs if f.endswith(".png"))
    print(f"贴片总数: {total_png}")


if __name__ == "__main__":
    main()
