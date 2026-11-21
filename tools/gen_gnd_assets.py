#!/usr/bin/env python3
"""生成 GND 标线纹理库（程序化，可控参数）。

类型覆盖：
  白/黄实线、白/黄虚线、双黄线、斑马线、导向箭头、
  停止线、路缘线、减速带、网格线
"""
import json
import os

import cv2
import numpy as np

OUT = "/data/gnd_assets"

SPECS = {
    # key: (宽, 高, 颜色BGR, 说明, 额外参数)
    "lane_solid_white":   dict(w=48,  h=1200, rgb=(238, 238, 238),
                               cn="白色实线", dash=False),
    "lane_solid_yellow":  dict(w=48,  h=1200, rgb=(60, 200, 235),
                               cn="黄色实线", dash=False),
    "lane_dashed_white":  dict(w=48,  h=1800, rgb=(238, 238, 238),
                               cn="白色虚线", dash=True, on=180, gap=300),
    "lane_dashed_yellow": dict(w=48,  h=1800, rgb=(60, 200, 235),
                               cn="黄色虚线", dash=True, on=180, gap=300),
    "double_yellow":      dict(w=120, h=1200, rgb=(60, 200, 235),
                               cn="双黄线", dash=False, gap_mid=30),
    "road_edge_white":    dict(w=64,  h=1200, rgb=(238, 238, 238),
                               cn="路缘白线", dash=False),
    "crosswalk_zebra":    dict(w=600, h=300, rgb=(235, 235, 235),
                               cn="斑马线", dash=True, on=250, gap=250,
                               horizontal=True),
    "stop_line":          dict(w=1200, h=120, rgb=(238, 238, 238),
                               cn="停止线", dash=False, horizontal=True),
    "arrow_straight":     dict(w=400, h=800, rgb=(238, 238, 238),
                               cn="直行箭头", special="arrow_straight"),
    "arrow_left":         dict(w=400, h=800, rgb=(238, 238, 238),
                               cn="左转箭头", special="arrow_left"),
    "arrow_right":        dict(w=400, h=800, rgb=(238, 238, 238),
                               cn="右转箭头", special="arrow_right"),
    "hatch_zone":         dict(w=600, h=1200, rgb=(238, 238, 238),
                               cn="导流线", special="hatch"),
    "speed_bump":         dict(w=1200, h=200, rgb=(235, 235, 60),
                               cn="减速带", special="bump"),
}


def make_solid(w, h, rgb, wear=0.0):
    img = np.full((h, w, 3), rgb, np.uint8)
    a = np.full((h, w), 255, np.uint8)
    return img, a


def make_dashed(w, h, rgb, on, gap, horizontal=False):
    img = np.zeros((h, w, 3), np.uint8)
    a = np.zeros((h, w), np.uint8)
    pos = 0
    while pos < (w if horizontal else h):
        end = min(pos + on, w if horizontal else h)
        if horizontal:
            img[:, pos:end] = rgb
            a[:, pos:end] = 255
        else:
            img[pos:end, :] = rgb
            a[pos:end, :] = 255
        pos = end + gap
    return img, a


def make_double(w, h, rgb, gap_mid):
    img = np.zeros((h, w, 3), np.uint8)
    a = np.zeros((h, w), np.uint8)
    each = (w - gap_mid) // 2
    img[:, :each] = rgb
    a[:, :each] = 255
    img[:, each + gap_mid:] = rgb
    a[:, each + gap_mid:] = 255
    return img, a


def make_arrow(w, h, rgb, direction):
    img = np.zeros((h, w, 3), np.uint8)
    a = np.zeros((h, w), np.uint8)
    cx = w // 2
    shaft_w = int(w * 0.18)
    head_w = int(w * 0.42)
    head_h = int(h * 0.34)
    body_top = int(h * 0.12)

    # 箭杆
    cv2.rectangle(img, (cx - shaft_w // 2, h // 3),
                  (cx + shaft_w // 2, h - 20), rgb, -1)
    cv2.rectangle(a, (cx - shaft_w // 2, h // 3),
                  (cx + shaft_w // 2, h - 20), 255, -1)

    # 箭头
    pts = np.array([[cx, body_top],
                    [cx + head_w // 2, h // 3 + head_h // 2],
                    [cx - head_w // 2, h // 3 + head_h // 2]], np.int32)
    cv2.fillPoly(img, [pts], rgb)
    cv2.fillPoly(a, [pts], 255)

    if direction in ("left", "right"):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE
                         if direction == "right" else cv2.ROTATE_90_COUNTERCLOCKWISE)
        a = cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE
                       if direction == "right" else cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img, a


def make_hatch(w, h, rgb):
    img = np.zeros((h, w, 3), np.uint8)
    a = np.zeros((h, w), np.uint8)
    step = 90
    lw = 16
    for x in range(-h, w + h, step):
        p1 = (x, 0)
        p2 = (x + h, h)
        cv2.line(img, p1, p2, rgb, lw)
        cv2.line(a, p1, p2, 255, lw)
    return img, a


def make_bump(w, h, rgb):
    img = np.zeros((h, w, 3), np.uint8)
    a = np.zeros((h, w), np.uint8)
    seg = 100
    for i, x in enumerate(range(0, w, seg)):
        col = rgb if i % 2 == 0 else (40, 40, 40)
        img[:, x:x + seg] = col
        a[:, x:x + seg] = 255
    return img, a


def apply_wear(img, alpha, wear, seed):
    if wear <= 0.01:
        return img, alpha
    h, w = alpha.shape
    rng = np.random.default_rng(seed)
    low = rng.random((max(h // 10, 2), max(w // 10, 2))).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    low = cv2.GaussianBlur(low, (0, 0), max(min(h, w) * 0.015, 1.2))
    thr = 0.08 + 0.72 * wear
    k = np.ones((h, w), np.float32)
    holes = low < thr
    k[holes] = np.clip(low[holes] / max(thr, 1e-3), 0, 1)
    return img, (alpha.astype(np.float32) * k).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for key, sp in SPECS.items():
        w, h, rgb = sp["w"], sp["h"], sp["rgb"]
        special = sp.get("special")
        if special == "arrow_straight":
            img, a = make_arrow(w, h, rgb, "straight")
        elif special == "arrow_left":
            img, a = make_arrow(w, h, rgb, "left")
        elif special == "arrow_right":
            img, a = make_arrow(w, h, rgb, "right")
        elif special == "hatch":
            img, a = make_hatch(w, h, rgb)
        elif special == "bump":
            img, a = make_bump(w, h, rgb)
        elif sp.get("gap_mid"):
            img, a = make_double(w, h, rgb, sp["gap_mid"])
        elif sp.get("dash"):
            img, a = make_dashed(w, h, rgb, sp["on"], sp["gap"],
                                 sp.get("horizontal", False))
        else:
            img, a = make_solid(w, h, rgb)

        # 干净版
        d = os.path.join(OUT, key)
        os.makedirs(d, exist_ok=True)
        rgba = np.dstack([img, a])
        cv2.imwrite(os.path.join(d, "tex.png"), rgba)

        # 磨损版（多个档位）
        for wear in (0.25, 0.55):
            i2, a2 = apply_wear(img.copy(), a.copy(), wear, hash(key) & 0xFFFF)
            cv2.imwrite(os.path.join(d, "tex_wear%d.png" % int(wear * 100)),
                        np.dstack([i2, a2]))

        meta = {
            "key": key,
            "label": sp["cn"],
            "size": [w, h],
            "color_bgr": list(rgb),
            "dash": sp.get("dash", False),
            "horizontal": sp.get("horizontal", False),
            "source": "procedural",
            "license": "internal",
            "files": ["tex.png", "tex_wear25.png", "tex_wear55.png"],
        }
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        n += 1
        print("  %-22s %-10s %dx%d" % (key, sp["cn"], w, h))

    print("\n共 %d 种标线纹理 → %s" % (n, OUT))


if __name__ == "__main__":
    main()
