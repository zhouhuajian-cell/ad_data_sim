#!/usr/bin/env python3
"""生成示例素材（含多视角、日夜），作为你放素材的格式参考。

这些是程序画的参考图，你可以直接替换成真实素材。
"""
import json
import os

import cv2
import numpy as np

OBJ = "/data/obj_assets"
GND = "/data/gnd_assets"


def _mk(p):
    os.makedirs(p, exist_ok=True)
    return p


def draw_cone(view="front", light="day", h=400):
    """锥桶。front=正三角，side=侧三角，rear=正三角（带反光条）。"""
    w = int(h * 0.40)
    im = np.zeros((h, w, 4), np.uint8)

    # 颜色
    if light == "night":
        body = (30, 110, 220, 255)      # 偏暗
        stripe = (230, 230, 230, 255)
    else:
        body = (40, 145, 245, 255)      # 橙
        stripe = (245, 245, 245, 255)
    base = (35, 35, 35, 255)

    # 梯形锥体
    top_w = int(w * 0.18)
    bot_w = int(w * 0.82)
    top_y = int(h * 0.06)
    bot_y = int(h * 0.90)
    pts = np.array([
        [w // 2 - top_w // 2, top_y],
        [w // 2 + top_w // 2, top_y],
        [w // 2 + bot_w // 2, bot_y],
        [w // 2 - bot_w // 2, bot_y],
    ], np.int32)
    cv2.fillPoly(im, [pts], body)

    # 反光条（2 条）
    for frac in (0.42, 0.58):
        y = int(top_y + (bot_y - top_y) * frac)
        hh = max(int(h * 0.055), 4)
        # 按梯形宽度插值
        t = (frac - 0.0) / 1.0
        cur_w = int(top_w + (bot_w - top_w) * t)
        x0 = w // 2 - cur_w // 2
        cv2.rectangle(im, (x0, y), (x0 + cur_w, y + hh), stripe, -1)

    # 底座
    cv2.rectangle(im, (int(w * 0.08), bot_y),
                  (int(w * 0.92), h - 2), base, -1)

    # 视角差异：side 收窄
    if view == "side_left" or view == "side_right":
        nw = max(int(w * 0.72), 8)
        sy = int(nw / w * 100) / 100.0
        im2 = cv2.resize(im, (nw, h), interpolation=cv2.INTER_AREA)
        out = np.zeros((h, w, 4), np.uint8)
        x0 = (w - nw) // 2
        out[:, x0:x0 + nw] = im2
        im = out

    return im


def draw_car(view="front", light="day", h=260):
    """轿车：front/rear=车头尾立面，side=侧视。"""
    if view in ("side_left", "side_right"):
        w = int(h * 2.6)
    else:
        w = int(h * 1.2)
    im = np.zeros((h, w, 4), np.uint8)

    if light == "night":
        body = (70, 70, 90, 255)
        glass = (35, 35, 45, 255)
    else:
        body = (200, 200, 205, 255)
        glass = (80, 90, 100, 255)
    wheel = (28, 28, 28, 255)

    if view in ("side_left", "side_right"):
        # 车身
        cv2.rectangle(im, (int(w * 0.02), int(h * 0.45)),
                      (int(w * 0.98), int(h * 0.80)), body, -1)
        # 车顶
        pts = np.array([
            [int(w * 0.20), int(h * 0.45)],
            [int(w * 0.30), int(h * 0.18)],
            [int(w * 0.70), int(h * 0.18)],
            [int(w * 0.80), int(h * 0.45)],
        ], np.int32)
        cv2.fillPoly(im, [pts], body)
        # 车窗
        cv2.rectangle(im, (int(w * 0.32), int(h * 0.22)),
                      (int(w * 0.68), int(h * 0.43)), glass, -1)
        # 轮子
        for fx in (0.18, 0.82):
            cv2.circle(im, (int(w * fx), int(h * 0.80)),
                       int(h * 0.16), wheel, -1)
    else:
        # 车尾/车头立面
        cv2.rectangle(im, (int(w * 0.05), int(h * 0.42)),
                      (int(w * 0.95), int(h * 0.82)), body, -1)
        # 车窗
        cv2.rectangle(im, (int(w * 0.16), int(h * 0.16)),
                      (int(w * 0.84), int(h * 0.44)), glass, -1)
        # 车顶
        cv2.rectangle(im, (int(w * 0.16), int(h * 0.10)),
                      (int(w * 0.84), int(h * 0.18)), body, -1)
        # 轮子
        for fx in (0.16, 0.84):
            cv2.rectangle(im, (int(w * fx) - int(w * 0.07), int(h * 0.78)),
                          (int(w * fx) + int(w * 0.07), h - 2), wheel, -1)

    return im


def draw_pedestrian(view="front", light="day", h=340):
    w = int(h * 0.38)
    im = np.zeros((h, w, 4), np.uint8)
    if light == "night":
        cloth = (60, 60, 90, 255)
        skin = (110, 120, 150, 255)
    else:
        cloth = (80, 90, 170, 255)
        skin = (140, 170, 210, 255)
    # 头
    cv2.circle(im, (w // 2, int(h * 0.10)), int(h * 0.075), skin, -1)
    # 躯干
    cv2.rectangle(im, (int(w * 0.26), int(h * 0.19)),
                  (int(w * 0.74), int(h * 0.56)), cloth, -1)
    # 腿
    cv2.rectangle(im, (int(w * 0.30), int(h * 0.56)),
                  (int(w * 0.46), h - 2), (50, 50, 70, 255), -1)
    cv2.rectangle(im, (int(w * 0.54), int(h * 0.56)),
                  (int(w * 0.70), h - 2), (50, 50, 70, 255), -1)
    return im


SPECS = {
    "cone": (draw_cone, {"width": 0.28, "height": 0.70, "length": 0.28},
             ["front", "side_left"], ["day", "night"]),
    "car": (draw_car, {"width": 1.80, "height": 1.50, "length": 4.30},
            ["front", "rear", "side_left"], ["day", "night"]),
    "pedestrian": (draw_pedestrian,
                   {"width": 0.55, "height": 1.70, "length": 0.45},
                   ["front"], ["day", "night"]),
}


def gen_obj():
    for cls, (fn, size, views, lights) in SPECS.items():
        var = "default"
        d = _mk(os.path.join(OBJ, cls))
        n = 0
        for v in views:
            for L in lights:
                im = fn(v, L)
                name = v + ("" if L == "day" else "_" + L) + ".png"
                cv2.imwrite(os.path.join(d, name), im)
                n += 1
        meta = {
            "class": cls, "variant": var,
            "size_m": size,
            "anchor": "bottom_center",
            "views": views, "lighting": lights,
            "source": "reference-placeholder",
            "license": "internal",
            "_note": "\u8fd9\u662f\u7a0b\u5e8f\u753b\u7684\u53c2\u8003\u56fe\uff0c\u53ef\u76f4\u63a5\u66ff\u6362",
        }
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print("  %-12s %d 张 → %s" % (cls, n, d))


def gen_gnd():
    """GND 重复单元纹理。"""
    GND_SPECS = {
        "lane_solid_white": {"width_m": 0.15, "tile_length_m": 4.0,
                             "color": "white", "dash": False},
        "lane_dashed_white": {"width_m": 0.15, "tile_length_m": 15.0,
                              "color": "white", "dash": True,
                              "dash_m": 6.0, "gap_m": 9.0},
        "lane_solid_yellow": {"width_m": 0.15, "tile_length_m": 4.0,
                              "color": "yellow", "dash": False},
        "road_edge_white": {"width_m": 0.20, "tile_length_m": 4.0,
                            "color": "white", "dash": False},
        "crosswalk_zebra": {"width_m": 0.45, "tile_length_m": 4.0,
                            "color": "white", "dash": True,
                            "dash_m": 0.45, "gap_m": 0.6},
    }
    PPM = 200

    for cls, sp in GND_SPECS.items():
        d = _mk(os.path.join(GND, cls))
        h = int(sp["tile_length_m"] * PPM)
        w = int(sp["width_m"] * PPM)
        h = max(h, 8)
        w = max(w, 8)

        im = np.zeros((h, w, 4), np.uint8)
        if sp["color"] == "yellow":
            col = (60, 200, 240, 255)
        else:
            col = (240, 240, 240, 255)

        if sp["dash"]:
            dash = int(sp["dash_m"] * PPM)
            gap = int(sp["gap_m"] * PPM)
            y = 0
            while y < h:
                y1 = min(y + dash, h)
                im[y:y1, :] = col
                y += dash + gap
        else:
            im[:, :] = col

        # 磨损：边缘稍暗 + 随机噪点
        rng = np.random.default_rng(hash(cls) % 2**32)
        noise = rng.random((h, w))
        wear = noise < 0.06
        im[..., 3] = np.where(wear, np.uint8(60), np.uint8(255))

        cv2.imwrite(os.path.join(d, "tex.png"), im)

        meta = {"class": cls}
        meta.update(sp)
        meta["ppm"] = PPM
        meta["_note"] = "\u53ef\u76f4\u63a5\u66ff\u6362\u6210\u771f\u5b9e\u7eb9\u7406"
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print("  %-22s %dx%d → %s" % (cls, w, h, d))


if __name__ == "__main__":
    print("=== 生成 OBJ 示例素材 ===")
    gen_obj()
    print("\n=== 生成 GND 示例纹理 ===")
    gen_gnd()
    print("\n完成。把真实素材放进去即可替换。")
