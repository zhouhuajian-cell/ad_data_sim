#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modules/gnd.py — GND（地面/车道线）模块

职责：
  1) 定义地面标线的类型（实线/虚线/双黄/停止线/箭头/路面文字）与规格
  2) 在 UE 关卡里铺标线：用细扁 Cube 贴在地面上（线宽 15cm、离地 1cm 防 z-fighting）
  3) 输出真值：车道线的世界盒（很扁）、附着路面、磨损等级

为什么车道线要单独一个模块：
  - 它的几何极扁（高度只有 1~2cm），投影后的 2D 框几乎是细长条，
    与 TSR/OBJ 的处理方式不同（难例阈值、磨损判定都只对它有意义）
  - 虚线要按「段长 + 间隔」参数化生成，而不是放单个 mesh

尺寸参考（国标）：
  线宽 15cm；虚线 400cm 实 + 600cm 空（城市道路）；双黄线间距 15cm
"""
import json

# 标线类型 → 规格
GND_TYPES = {
    "lane_solid_white":  {"width_cm": 15, "color": (255, 255, 255), "style": "solid"},
    "lane_dashed_white": {"width_cm": 15, "color": (255, 255, 255), "style": "dashed",
                          "dash_cm": 400, "gap_cm": 600},
    "lane_solid_yellow": {"width_cm": 15, "color": (240, 200, 60), "style": "solid"},
    "lane_double_yellow": {"width_cm": 15, "color": (240, 200, 60), "style": "double",
                           "gap_between_cm": 15},
    "stop_line":         {"width_cm": 40, "color": (255, 255, 255), "style": "solid"},
    "arrow_straight":    {"width_cm": 300, "color": (255, 255, 255), "style": "mark"},
}

STENCIL = 1          # 语义通道：地面标线归到 road（1），需要单独区分时改成 7
LINE_THICKNESS_CM = 2   # 离地厚度：太薄会 z-fighting，太厚投影后框会变高
GROUND_Z_CM = 1         # 贴地高度


def spec_of(gnd_type):
    return GND_TYPES.get(gnd_type, {"width_cm": 15, "color": (255, 255, 255), "style": "solid"})


def spawn_line(unreal, eas, gnd_type, start_cm=(0, 0), length_cm=2000.0, yaw=0.0,
               wear=0.0, label_prefix="GND"):
    """铺一条标线。start_cm 是 (x, y) 起点，沿 yaw 方向延伸 length_cm。

    虚线会被拆成多段（每段一个 Actor）；返回值是生成的 Actor 列表。
    """
    spec = spec_of(gnd_type)
    cube = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    actors = []

    x0, y0 = start_cm
    width_cm = spec["width_cm"]

    if spec["style"] == "dashed":
        dash = spec.get("dash_cm", 400)
        gap = spec.get("gap_cm", 600)
        pos = 0.0
        while pos < length_cm:
            seg = min(dash, length_cm - pos)
            actors.append(_place_segment(unreal, eas, cube, gnd_type,
                                         x0, y0, pos, seg, width_cm, yaw,
                                         label_prefix, wear))
            pos += dash + gap
    elif spec["style"] == "double":
        gap_bt = spec.get("gap_between_cm", 15)
        for sign in (-1, 1):
            off = sign * (width_cm + gap_bt) * 0.5
            actors.append(_place_segment(unreal, eas, cube, gnd_type,
                                         x0, y0, 0.0, length_cm, width_cm, yaw,
                                         label_prefix, wear, lateral_cm=off))
    else:
        actors.append(_place_segment(unreal, eas, cube, gnd_type,
                                     x0, y0, 0.0, length_cm, width_cm, yaw,
                                     label_prefix, wear))
    return actors


def _place_segment(unreal, eas, cube, gnd_type, x0, y0, start_off_cm, seg_len_cm,
                   width_cm, yaw, label_prefix, wear, lateral_cm=0.0):
    """沿 yaw 方向放一段标线，起点偏移 start_off_cm。"""
    import math
    rad = math.radians(yaw)
    # 前进方向
    fx, fy = math.cos(rad), math.sin(rad)
    # 侧向（左法线）
    nx, ny = -math.sin(rad), math.cos(rad)

    cx = x0 + fx * (start_off_cm + seg_len_cm * 0.5) + nx * lateral_cm
    cy = y0 + fy * (start_off_cm + seg_len_cm * 0.5) + ny * lateral_cm

    actor = eas.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(cx, cy, GROUND_Z_CM),
        unreal.Rotator(0.0, yaw, 0.0))
    actor.set_actor_label("%s_%s" % (label_prefix, gnd_type))
    # Cube 原始 100cm：X 方向拉长成线段，Y 方向是线宽，Z 是厚度
    actor.set_actor_scale3d(unreal.Vector(seg_len_cm / 100.0,
                                          width_cm / 100.0,
                                          LINE_THICKNESS_CM / 100.0))
    comp = actor.get_component_by_class(unreal.StaticMeshComponent)
    comp.set_static_mesh(cube)
    try:
        comp.set_editor_property("render_custom_depth", True)
        comp.set_editor_property("custom_depth_stencil_value", STENCIL)
    except Exception:
        pass
    return actor


def truth_entry(gnd_type, center_cm, length_cm, yaw=0.0, wear=0.0):
    """真值：车道线的世界盒（高度极扁）。"""
    spec = spec_of(gnd_type)
    return {
        "class": gnd_type,
        "class_id": STENCIL,
        "world_center_cm": list(center_cm),
        "world_extent_cm": [length_cm * 0.5, spec["width_cm"] * 0.5, LINE_THICKNESS_CM * 0.5],
        "yaw": yaw,
        "wear_level": float(wear),
    }


def default_layout():
    """一套常见的三车道标线布局（左右实线 + 中间虚线），供快速铺场景用。"""
    return [
        {"type": "lane_solid_white",  "start_cm": (0, -550), "length_cm": 4000, "yaw": 0.0},
        {"type": "lane_dashed_white", "start_cm": (0, -180), "length_cm": 4000, "yaw": 0.0},
        {"type": "lane_dashed_white", "start_cm": (0, 180),  "length_cm": 4000, "yaw": 0.0},
        {"type": "lane_solid_white",  "start_cm": (0, 550),  "length_cm": 4000, "yaw": 0.0},
    ]
