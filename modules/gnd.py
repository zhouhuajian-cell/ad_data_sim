#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modules/gnd.py — GND（地面/车道线）模块

职责：
  1) 定义地面标线类型（实线/虚线/双黄/停止线/箭头/斑马线）与国标规格
  2) 在 UE 关卡里铺标线：细扁 Cube 贴地，带颜色材质
  3) 输出真值：车道线的世界盒（高度极扁）、磨损等级

比 TSR/OBJ 更接近可用的原因：
  车道线是**平面图案**，不需要精细 3D 模型，基础几何体就能表达得很准。
  线上的磨损、断线、反光差异用参数控制，这是实拍难采集、UE 仿真有优势的部分。

尺寸参考（国标）：
  线宽 15cm；虚线 400cm 实 + 600cm 空（城市道路）；双黄线间距 15cm；
  斑马线宽 45cm、间隔 60cm。
"""
import math

# 标线类型 → 规格（width_cm 是线宽；dash/gap 只对虚线有效）
GND_TYPES = {
    "lane_solid_white":   {"width_cm": 15, "color": (0.92, 0.92, 0.92), "style": "solid"},
    "lane_dashed_white":  {"width_cm": 15, "color": (0.92, 0.92, 0.92), "style": "dashed",
                           "dash_cm": 400, "gap_cm": 600},
    "lane_solid_yellow":  {"width_cm": 15, "color": (0.85, 0.62, 0.10), "style": "solid"},
    "lane_double_yellow": {"width_cm": 15, "color": (0.85, 0.62, 0.10), "style": "double",
                           "gap_between_cm": 15},
    "stop_line":          {"width_cm": 40, "color": (0.92, 0.92, 0.92), "style": "solid"},
    "crosswalk_zebra":    {"width_cm": 45, "color": (0.92, 0.92, 0.92), "style": "zebra",
                           "gap_cm": 60, "strip_len_cm": 400},
    "road_edge_white":    {"width_cm": 20, "color": (0.92, 0.92, 0.92), "style": "solid"},
}

STENCIL = 1             # 语义通道：地面标线归到 road（1）
LINE_THICKNESS_CM = 2   # 离地厚度（太薄会 z-fighting，太厚投影的框会变高）
GROUND_Z_CM = 2         # 贴地高度（抬 2cm 避免与路面共面闪烁）

_MAT_CACHE = {}


def spec_of(gnd_type):
    return GND_TYPES.get(gnd_type, {"width_cm": 15, "color": (0.9, 0.9, 0.9), "style": "solid"})


def ensure_material(unreal, gnd_type):
    """为标线类型建一个纯色 Unlit 材质（白色/黄色），让线在渲染里看得见。

    为什么用 Unlit：路面标线是反光涂料，在图上呈现接近纯色；
    用 Unlit 能保证亮度稳定，也便于语义通道替换时识别。
    """
    if gnd_type in _MAT_CACHE:
        return _MAT_CACHE[gnd_type]

    spec = spec_of(gnd_type)
    r, g, b = spec["color"]
    name = "M_GND_%s" % gnd_type
    path = "/Game/Validation/GND/" + name
    try:
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            mat = unreal.load_asset(path)
            _MAT_CACHE[gnd_type] = mat
            return mat
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        lib = unreal.MaterialEditingLibrary
        mat = tools.create_asset(name, "/Game/Validation/GND", unreal.Material,
                                 unreal.MaterialFactoryNew())
        mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
        mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
        node = lib.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -300, 0)
        node.set_editor_property("constant", unreal.LinearColor(r, g, b, 1.0))
        lib.connect_material_property(node, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        lib.recompile_material(mat)
        unreal.EditorAssetLibrary.save_loaded_asset(mat)
        _MAT_CACHE[gnd_type] = mat
        return mat
    except Exception:
        _MAT_CACHE[gnd_type] = None
        return None


def spawn_line(unreal, eas, gnd_type, start_cm=(0, 0), length_cm=2000.0, yaw=0.0,
               wear=0.0, label_prefix="GND"):
    """铺一条标线。start_cm 是 (x, y) 起点，沿 yaw 方向延伸 length_cm。

    虚线拆成多段；双黄线并排两条；斑马线垂直于走向排列。
    wear 0~1：磨损程度，越大线越窄（模拟磨掉边缘）。
    返回生成的 Actor 列表。
    """
    spec = spec_of(gnd_type)
    cube = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    if cube is None:
        raise RuntimeError("找不到 Cube")
    mat = ensure_material(unreal, gnd_type)

    actors = []
    x0, y0 = start_cm
    width_cm = spec["width_cm"] * (1.0 - 0.45 * float(wear))    # 磨损让线变窄

    style = spec["style"]
    if style == "dashed":
        dash = spec.get("dash_cm", 400)
        gap = spec.get("gap_cm", 600)
        pos = 0.0
        while pos < length_cm:
            seg = min(dash, length_cm - pos)
            if seg > 1:
                actors.append(_place_segment(unreal, eas, cube, mat, gnd_type,
                                             x0, y0, pos, seg, width_cm, yaw,
                                             label_prefix))
            pos += dash + gap

    elif style == "double":
        gap_bt = spec.get("gap_between_cm", 15)
        for sign in (-1, 1):
            off = sign * (width_cm + gap_bt) * 0.5
            actors.append(_place_segment(unreal, eas, cube, mat, gnd_type,
                                         x0, y0, 0.0, length_cm, width_cm, yaw,
                                         label_prefix, lateral_cm=off))

    elif style == "zebra":
        # 斑马线：沿 yaw 方向依次放平行的短条（每条长度 strip_len_cm，垂直于走向）
        strip_len = spec.get("strip_len_cm", 400)
        gap = spec.get("gap_cm", 60)
        pos = 0.0
        while pos < length_cm:
            actors.append(_place_segment(unreal, eas, cube, mat, gnd_type,
                                         x0, y0, pos, width_cm, strip_len, yaw + 90.0,
                                         label_prefix))
            pos += width_cm + gap
            if len(actors) > 200:      # 防呆：别铺出上千个 actor
                break

    else:
        actors.append(_place_segment(unreal, eas, cube, mat, gnd_type,
                                     x0, y0, 0.0, length_cm, width_cm, yaw,
                                     label_prefix))
    return actors


def _place_segment(unreal, eas, cube, mat, gnd_type, x0, y0, start_off_cm,
                   seg_len_cm, width_cm, yaw, label_prefix, lateral_cm=0.0):
    """沿 yaw 方向放一段标线。"""
    rad = math.radians(yaw)
    fx, fy = math.cos(rad), math.sin(rad)
    nx, ny = -math.sin(rad), math.cos(rad)

    cx = x0 + fx * (start_off_cm + seg_len_cm * 0.5) + nx * lateral_cm
    cy = y0 + fy * (start_off_cm + seg_len_cm * 0.5) + ny * lateral_cm

    actor = eas.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(cx, cy, GROUND_Z_CM),
        unreal.Rotator(0.0, yaw, 0.0))
    actor.set_actor_label("%s_%s" % (label_prefix, gnd_type))
    # Cube 原始 100cm：X 拉成线段长，Y 是线宽，Z 是厚度
    actor.set_actor_scale3d(unreal.Vector(seg_len_cm / 100.0,
                                          width_cm / 100.0,
                                          LINE_THICKNESS_CM / 100.0))
    comp = actor.get_component_by_class(unreal.StaticMeshComponent)
    comp.set_static_mesh(cube)
    if mat is not None:
        comp.set_material(0, mat)
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
        "world_extent_cm": [length_cm * 0.5, spec["width_cm"] * 0.5,
                            LINE_THICKNESS_CM * 0.5],
        "yaw": yaw,
        "wear_level": float(wear),
    }


def default_layout():
    """三车道标线布局：左实线 + 两条虚线 + 右实线。"""
    return [
        {"type": "road_edge_white",   "start_cm": (0, -550), "length_cm": 4000, "yaw": 0},
        {"type": "lane_dashed_white", "start_cm": (0, -180), "length_cm": 4000, "yaw": 0,
         "wear": 0.3},
        {"type": "lane_dashed_white", "start_cm": (0, 180),  "length_cm": 4000, "yaw": 0,
         "wear": 0.3},
        {"type": "road_edge_white",   "start_cm": (0, 550),  "length_cm": 4000, "yaw": 0},
    ]
