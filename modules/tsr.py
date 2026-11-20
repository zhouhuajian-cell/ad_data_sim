#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modules/tsr.py — TSR（交通标识）模块

职责：
  1) 定义 TSR 资产的两级类别表（大类对齐 YOLO，子类对应具体标牌）
  2) 在 UE 关卡里生成路标：标准模板（牌面 + 立柱）+ 上传的图案贴图
  3) 输出真值描述（类别、世界盒、stencil、磨损等），供 batch_capture 投影成 2D 框
  4) 对外提供「类别名规范化」与「按 YOLO 大类找素材」两个工具，供 2D 替换复用

两级类别的必要性：
  YOLO 模型（tsr_sign.pt）只分 7 个大类，但素材库需要细分到具体标牌
  （限速60 / 限速80 / 禁止左转 …）。所以：
    - 大类（category）：用于和检测结果匹配
    - 子类（subtype）：用于区分素材，也直接作为训练标注的类别名
  素材文件名约定：<大类>_<子类>.png，例如 speed_limit_60.png、prohibition_no_right.png

设计要点：
  - 不依赖外部 3D 模型：牌面/立柱都用 UE 基础 Cube 缩放，任何工程都能跑
  - 单位坑：UE 世界位置是厘米，set_actor_scale3d 是倍数（Cube 原始 100cm）
  - 语义靠 CustomDepthStencilValue，traffic_sign = 4
"""
import json
import math
import os

# ---------------------------------------------------------------------------
# 两级类别表
# ---------------------------------------------------------------------------
# 大类 → {中文名, 子类表, 形状, 尺寸}
# 子类表里 value 是该子类的中文说明，key 会拼进类别名（如 speed_limit + 60）
TSR_CATALOG = {
    "speed_limit": {
        "label": "限速",
        "shape": "circle",
        "size_cm": {"diameter": 60},
        "subtypes": {
            "40": "限速40", "50": "限速50", "60": "限速60",
            "70": "限速70", "80": "限速80", "100": "限速100", "120": "限速120",
        },
    },
    "prohibition": {
        "label": "禁止",
        "shape": "circle",
        "size_cm": {"diameter": 60},
        "subtypes": {
            "no_right": "禁止右转", "no_left": "禁止左转",
            "no_entry": "禁止驶入", "no_parking": "禁止停车",
            "no_stopping": "禁止长停", "no_u_turn": "禁止掉头",
            "no_horn": "禁止鸣笛",
        },
    },
    "warning": {
        "label": "警告",
        "shape": "triangle",
        "size_cm": {"side": 70},
        "subtypes": {
            "bump": "路面不平", "curve_left": "向左急弯",
            "curve_right": "向右急弯", "pedestrian": "注意行人",
            "children": "注意儿童", "crossroad": "十字交叉",
            "slippery": "易滑", "narrow": "窄路",
        },
    },
    "mandatory": {
        "label": "指令",
        "shape": "circle",
        "size_cm": {"diameter": 60},
        "subtypes": {
            "go_straight": "直行", "turn_left": "左转", "turn_right": "右转",
            "keep_left": "靠左行驶", "keep_right": "靠右行驶",
            "roundabout": "环岛行驶", "min_speed": "最低限速",
        },
    },
    "guide": {
        "label": "指路",
        "shape": "rect",
        "size_cm": {"w": 120, "h": 60},
        "subtypes": {
            "direction": "方向指路", "place": "地名", "distance": "距离",
            "highway": "高速入口", "exit": "出口",
        },
    },
    "signal": {
        "label": "信号",
        "shape": "circle",
        "size_cm": {"diameter": 30},
        "subtypes": {"red": "红灯", "yellow": "黄灯", "green": "绿灯"},
    },
    "crosswalk": {
        "label": "人行横道",
        "shape": "rect",
        "size_cm": {"w": 70, "h": 70},
        "subtypes": {"zebra": "斑马线", "sign": "人行横道标志"},
    },
}

# 兼容旧代码：大类名集合
TSR_CLASSES = {k: v["label"] for k, v in TSR_CATALOG.items()}

STENCIL = 4          # 语义通道里 TSR 的模板值（见 semantic_encode.SEMANTIC_CLASSES）
POLE_DIAMETER_CM = 6
PLATE_THICKNESS_CM = 5


# ---------------------------------------------------------------------------
# 类别工具
# ---------------------------------------------------------------------------
def parse_class_name(name):
    """把 'speed_limit_60' 之类的名字拆成 (大类, 子类)；无子类时子类为空串。"""
    name = str(name or "").strip()
    if not name:
        return "", ""
    for cat in TSR_CATALOG:
        if name == cat:
            return cat, ""
        if name.startswith(cat + "_"):
            return cat, name[len(cat) + 1:]
    return name, ""


def normalize_class_name(name):
    """规范成 '<大类>_<子类>' 形式；无法识别时原样返回。"""
    cat, sub = parse_class_name(name)
    if cat in TSR_CATALOG:
        return cat if not sub else "%s_%s" % (cat, sub)
    return name


def category_of(name):
    """取大类（用于和 YOLO 检测结果匹配）。"""
    cat, _ = parse_class_name(name)
    return cat if cat in TSR_CATALOG else ""


def subtype_of(name):
    _, sub = parse_class_name(name)
    return sub


def class_label(name):
    """人类可读的中文名，如 speed_limit_60 → 限速60。"""
    cat, sub = parse_class_name(name)
    spec = TSR_CATALOG.get(cat)
    if not spec:
        return name
    if not sub:
        return spec["label"]
    return spec["subtypes"].get(sub, "%s %s" % (spec["label"], sub))


def all_class_names():
    """展开所有可用类别名（大类 + 所有子类），供素材库校验与标注枚举。"""
    out = []
    for cat, spec in TSR_CATALOG.items():
        out.append(cat)
        for sub in spec["subtypes"]:
            out.append("%s_%s" % (cat, sub))
    return out


def material_for(category, assets):
    """从素材表里挑出某大类下的图案路径。

    assets: {规范化类别名: 路径}（由 scan_assets 生成）
    优先精确匹配该大类下第一个子类，再退回大类本身。
    """
    if category in assets:
        return assets[category]
    for name, path in assets.items():
        if category_of(name) == category:
            return path
    return None


def scan_assets(root):
    """扫描素材目录，返回 {规范化类别名: 文件路径}。

    文件名示例：speed_limit_60.png / prohibition_no_right.jpg / warning.png
    无法识别大类的文件直接忽略（不做猜测，避免标错类别）。
    """
    table = {}
    if not root or not os.path.isdir(root):
        return table
    for fn in sorted(os.listdir(root)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = os.path.splitext(fn)[0]
        name = normalize_class_name(stem)
        if category_of(name):
            table.setdefault(name, os.path.join(root, fn))
    return table


# ---------------------------------------------------------------------------
# 尺寸
# ---------------------------------------------------------------------------
def _spec(cls):
    cat, _ = parse_class_name(cls)
    return TSR_CATALOG.get(cat, TSR_CATALOG["speed_limit"])


def plate_size_cm(cls):
    """返回 (宽, 高) 厘米。圆形按直径、三角形按外接矩形。"""
    spec = _spec(cls)
    shape = spec["shape"]
    size = spec["size_cm"]
    if shape == "circle":
        d = size.get("diameter", 60)
        return d, d
    if shape == "triangle":
        s = size.get("side", 70)
        return s, int(round(s * math.sqrt(3) / 2))
    return size.get("w", 70), size.get("h", 70)


def plate_scale(cls, extra_scale=1.0):
    """把物理尺寸换算成 Cube 的缩放倍数（Cube 原始 100cm）。"""
    w_cm, h_cm = plate_size_cm(cls)
    return (PLATE_THICKNESS_CM / 100.0 * extra_scale,
            w_cm / 100.0 * extra_scale,
            h_cm / 100.0 * extra_scale)


def shape_of(cls):
    return _spec(cls)["shape"]


# ---------------------------------------------------------------------------
# UE 生成
# ---------------------------------------------------------------------------
def spawn_sign(unreal, eas, les, cls, material_path, pos_cm=(2000, 300, 220),
               yaw=180.0, scale=1.0, label_prefix="TSR"):
    """在关卡里生成一个路标（牌面 + 立柱），返回生成的 Actor 列表。"""
    cube = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    if cube is None:
        raise RuntimeError("找不到 /Engine/BasicShapes/Cube.Cube")

    w_cm, h_cm = plate_size_cm(cls)
    sx, sy, sz = plate_scale(cls, scale)
    px, py, pz = pos_cm

    plate = eas.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(px, py, pz), unreal.Rotator(0.0, yaw, 0.0))
    plate.set_actor_label("%s_%s_Plate" % (label_prefix, normalize_class_name(cls)))
    plate.set_actor_scale3d(unreal.Vector(sx, sy, sz))
    pm = plate.get_component_by_class(unreal.StaticMeshComponent)
    pm.set_static_mesh(cube)
    if material_path:
        mat = unreal.load_asset(material_path)
        if mat is not None:
            pm.set_material(0, mat)
    _set_stencil(pm)

    plate_half_cm = h_cm * 0.5 * scale
    pole_h_cm = max(20.0, pz - plate_half_cm)
    pole_z_cm = pole_h_cm * 0.5
    pole = eas.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(px, py, pole_z_cm), unreal.Rotator(0.0, yaw, 0.0))
    pole.set_actor_label("%s_%s_Pole" % (label_prefix, normalize_class_name(cls)))
    pole.set_actor_scale3d(unreal.Vector(POLE_DIAMETER_CM / 100.0,
                                         POLE_DIAMETER_CM / 100.0,
                                         pole_h_cm / 100.0))
    qm = pole.get_component_by_class(unreal.StaticMeshComponent)
    qm.set_static_mesh(cube)
    _set_stencil(qm)

    return [plate, pole]


def _set_stencil(comp):
    try:
        comp.set_editor_property("render_custom_depth", True)
        comp.set_editor_property("custom_depth_stencil_value", STENCIL)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 真值
# ---------------------------------------------------------------------------
def truth_entry(cls, pos_cm, yaw, scale=1.0, wear=0.0):
    """生成一条 TSR 真值（世界盒 / 类别 / 磨损），与 batch_capture 的 objects 对齐。"""
    name = normalize_class_name(cls)
    w_cm, h_cm = plate_size_cm(name)
    return {
        "class": name,
        "category": category_of(name),
        "label": class_label(name),
        "class_id": STENCIL,
        "world_center_cm": list(pos_cm),
        "world_extent_cm": [PLATE_THICKNESS_CM * 0.5 * scale,
                            w_cm * 0.5 * scale,
                            h_cm * 0.5 * scale],
        "yaw": yaw,
        "wear_level": float(wear),
        "shape": shape_of(name),
    }


if __name__ == "__main__":
    print("TSR 大类：%d 个" % len(TSR_CATALOG))
    total = 0
    for cat, spec in TSR_CATALOG.items():
        total += len(spec["subtypes"])
        print("  %-14s %-6s 子类 %d：%s" % (cat, spec["label"], len(spec["subtypes"]),
                                          "、".join(spec["subtypes"].keys())))
    print("子类合计：%d 个，可用类别名 %d 个" % (total, len(all_class_names())))
    print()
    print("类别名解析自检：")
    for n in ["speed_limit_60", "prohibition_no_right", "warning", "landline_unknown"]:
        print("  %-26s → 大类=%-12s 子类=%-12s 中文=%s"
              % (n, category_of(n) or "(无)", subtype_of(n) or "(无)", class_label(n)))
    print()
    print("尺寸自检：")
    for n in ["speed_limit_60", "warning_bump", "guide_exit"]:
        print("  %-20s %-8s %s cm  scale=%s"
              % (n, shape_of(n), plate_size_cm(n),
                 [round(v, 3) for v in plate_scale(n)]))
