#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modules/obj.py — OBJ（交通目标）模块

职责：
  1) 定义交通目标类别（车辆 / 行人 / 锥桶 / 护栏 等）与典型尺寸
  2) 在 UE 关卡里放置目标（当前用基础几何体做占位，后续可换真实 3D 模型）
  3) 输出真值描述：世界盒、类别、stencil、距离/朝向

为什么先用占位几何体：
  服务器上没有真实 3D 资产（车/人模型需要 Fab 或美术产出），
  但真值链路（投影 / 遮挡 / 难例判定）可以先跑通并验证。
  换成真实模型时只需替换 mesh 路径，模块的其它逻辑不变。

尺寸参考（厘米，常见乘用车/行人/锥桶）：
  轿车 450x180x150，SUV 470x190x175，货车 900x250x320
  行人 45x30x170，锥桶 40x40x70
"""
import json

# 类别 → (宽X, 宽Y, 高Z) 厘米；Y 是横向、X 是纵向、Z 是高度
OBJ_CLASSES = {
    "car":        {"size_cm": (450, 180, 150), "stencil": 2},
    "suv":        {"size_cm": (470, 190, 175), "stencil": 2},
    "truck":      {"size_cm": (900, 250, 320), "stencil": 2},
    "pedestrian": {"size_cm": (45, 30, 170), "stencil": 3},
    "cone":       {"size_cm": (40, 40, 70), "stencil": 5},
    "guardrail":  {"size_cm": (400, 20, 80), "stencil": 6},
}

ASSET_HINTS = {
    # 有真实模型时在这里填路径；为空则用基础 Cube/Cylinder 占位
    "car": "",
    "suv": "",
    "truck": "",
    "pedestrian": "",
    "cone": "",
    "guardrail": "",
}


def size_cm(cls):
    return OBJ_CLASSES.get(cls, {"size_cm": (100, 100, 100), "stencil": 2})["size_cm"]


def stencil_of(cls):
    return OBJ_CLASSES.get(cls, {"stencil": 2})["stencil"]


def spawn_object(unreal, eas, cls, pos_cm=(1200, 0, 0), yaw=0.0, scale=1.0,
                 label_prefix="OBJ"):
    """在关卡里放一个目标（底部贴地），返回 Actor。

    pos_cm 是**底部中心**（x,y 是平面位置，z 一般是 0=地面）；
    内部会按高度把 Actor 抬到中心位置，这样摆放时不用手算半高。
    """
    asset = ASSET_HINTS.get(cls, "")
    if asset:
        mesh = unreal.load_asset(asset)
        shape = None
    else:
        # 占位：车辆用扁 Cube，行人/锥桶细高，护栏长条
        shape = "cube"
        mesh = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    if mesh is None:
        raise RuntimeError("找不到网格资源（类 %s）" % cls)

    w_cm, d_cm, h_cm = size_cm(cls)
    x, y, z_bottom = pos_cm
    center_z = z_bottom + h_cm * 0.5 * scale

    actor = eas.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(x, y, center_z),
        unreal.Rotator(0.0, yaw, 0.0))
    actor.set_actor_label("%s_%s" % (label_prefix, cls))
    actor.set_actor_scale3d(unreal.Vector(w_cm / 100.0 * scale,
                                          d_cm / 100.0 * scale,
                                          h_cm / 100.0 * scale))
    comp = actor.get_component_by_class(unreal.StaticMeshComponent)
    comp.set_static_mesh(mesh)
    try:
        comp.set_editor_property("render_custom_depth", True)
        comp.set_editor_property("custom_depth_stencil_value", stencil_of(cls))
    except Exception:
        pass
    return actor


def truth_entry(cls, pos_cm, yaw=0.0, scale=1.0):
    """真值描述：世界盒以「目标中心」给出（与 tsr.py 的约定一致）。"""
    w_cm, d_cm, h_cm = size_cm(cls)
    x, y, z_bottom = pos_cm
    return {
        "class": cls,
        "class_id": stencil_of(cls),
        "world_center_cm": [x, y, z_bottom + h_cm * 0.5 * scale],
        "world_extent_cm": [w_cm * 0.5 * scale, d_cm * 0.5 * scale, h_cm * 0.5 * scale],
        "yaw": yaw,
    }


def batch_layout(count=5, start_x_cm=1500, gap_cm=800, lane_offset_cm=350):
    """生成一排目标的位置（用于批量出图时快速铺满场景）。"""
    classes = ["car", "suv", "truck", "pedestrian", "cone"]
    out = []
    for i in range(count):
        cls = classes[i % len(classes)]
        out.append({
            "class": cls,
            "pos_cm": [start_x_cm + i * gap_cm,
                       (lane_offset_cm if i % 2 == 0 else -lane_offset_cm),
                       0],
            "yaw": 180.0 if i % 2 == 0 else 0.0,
            "scale": 1.0,
        })
    return out
