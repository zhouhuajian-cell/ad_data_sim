#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modules/obj.py — OBJ（交通目标）模块

职责：
  1) 定义交通目标类别与典型尺寸
  2) 在 UE 关卡里用「基础几何体组合」拼出可辨认的目标
  3) 输出真值描述：世界盒、类别、stencil

关于模型精度的诚实说明：
  本机没有可用的 3D 资产（插件里的 Mannequin / MultiWheelVehicle 加载不到，
  HuggingFace 上也没有可直接用的车辆/行人模型——这类资源在 Sketchfab/Fab 等模型市场）。
  所以这里用引擎自带的 Cube/Sphere/Cylinder/Cone 组合出「能看出是车/人/锥桶」的形状：
    轿车 = 车身 + 车舱 + 4 轮胎
    行人 = 躯干 + 头 + 腿 + 手臂
    锥桶 = 圆锥 + 底座
  用途：跑通渲染链路、验证真值投影与遮挡、产出结构化数据。
  局限：低多边形风格，**不适合直接当训练图像**。
  换成真实模型时，只需在 ASSET_HINTS 里填上 StaticMesh 路径，其余逻辑不变。

尺寸参考（厘米，常见乘用车/行人/锥桶）：
  轿车 450x180x150，SUV 470x190x175，货车 900x250x320
  行人 45x30x170，锥桶 40x40x70
"""
import math

# 类别 → (宽X, 宽Y, 高Z) 厘米；X 纵向、Y 横向、Z 高度
OBJ_CLASSES = {
    "car":        {"size_cm": (450, 180, 150), "stencil": 2},
    "suv":        {"size_cm": (470, 190, 175), "stencil": 2},
    "truck":      {"size_cm": (900, 250, 320), "stencil": 2},
    "pedestrian": {"size_cm": (45, 30, 170), "stencil": 3},
    "cone":       {"size_cm": (40, 40, 70), "stencil": 5},
    "guardrail":  {"size_cm": (400, 20, 80), "stencil": 6},
}

# 有真实模型时填这里（StaticMesh 路径）；留空则用基础几何体组合
ASSET_HINTS = {
    "car": "", "suv": "", "truck": "",
    "pedestrian": "", "cone": "", "guardrail": "",
}

_SHAPES = {}


def _shape(unreal, name):
    """缓存基础形状资产，避免反复 load。"""
    if name not in _SHAPES:
        _SHAPES[name] = unreal.load_asset("/Engine/BasicShapes/%s.%s" % (name, name))
    return _SHAPES[name]


def size_cm(cls):
    return OBJ_CLASSES.get(cls, {"size_cm": (100, 100, 100), "stencil": 2})["size_cm"]


def stencil_of(cls):
    return OBJ_CLASSES.get(cls, {"stencil": 2})["stencil"]


def _part(unreal, eas, mesh, label, local_cm, scale_xyz, yaw, origin, stencil):
    """放一个零件：local_cm 是相对目标原点的偏移（厘米），scale 是 Cube 倍数。"""
    rad = math.radians(yaw)
    # 局部坐标按 yaw 旋转到世界坐标
    lx, ly, lz = local_cm
    wx = origin[0] + lx * math.cos(rad) - ly * math.sin(rad)
    wy = origin[1] + lx * math.sin(rad) + ly * math.cos(rad)
    wz = origin[2] + lz
    a = eas.spawn_actor_from_class(unreal.StaticMeshActor,
                                   unreal.Vector(wx, wy, wz),
                                   unreal.Rotator(0.0, yaw, 0.0))
    a.set_actor_label(label)
    a.set_actor_scale3d(unreal.Vector(*scale_xyz))
    comp = a.get_component_by_class(unreal.StaticMeshComponent)
    comp.set_static_mesh(mesh)
    try:
        comp.set_editor_property("render_custom_depth", True)
        comp.set_editor_property("custom_depth_stencil_value", stencil)
    except Exception:
        pass
    return a


def spawn_object(unreal, eas, cls, pos_cm=(1200, 0, 0), yaw=0.0, scale=1.0,
                 label_prefix="OBJ"):
    """在关卡里放一个目标（底部贴地），返回生成的 Actor 列表。

    pos_cm 是**底部中心**（x,y 平面位置，z 一般是 0=地面）。
    """
    st = stencil_of(cls)
    W, D, H = size_cm(cls)
    S = scale
    x, y, z0 = pos_cm

    # 有真实模型就直接用
    asset = ASSET_HINTS.get(cls, "")
    if asset:
        mesh = unreal.load_asset(asset)
        if mesh is not None:
            return [_part(unreal, eas, mesh, "%s_%s" % (label_prefix, cls),
                          (0, 0, H * 0.5 * S), (0.01 * S, 0.01 * S, 0.01 * S), yaw,
                          (x, y, z0), st)]

    cube = _shape(unreal, "Cube")
    cyl = _shape(unreal, "Cylinder")
    sph = _shape(unreal, "Sphere")
    cone = _shape(unreal, "Cone")
    actors = []

    def scale_xyz(w_cm, d_cm, h_cm):
        """厘米 → Cube 倍数（Cube 原始 100cm）。"""
        return (w_cm / 100.0, d_cm / 100.0, h_cm / 100.0)

    if cls in ("car", "suv", "truck"):
        # 车身（占高度的 55%）
        body_h = H * 0.55 * S
        actors.append(_part(unreal, eas, cube, "%s_%s_body" % (label_prefix, cls),
                            (0, 0, body_h * 0.5), scale_xyz(W * S, D * S, body_h),
                            yaw, (x, y, z0), st))
        # 车舱（占 40%，稍窄、靠后）
        cab_h = H * 0.40 * S
        cab_d = D * 0.82 * S
        actors.append(_part(unreal, eas, cube, "%s_%s_cab" % (label_prefix, cls),
                            (-W * 0.02 * S, 0, body_h + cab_h * 0.5),
                            scale_xyz(W * 0.55 * S, cab_d, cab_h),
                            yaw, (x, y, z0), st))
        # 四个轮胎（Cylinder 竖放，直径约 65cm）
        wheel_d = 65.0 * S
        wheel_w = 20.0 * S
        for sx in (-1, 1):
            for sy in (-1, 1):
                lx = sx * W * 0.32 * S
                ly = sy * (D * 0.5 + wheel_w * 0.1) * S
                actors.append(_part(unreal, eas, cyl,
                                    "%s_%s_wheel" % (label_prefix, cls),
                                    (lx, ly, wheel_d * 0.5),
                                    (wheel_w / 100.0, wheel_d / 100.0, wheel_d / 100.0),
                                    yaw + 90.0, (x, y, z0), st))

    elif cls == "pedestrian":
        # 躯干 45%
        torso_h = H * 0.45 * S
        actors.append(_part(unreal, eas, cube, "%s_%s_torso" % (label_prefix, cls),
                            (0, 0, H * 0.30 * S + torso_h * 0.5),
                            scale_xyz(W * S, D * S, torso_h), yaw, (x, y, z0), st))
        # 头（球体，直径约 22cm）
        head_d = 22.0 * S
        actors.append(_part(unreal, eas, sph, "%s_%s_head" % (label_prefix, cls),
                            (0, 0, H * 0.30 * S + torso_h + head_d * 0.5),
                            (head_d / 100.0, head_d / 100.0, head_d / 100.0 * 1.15),
                            yaw, (x, y, z0), st))
        # 两条腿（占底部 30%）
        leg_h = H * 0.30 * S
        for sy in (-1, 1):
            actors.append(_part(unreal, eas, cube, "%s_%s_leg" % (label_prefix, cls),
                                (0, sy * W * 0.25 * S, leg_h * 0.5),
                                scale_xyz(W * 0.38 * S, D * 0.85 * S, leg_h),
                                yaw, (x, y, z0), st))
        # 两条手臂
        arm_h = H * 0.34 * S
        for sy in (-1, 1):
            actors.append(_part(unreal, eas, cube, "%s_%s_arm" % (label_prefix, cls),
                                (0, sy * W * 0.62 * S, H * 0.30 * S + torso_h * 0.72),
                                scale_xyz(W * 0.35 * S, D * 0.75 * S, arm_h),
                                yaw, (x, y, z0), st))

    elif cls == "cone":
        # 锥体（Cone 默认尖朝上，高 100cm，底面直径 100cm）
        actors.append(_part(unreal, eas, cone, "%s_%s_cone" % (label_prefix, cls),
                            (0, 0, H * 0.85 * S * 0.5 + H * 0.15 * S),
                            (W / 100.0 * S, D / 100.0 * S, H * 0.85 * S / 100.0),
                            yaw, (x, y, z0), st))
        # 底座
        actors.append(_part(unreal, eas, cube, "%s_%s_base" % (label_prefix, cls),
                            (0, 0, H * 0.075 * S),
                            scale_xyz(W * 1.25 * S, D * 1.25 * S, H * 0.15 * S),
                            yaw, (x, y, z0), st))

    elif cls == "guardrail":
        # 横梁 + 两根立柱
        actors.append(_part(unreal, eas, cube, "%s_%s_beam" % (label_prefix, cls),
                            (0, 0, H * 0.75 * S),
                            scale_xyz(W * S, D * S, H * 0.22 * S),
                            yaw, (x, y, z0), st))
        for sx in (-0.4, 0.4):
            actors.append(_part(unreal, eas, cube, "%s_%s_post" % (label_prefix, cls),
                                (sx * W * S, 0, H * 0.5 * S),
                                scale_xyz(12 * S, 12 * S, H * S),
                                yaw, (x, y, z0), st))

    else:
        # 未知类别：一个盒子兜底
        actors.append(_part(unreal, eas, cube, "%s_%s" % (label_prefix, cls),
                            (0, 0, H * 0.5 * S),
                            scale_xyz(W * S, D * S, H * S), yaw, (x, y, z0), st))

    return actors


def truth_entry(cls, pos_cm, yaw=0.0, scale=1.0):
    """真值描述：世界盒以「目标中心」给出（与 tsr.py 的约定一致）。"""
    W, D, H = size_cm(cls)
    x, y, z_bottom = pos_cm
    return {
        "class": cls,
        "class_id": stencil_of(cls),
        "world_center_cm": [x, y, z_bottom + H * 0.5 * scale],
        "world_extent_cm": [W * 0.5 * scale, D * 0.5 * scale, H * 0.5 * scale],
        "yaw": yaw,
    }


def batch_layout(count=5, start_x_cm=1500, gap_cm=800, lane_offset_cm=350):
    """生成一排目标的位置（用于批量出图时快速铺满场景）。"""
    classes = ["car", "suv", "truck", "pedestrian", "cone"]
    out = []
    for i in range(count):
        out.append({
            "class": classes[i % len(classes)],
            "pos_cm": [start_x_cm + i * gap_cm,
                       (lane_offset_cm if i % 2 == 0 else -lane_offset_cm), 0],
            "yaw": 180.0 if i % 2 == 0 else 0.0,
            "scale": 1.0,
        })
    return out
