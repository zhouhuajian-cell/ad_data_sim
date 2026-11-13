# -*- coding: utf-8 -*-
# capture_math.py — 四通道捕获用的纯数学（不放引擎依赖，Windows 上也能单独测）
# 干什么：相机基向量、世界点投屏、三维框投影成二维框、屏幕裁切、按深度归一化、遮挡比例近似。
# 为什么独立成文件：batch_capture.py 顶部 import unreal，只能在引擎里跑；本文件只用 math，
#   可由 tools/dryrun_check.py 在引擎外逐条验证投影公式（阶段三验收里最容易错的一环）。
# 坐标系：UE 左手系，X 前、Y 右、Z 上；yaw 绕 Z（+90 = +Y），pitch 绕 Y（+ = 抬头），roll 忽略。
# 像素坐标：左上角原点，x 向右、y 向下，与附录 B 的 bbox_2d「左上与右下」一致。
import math

__all__ = [
    "camera_basis", "world_to_screen", "project_aabb", "clip_bbox",
    "box_area", "boxes_overlap_ratio", "normalize_depth",
    "hfov_from_focal", "focal_from_hfov",
]


def _rad(deg):
    return float(deg) * math.pi / 180.0


def camera_basis(yaw_deg, pitch_deg):
    """返回 (forward, right, up) 三个单位向量；roll 固定为 0。"""
    y, p = _rad(yaw_deg), _rad(pitch_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    forward = (cp * cy, cp * sy, sp)          # UE FRotator 的 X 轴
    right = (-sy, cy, 0.0)                    # yaw+90° 方向
    up = (forward[1] * right[2] - forward[2] * right[1],
          forward[2] * right[0] - forward[0] * right[2],
          forward[0] * right[1] - forward[1] * right[0])   # forward × right
    return forward, right, up


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def world_to_screen(point, cam_loc, yaw_deg, pitch_deg, hfov_deg, width, height):
    """世界点 → 像素坐标。

    返回 (u, v, depth)：depth 为沿光轴距离（米，>0 表示在相机前方）；u/v 允许落在画面外，
    需要判可见性时用 depth > 0 且 0<=u<=width、0<=v<=height 自行判断。
    焦距按「方形像素」由水平 FOV 与画面宽度推出，因此竖屏/非 16:9 也不会拉伸。
    """
    fwd, right, up = camera_basis(yaw_deg, pitch_deg)
    rel = (point[0] - cam_loc[0], point[1] - cam_loc[1], point[2] - cam_loc[2])
    depth = _dot(rel, fwd)
    if depth <= 1e-6:
        # 相机后方或正好在光心上：仍返回有限值供调用方判断，除以极小正数避免除零
        safe = 1e-6
        focal = (width / 2.0) / math.tan(_rad(hfov_deg) / 2.0)
        return (width / 2.0 + focal * _dot(rel, right) / safe,
                height / 2.0 - focal * _dot(rel, up) / safe, depth)
    focal = (width / 2.0) / math.tan(_rad(hfov_deg) / 2.0)
    u = width / 2.0 + focal * _dot(rel, right) / depth
    v = height / 2.0 - focal * _dot(rel, up) / depth
    return (u, v, depth)


def _rotate_z(vec, deg):
    r = _rad(deg)
    c, s = math.cos(r), math.sin(r)
    return (vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c, vec[2])


def project_aabb(center, extent, object_yaw_deg, cam_loc, cam_yaw_deg, cam_pitch_deg,
                 hfov_deg, width, height, near=0.1):
    """三维包围盒（中心 + 半尺寸，绕 Z 旋转）→ 二维像素框。

    返回 (x1, y1, x2, y2, depth, truncated)：
      depth 取 8 个角点在光轴方向的最小正值（最近处，用于距离分层）；
      truncated=True 表示框有一部分被画面边界截断（返回的框已裁到画面内）。
    盒子整体在相机后方/贴脸不可见时返回 None。
    """
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                off = (sx * extent[0], sy * extent[1], sz * extent[2])
                off = _rotate_z(off, object_yaw_deg)
                pts.append((center[0] + off[0], center[1] + off[1], center[2] + off[2]))
    xs, ys, depths = [], [], []
    for p in pts:
        u, v, d = world_to_screen(p, cam_loc, cam_yaw_deg, cam_pitch_deg,
                                  hfov_deg, width, height)
        if d <= near:
            continue
        xs.append(u)
        ys.append(v)
        depths.append(d)
    if not xs:
        return None
    raw = (min(xs), min(ys), max(xs), max(ys))
    clipped = clip_bbox(raw, width, height)
    if clipped is None:
        return None
    truncated = (clipped != raw)
    return (clipped[0], clipped[1], clipped[2], clipped[3], min(depths), truncated)


def clip_bbox(bbox, width, height):
    """把框裁到画面内；完全在画面外返回 None。"""
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (x1, y1, x2, y2)


def box_area(box):
    if not box:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def boxes_overlap_ratio(box, distance, others):
    """遮挡比例近似：被「更近的」框覆盖的面积占自身面积之比，裁到 0~1。

    others 为 [(box, distance)]；只统计比自身更近的目标（被挡住的语义），
    同距离或更远的不算遮挡。多个近目标的重叠区域会重复计面积，属上界估计；
    真实遮挡比例仍应在编辑器内用深度/模板逐帧复核（阶段三验收项）。
    """
    a = box_area(box)
    if a <= 0:
        return 0.0
    x1, y1, x2, y2 = box
    covered = 0.0
    for ob, dist in others:
        if not ob or dist >= distance:
            continue
        ox1, oy1, ox2, oy2 = ob
        ix = max(0.0, min(x2, ox2) - max(x1, ox1))
        iy = max(0.0, min(y2, oy2) - max(y1, oy1))
        covered += ix * iy
    return max(0.0, min(1.0, covered / a))


def normalize_depth(depth_m, far=200.0):
    """0~200m 线性映射到 0~65535（方案 3.4：深度落 16bit 单通道）。"""
    if depth_m <= 0:
        return 0
    ratio = min(1.0, float(depth_m) / float(far))
    return int(round(ratio * 65535.0))


def hfov_from_focal(focal_mm, film_width_mm):
    """由焦距与胶片背板宽度算水平 FOV（度）。"""
    if focal_mm <= 0 or film_width_mm <= 0:
        raise ValueError("焦距与胶片背板宽度必须为正")
    return 2.0 * math.degrees(math.atan(film_width_mm / (2.0 * focal_mm)))


def focal_from_hfov(hfov_deg, film_width_mm):
    """由水平 FOV 与胶片背板宽度反算焦距（毫米）。"""
    if not (0 < hfov_deg < 180):
        raise ValueError("水平 FOV 必须在 (0, 180) 度之间")
    return film_width_mm / (2.0 * math.tan(_rad(hfov_deg) / 2.0))
