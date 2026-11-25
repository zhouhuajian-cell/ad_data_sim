#!/usr/bin/env python3
"""3D 碰撞校验 + 遮挡掩码（替代 2D 像素校验）。

核心判据（不只看点云距离，而是完整的 3D 关系）：
  ① 真实车 3D 包围盒（来自逐帧 XML 标注，53 辆/帧）
  ② OBB-OBB 分离轴检测（虚拟车 vs 真实车）
  ③ 安全余量（纵向 1.5m / 横向 0.8m）
  ④ 遮挡掩码（3D 在前 -> 允许 2D 重叠，标 occluded）
  ⑤ 时间同步检查（LiDAR 与图像时间差）
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np


# ============================================================
# ① 真实车 3D 框解析
# ============================================================
def load_real_vehicles(xml_path):
    """从逐帧 XML 读出所有真实车辆的 3D 框。

    Returns: list of dict(cx,cy,cz,L,W,H,yaw,vx,vy,n_pts,uuid)
    """
    out = []
    if not os.path.isfile(xml_path):
        return out
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return out

    for vnode in root.findall(".//LabelInfo/Vehicle"):
        for o in vnode.findall("Object"):
            try:
                d = {
                    "uuid": (o.findtext("uuid") or "").strip(),
                    "cx": float(o.findtext("objWx") or 0),
                    "cy": float(o.findtext("objWy") or 0),
                    "cz": float(o.findtext("objWz") or 0),
                    "L": float(o.findtext("objLength") or 0),
                    "W": float(o.findtext("objWidth") or 0),
                    "H": float(o.findtext("objHeight") or 0),
                    "yaw": float(o.findtext("objYaw") or 0),
                    "vx": float(o.findtext("objVelocityX") or 0),
                    "vy": float(o.findtext("objVelocityY") or 0),
                    "n_pts": int(o.findtext("lidarPtsCnt") or 0),
                    "deleted": int(o.findtext("delete") or 0),
                }
            except (TypeError, ValueError):
                continue
            if d["deleted"] or d["L"] <= 0.1 or d["W"] <= 0.1:
                continue
            out.append(d)
    return out


def load_pedestrians(xml_path):
    """行人也算障碍物。"""
    out = []
    if not os.path.isfile(xml_path):
        return out
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return out
    for tag in ("Pedestrian", "Rider", "Three"):
        for vnode in root.findall(".//LabelInfo/" + tag):
            for o in vnode.findall("Object"):
                try:
                    L = float(o.findtext("objLength") or 0.5)
                    W = float(o.findtext("objWidth") or 0.5)
                    if L <= 0.05 or W <= 0.05:
                        L = W = 0.5
                    out.append({
                        "uuid": (o.findtext("uuid") or "").strip(),
                        "cx": float(o.findtext("objWx") or 0),
                        "cy": float(o.findtext("objWy") or 0),
                        "cz": float(o.findtext("objWz") or 0),
                        "L": L, "W": W,
                        "H": float(o.findtext("objHeight") or 1.7),
                        "yaw": float(o.findtext("objYaw") or 0),
                        "vx": float(o.findtext("objVelocityX") or 0),
                        "vy": float(o.findtext("objVelocityY") or 0),
                        "n_pts": int(o.findtext("lidarPtsCnt") or 0),
                        "deleted": int(o.findtext("delete") or 0),
                        "cls": tag,
                    })
                except (TypeError, ValueError):
                    continue
    return [o for o in out if not o["deleted"]]


# ============================================================
# ② OBB-OBB 分离轴检测（3D）
# ============================================================
def _obb_axes(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def obb_gap(a, b, margin_long=1.5, margin_lat=0.8):
    """两个 OBB 的最小分离距离（>0 表示不碰撞）。

    a, b: dict(cx, cy, cz, L, W, H, yaw)
    返回 (gap, reason)
      gap > 0  -> 安全
      gap <= 0 -> 碰撞
    """
    La = a["L"] + margin_long * 2
    Wa = a["W"] + margin_lat * 2
    Lb = b["L"] + margin_long * 2
    Wb = b["W"] + margin_lat * 2

    Ra = _obb_axes(a["yaw"])
    Rb = _obb_axes(b["yaw"])
    d = np.array([b["cx"] - a["cx"], b["cy"] - a["cy"],
                  b["cz"] - a["cz"]])

    # 半长
    ha = np.array([La / 2, Wa / 2, a["H"] / 2])
    hb = np.array([Lb / 2, Wb / 2, b["H"] / 2])

    # 6 个轴 + 9 个叉积轴
    axes = []
    for i in range(3):
        axes.append(Ra[:, i])
        axes.append(Rb[:, i])
    for i in range(3):
        for j in range(3):
            ax = np.cross(Ra[:, i], Rb[:, j])
            n = np.linalg.norm(ax)
            if n > 1e-6:
                axes.append(ax / n)

    min_gap = 1e9
    for ax in axes:
        proj_a = abs(ax @ (Ra @ ha * 1.0)) if False else 0.0
        ra = sum(ha[k] * abs(ax @ Ra[:, k]) for k in range(3))
        rb = sum(hb[k] * abs(ax @ Rb[:, k]) for k in range(3))
        dist = abs(ax @ d)
        gap = dist - (ra + rb)
        if gap < min_gap:
            min_gap = gap
        if gap > 0:
            # 有分离轴 -> 3D 不重叠
            return float(gap), "separated"

    return float(min_gap), "overlap"


def check_placement(virt, reals, margin_long=1.5, margin_lat=0.8,
                    min_real_pts=8):
    """校验虚拟车位姿是否与所有真实障碍物 3D 碰撞。

    Returns: (ok, info)
    """
    virt_obb = {"cx": virt["x"], "cy": virt["y"],
                "cz": virt.get("z", 0) + virt["size"][2] / 2,
                "L": virt["size"][0], "W": virt["size"][1],
                "H": virt["size"][2], "yaw": math.radians(virt["yaw"])}

    worst = None
    for r in reals:
        if r["n_pts"] < min_real_pts:
            continue      # 点太少的不可信，跳过
        # 粗筛：距离超过 30m 不可能碰撞
        dist = math.hypot(r["cx"] - virt_obb["cx"],
                          r["cy"] - virt_obb["cy"])
        if dist > 30:
            continue
        real_obb = {"cx": r["cx"], "cy": r["cy"], "cz": r["cz"],
                    "L": r["L"], "W": r["W"], "H": r["H"],
                    "yaw": r["yaw"]}
        gap, why = obb_gap(virt_obb, real_obb, margin_long, margin_lat)
        if why == "overlap":
            return False, {"reason": "3d_collision",
                           "hit": r.get("uuid", ""),
                           "gap": gap}
        if worst is None or gap < worst[0]:
            worst = (gap, r.get("uuid", ""))

    return True, {"reason": "ok",
                  "min_gap": worst[0] if worst else None}


# ============================================================
# ③ 遮挡判定（3D 在前 -> 允许 2D 重叠）
# ============================================================
def occlusion_status(virt, reals, K, R, t, inst_mask=None,
                     min_real_pts=8):
    """判断虚拟车是否被真实物体遮挡（不是碰撞）。

    如果真实物在虚拟车【前方】且 2D 投影重叠 -> 合理遮挡，保留。
    返回 (occluded, ratio, blockers)
    """
    from lidar_depth_core import project_points

    # 虚拟车 8 角点
    L, W, H = virt["size"]
    yaw = math.radians(virt["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for dx in (-L / 2, L / 2):
        for dy in (-W / 2, W / 2):
            for dz in (0.0, H):
                corners.append([virt["x"] + dx * c - dy * s,
                                virt["y"] + dx * s + dy * c,
                                virt.get("z", 0) + dz])
    uv, zc, ok = project_points(np.array(corners), K, R, t)
    if not ok.all() or not np.isfinite(uv).all():
        return False, 0.0, []
    vx0, vy0 = uv[:, 0].min(), uv[:, 1].min()
    vx1, vy1 = uv[:, 0].max(), uv[:, 1].max()
    v_depth = float(np.median(zc[ok]))

    blockers = []
    for r in reals:
        if r["n_pts"] < min_real_pts:
            continue
        # 真实物必须比虚拟车【近】
        rr = {"cx": r["cx"], "cy": r["cy"], "cz": r["cz"],
              "L": r["L"], "W": r["W"], "H": r["H"], "yaw": r["yaw"]}
        rc, _ = _obb_corners(rr)
        ruv, rzc, rok = project_points(rc, K, R, t)
        if not rok.all() or not np.isfinite(ruv).all():
            continue
        r_depth = float(np.median(rzc[rok]))
        if r_depth >= v_depth - 0.3:
            continue        # 不在前方
        # 2D 框重叠
        rx0, ry0 = ruv[:, 0].min(), ruv[:, 1].min()
        rx1, ry1 = ruv[:, 0].max(), ruv[:, 1].max()
        ox = max(0.0, min(vx1, rx1) - max(vx0, rx0))
        oy = max(0.0, min(vy1, ry1) - max(vy0, ry0))
        if ox * oy > 1.0:
            blockers.append({"uuid": r.get("uuid", ""),
                             "depth": round(r_depth, 2),
                             "overlap_px": round(ox * oy, 1)})

    if not blockers:
        return False, 0.0, []

    # 估算遮挡面积比
    v_area = max((vx1 - vx0) * (vy1 - vy0), 1.0)
    occ_area = sum(b["overlap_px"] for b in blockers)
    ratio = min(occ_area / v_area, 1.0)
    return True, ratio, blockers


def _obb_corners(o):
    L, W, H = o["L"], o["W"], o["H"]
    yaw = o["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx in (-L / 2, L / 2):
        for dy in (-W / 2, W / 2):
            for dz in (-H / 2, H / 2):
                pts.append([o["cx"] + dx * c - dy * s,
                            o["cy"] + dx * s + dy * c,
                            o["cz"] + dz])
    return np.array(pts), np.array([o["cx"], o["cy"], o["cz"]])


# ============================================================
# ④ 时间同步检查
# ============================================================
def check_time_sync(img_path, xml_path, lidar_path, max_skew=0.15):
    """检查图像 / LiDAR / 标注时间戳是否对齐。

    Returns: dict(ok, skew, note)
    """
    def ts_of(p):
        b = os.path.splitext(os.path.basename(p))[0]
        try:
            return float(b.split("-")[-1])
        except Exception:
            return None

    ti = ts_of(img_path)
    tx = ts_of(xml_path)
    tl = ts_of(lidar_path)
    vals = [v for v in (ti, tx, tl) if v is not None]
    if len(vals) < 2:
        return {"ok": True, "skew": None, "note": "无时间戳"}
    skew = max(vals) - min(vals)
    return {"ok": skew <= max_skew,
            "skew": round(skew, 4),
            "note": "ok" if skew <= max_skew else "时间差 %.3fs" % skew}
