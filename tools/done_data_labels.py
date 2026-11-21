"""解析 Maxieye done_data 的 3D 标注 XML，用 XML 自带标定投影到图上验证。

XML 结构：
  <Lidar>
    <CaptureInfo>
      <Lidar2CamParam><cam_st_x2f_front>
        intrinsic_matrix0..8      (3×3 行优先)
        distortion_coeffs0..7
        rotation_vector0..2       (Rodrigues)
        translation_vector0..2
      </cam_st_x2f_front></Lidar2CamParam>
      <EgoPoseParam>              (车辆全局位姿)
    </CaptureInfo>
    <LabelInfo>
      <Pedestrian Amount="18"><Object kind="" tracking_id=...>...</Object></Pedestrian>
      <Vehicle Amount="53"><Object kind="car">...</Object></Vehicle>
    </LabelInfo>
  </Lidar>

每个 Object 的子标签：
  objLength/objWidth/objHeight   尺寸（米）
  objWx/objWy/objWz              中心位置（米，车体系）
  objRoll/Pitch/Yaw              朝向
  lidarPtsCnt / motion / objCertainty / inVisible / cleanType / delete
"""
import glob
import os

import cv2
import numpy as np

import xml.etree.ElementTree as ET


# 类别归类：XML 的 kind 可能为空
KIND_BY_PARENT = {
    "Vehicle": "car",
    "Pedestrian": "pedestrian",
    "Rider": "rider",
    "TrafficSign": "traffic_sign",
    "TrafficLight": "traffic_light",
    "Cone": "cone",
    "Barrier": "barrier",
}


def rodrigues(rvec):
    """Rodrigues 向量 → 旋转矩阵。"""
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def parse_xml(path):
    """解析一个标注 XML。返回 dict。"""
    root = ET.parse(path).getroot()

    out = {"path": path, "cameras": {}, "objects": [],
           "ego": None, "env": {}}

    # ---- 标定 ----
    l2c = root.find(".//Lidar2CamParam")
    if l2c is not None:
        for cam in l2c:
            g = {c.tag: (c.text or "").strip() for c in cam}
            K = np.array([[float(g.get("intrinsic_matrix%d" % i, 0))
                           for i in range(3)],
                          [float(g.get("intrinsic_matrix%d" % i, 0))
                           for i in range(3, 6)],
                          [float(g.get("intrinsic_matrix%d" % i, 0))
                           for i in range(6, 9)]], float)
            rv = [float(g.get("rotation_vector%d" % i, 0)) for i in range(3)]
            tv = [float(g.get("translation_vector%d" % i, 0)) for i in range(3)]
            dist = [float(g.get("distortion_coeffs%d" % i, 0))
                    for i in range(8)]
            R = rodrigues(rv)
            out["cameras"][cam.tag] = {
                "name": cam.tag,
                "K": K.tolist(),
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "rvec": rv, "tvec": tv,
                "R": R.tolist(), "t": tv,
                "distortion": dist,
                "valid": cam.get("validFlag", "0") == "1",
            }

    # ---- 车辆位姿 ----
    ego = root.find(".//EgoPoseParam")
    if ego is not None:
        g = {c.tag: (c.text or "").strip() for c in ego}
        out["ego"] = {k: float(v) for k, v in g.items()
                      if k in ("wx", "wy", "wz", "roll", "pitch", "yaw")}

    # ---- 环境 ----
    mi = root.find(".//MultiFrameInfo")
    if mi is not None:
        out["env"] = {c.tag: (c.text or "").strip() for c in mi}

    # ---- 物体 ----
    li = root.find(".//LabelInfo")
    if li is not None:
        for grp in li:
            parent_kind = KIND_BY_PARENT.get(grp.tag, grp.tag.lower())
            for o in grp.findall("Object"):
                g = {c.tag: (c.text or "").strip() for c in o}
                kind = o.get("kind") or parent_kind
                rec = {
                    "kind": kind,
                    "parent": grp.tag,
                    "tracking_id": o.get("tracking_id"),
                    "uuid": g.get("uuid", ""),
                    "L": _f(g, "objLength"),
                    "W": _f(g, "objWidth"),
                    "H": _f(g, "objHeight"),
                    "Wx": _f(g, "objWx"),
                    "Wy": _f(g, "objWy"),
                    "Wz": _f(g, "objWz"),
                    "roll": _f(g, "objRoll"),
                    "pitch": _f(g, "objPitch"),
                    "yaw": _f(g, "objYaw"),
                    "pts": int(_f(g, "lidarPtsCnt")),
                    "delete": int(_f(g, "delete")),
                    "invisible": int(_f(g, "inVisible")),
                    "certainty": int(_f(g, "objCertainty")),
                    "cleanType": int(_f(g, "cleanType")),
                }
                out["objects"].append(rec)

    return out


def _f(g, k, d=0.0):
    try:
        return float(g.get(k, d))
    except (TypeError, ValueError):
        return d


def object_corners(o):
    """3D 框 8 角点（车体系）。"""
    L, W, H = o["L"], o["W"], o["H"]
    x, y, z = o["Wx"], o["Wy"], o["Wz"]
    yaw = o["yaw"]

    # 局部框（中心在原点）
    lx = L / 2.0
    ly = W / 2.0
    lz = H / 2.0
    local = np.array([
        [-lx, -ly, -lz], [lx, -ly, -lz], [lx, ly, -lz], [-lx, ly, -lz],
        [-lx, -ly, lz], [lx, -ly, lz], [lx, ly, lz], [-lx, ly, lz],
    ], float)

    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    return local @ Rz.T + np.array([x, y, z], float)


def project_body_to_pixel(pts, cam, use_distortion=False):
    """车体系 3D 点 → 像素。

    XML 的 rotation/translation_vector 是 cam→body 还是 body→cam 需实测确认，
    这里默认按 OpenCV 的 world→camera 处理：p_cam = R @ p_body + t
    """
    P = np.asarray(pts, float).reshape(-1, 3)
    R = np.array(cam["R"], float)
    t = np.array(cam["t"], float)
    K = np.array(cam["K"], float)

    Pc = P @ R.T + t
    z = Pc[:, 2]
    ok = z > 1e-6
    uv = np.full((len(P), 2), np.nan)
    if ok.any():
        q = Pc[ok] @ K.T
        uv[ok, 0] = q[:, 0] / q[:, 2]
        uv[ok, 1] = q[:, 1] / q[:, 2]
    return uv, z, ok


def find_label_dir(dataset_root="/mnt/done_data_new/18_bev-obj"):
    """找所有含已终审标注的批次目录。"""
    out = []
    for p in glob.glob(os.path.join(dataset_root, "*", "*", "已终审", "*", "*")):
        if os.path.isdir(p):
            out.append(p)
    return out


def list_frames(batch_dir):
    """列出批次下的帧目录。"""
    return sorted([p for p in glob.glob(os.path.join(batch_dir, "*"))
                   if os.path.isdir(p)])


if __name__ == "__main__":
    import sys

    B = ("/mnt/done_data_new/18_bev-obj/00_M57/00_daizhongshen_changchun/"
         "已终审/hs5001/20261120_HS5001_changchun_1v_closecar/"
         "133007-1788053647.136")
    F = os.path.join(B, "label_lidar_hs_pandar128_topmiddle",
                     "1788053647.136.xml")

    d = parse_xml(F)
    print("=== 相机标定 ===")
    for name, c in d["cameras"].items():
        print("  %s (valid=%s)" % (name, c["valid"]))
        print("    fx=%.2f fy=%.2f cx=%.2f cy=%.2f"
              % (c["fx"], c["fy"], c["cx"], c["cy"]))
        print("    rvec=%s" % np.round(c["rvec"], 3).tolist())
        print("    tvec=%s" % np.round(c["tvec"], 3).tolist())

    print("\n=== 环境 ===", d["env"])
    print("=== Ego ===", d["ego"])

    print("\n=== 物体 %d 个 ===" % len(d["objects"]))
    from collections import Counter
    print("  类别:", dict(Counter(o["kind"] for o in d["objects"])))

    # 投影测试
    cam = d["cameras"].get("cam_st_x2f_front")
    if cam:
        print("\n=== 投影测试（前 5 个 car）===")
        cars = [o for o in d["objects"] if o["kind"] == "car"][:5]
        for o in cars:
            c8 = object_corners(o)
            uv, z, ok = project_body_to_pixel(c8, cam)
            if not ok.all():
                print("  %s: 部分点在相机后方" % o["tracking_id"])
                continue
            x0, y0 = uv[:, 0].min(), uv[:, 1].min()
            x1, y1 = uv[:, 0].max(), uv[:, 1].max()
            print("  id=%-6s 世界(%.1f,%.1f,%.1f) → 像素框 (%.0f,%.0f)-(%.0f,%.0f) z=%.1fm"
                  % (o["tracking_id"], o["Wx"], o["Wy"], o["Wz"],
                     x0, y0, x1, y1, z.mean()))
