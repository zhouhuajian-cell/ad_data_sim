#!/usr/bin/env python3
"""把仿真产物导出为真实 to_label 格式的 XML（OBJ）。

目标格式（与 18_bev-obj/.../已送/<clip>/<id-ts>/label_lidar_*/<ts>.xml 完全同构）：
  <Lidar>
    <CaptureInfo>
      <EgoPoseParam validFlag="1"> wx,wy,wz,roll,pitch,yaw,qx,qy,qz,qw
      <Lidar2CamParam><cam_st_x2f_front>
          intrinsic_matrix0..8, distortion_coeffs0..7,
          rotation_vector0..2, translation_vector0..2
      </cam_st_x2f_front></Lidar2CamParam>
      <VehicleInfo validFlag="0">...</VehicleInfo>
      <MultiFrameInfo validFlag="1"/>
    </CaptureInfo>
    <LabelInfo>
      <Three Amount="n"/> <Cone Amount="n"/> <Unknown/> <Unsure/> <Special/>
      <Vehicle Amount="n">
        <Object ID="i" tracking_id="t" kind="car">
          uuid, objWx, objWy, objWz, objLength, objWidth, objHeight,
          objYaw, objVelocityX/Y/Z, objRoll, objPitch, objCertainty,
          motion, lidarPtsCnt, inVisible
        </Object>
      </Vehicle>
    </LabelInfo>
  </Lidar>

用法:
    python export_to_label.py --ann <sim_annotation.json> --out <dir>
"""
import argparse
import json
import math
import os
import uuid as _uuid
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

# 我们的类别 -> 真实 kind
KIND_MAP = {
    "car": "car", "sedan": "car", "suv": "car", "taxi": "car",
    "police": "car", "van": "van", "truck": "truck", "bus": "bus",
    "tractor": "truck",
}
# GROUP-FULL-8: 真实的 8 个大组
#   Three(三轮车) / Cone(锥桶) / Unknown / Unsure / Special
#   Vehicle(车辆) / Rider(骑行者) / Pedestrian(行人)
GROUP_MAP = {
    "cone": "Cone",
    "traffic_cone": "Cone",
    "three": "Three",
    "tricycle": "Three",
    "rider": "Rider",
    "bicycle": "Rider",
    "motorcycle": "Rider",
    "scooter": "Rider",
    "cyclist": "Rider",
    "pedestrian": "Pedestrian",
    "person": "Pedestrian",
}
# Pedestrian 组专有字段
GROUP_EXTRA = {
    "Pedestrian": ["crowd"],
}


def _fmt(v, nd=6):
    return ("%." + str(nd) + "f") % float(v)


def _uuid_hex(seed):
    """由种子生成稳定的 uuid（同 tracking_id 跨帧一致）。"""
    return _uuid.uuid5(_uuid.NAMESPACE_OID, str(seed)).hex


def _emit_group(parent, gname, objs):
    """GROUP-FULL-8: 输出一个大组（含 Object 列表）。"""
    grp = ET.SubElement(parent, gname, {"Amount": str(len(objs))})
    extras = GROUP_EXTRA.get(gname, [])
    for idx, o in enumerate(objs):
        tid = int(o.get("tracking_id", idx))
        attrs = {"ID": str(idx), "tracking_id": str(tid)}
        kind = o.get("kind") or KIND_MAP.get(
            str(o.get("cls", "")).lower(), "car")
        if gname == "Vehicle":
            attrs["kind"] = kind
        obj = ET.SubElement(grp, "Object", attrs)
        ET.SubElement(obj, "uuid").text = o.get("uuid") or _uuid_hex(tid)
        ET.SubElement(obj, "objWx").text = _fmt(o.get("x", 0.0))
        ET.SubElement(obj, "objWy").text = _fmt(o.get("y", 0.0))
        ET.SubElement(obj, "objWz").text = _fmt(o.get("z", 0.0))
        ET.SubElement(obj, "objLength").text = _fmt(o.get("l", 4.5))
        ET.SubElement(obj, "objWidth").text = _fmt(o.get("w", 1.8))
        ET.SubElement(obj, "objHeight").text = _fmt(o.get("h", 1.5))
        ET.SubElement(obj, "objYaw").text = _fmt(o.get("yaw", 0.0))
        ET.SubElement(obj, "objVelocityX").text = _fmt(o.get("vx", 0.0), 4)
        ET.SubElement(obj, "objVelocityY").text = _fmt(o.get("vy", 0.0), 4)
        ET.SubElement(obj, "objVelocityZ").text = _fmt(o.get("vz", 0.0), 4)
        ET.SubElement(obj, "objRoll").text = _fmt(o.get("roll", 0.0), 1)
        ET.SubElement(obj, "objPitch").text = _fmt(o.get("pitch", 0.0), 1)
        ET.SubElement(obj, "objCertainty").text = _fmt(
            o.get("certainty", 0.9))
        ET.SubElement(obj, "motion").text = str(int(o.get("motion", 1)))
        ET.SubElement(obj, "lidarPtsCnt").text = str(int(o.get("pts", 0)))
        ET.SubElement(obj, "inVisible").text = str(int(o.get("visible", 1)))
        for ex in extras:
            ET.SubElement(obj, ex).text = str(int(o.get(ex, 0)))
    return grp


def build_xml(ego, K, distort, rotvec, tvec, objects, vehicle_info=None,
              timestamp=None):
    """构造 XML 树。

    ego: dict(wx,wy,wz,roll,pitch,yaw,qx,qy,qz,qw)
    K: 长度 9 的 list（行主序 3x3）
    distort: 长度 8 的 list
    rotvec: 长度 3（LiDAR->Cam 旋转向量）
    tvec: 长度 3（LiDAR->Cam 平移）
    objects: [{'kind':..., 'cls':..., 'x','y','z','l','w','h','yaw',
               'vx','vy','vz','certainty','motion','pts','visible',
               'tracking_id'}]
    """
    root = ET.Element("Lidar")

    # ---------- CaptureInfo ----------
    ci = ET.SubElement(root, "CaptureInfo")

    ep = ET.SubElement(ci, "EgoPoseParam", {"validFlag": "1"})
    for k in ("wx", "wy", "wz"):
        ET.SubElement(ep, "w" + k[1] if False else k).text = _fmt(ego[k])
    for k in ("roll", "pitch", "yaw"):
        ET.SubElement(ep, k).text = _fmt(ego[k])
    for k in ("qx", "qy", "qz", "qw"):
        ET.SubElement(ep, k).text = _fmt(ego.get(k, 0.0))

    l2c = ET.SubElement(ci, "Lidar2CamParam")
    cam = ET.SubElement(l2c, "cam_st_x2f_front")
    for i in range(9):
        ET.SubElement(cam, "intrinsic_matrix%d" % i).text = _fmt(K[i])
    for i in range(8):
        ET.SubElement(cam, "distortion_coeffs%d" % i).text = _fmt(
            distort[i] if i < len(distort) else 0.0)
    for i in range(3):
        ET.SubElement(cam, "rotation_vector%d" % i).text = _fmt(rotvec[i])
    for i in range(3):
        ET.SubElement(cam, "translation_vector%d" % i).text = _fmt(tvec[i])

    vi = ET.SubElement(ci, "VehicleInfo",
                       {"validFlag": "1" if vehicle_info else "0"})
    vin = vehicle_info or {}
    for k, dv in (("Timestamp", -999.0), ("Yawrate", -999.0),
                  ("Vehiclespeed", -999.0), ("BrkPdlPos", -999.0),
                  ("WhlAng", -999.0), ("WhlAngGrd", -999.0),
                  ("AccelActuPos", -999.0)):
        ET.SubElement(vi, k).text = _fmt(vin.get(k, dv))
    ET.SubElement(ci, "MultiFrameInfo", {"validFlag": "1"})

    # ---------- LabelInfo ----------
    li = ET.SubElement(root, "LabelInfo")

    groups = {"Three": [], "Cone": [], "Unknown": [], "Unsure": [],
              "Special": [], "Vehicle": [], "Rider": [], "Pedestrian": []}
    for o in objects:
        cls = str(o.get("cls", "")).lower()
        if cls in GROUP_MAP:
            groups[GROUP_MAP[cls]].append(o)
        else:
            groups["Vehicle"].append(o)

    # 空组直接自闭合
    for gname in ("Three", "Cone", "Unknown", "Unsure", "Special"):
        if groups[gname]:
            _emit_group(li, gname, groups[gname])
        else:
            ET.SubElement(li, gname, {"Amount": "0"})

    # Vehicle / Rider / Pedestrian（有对象就展开）
    for gname in ("Vehicle", "Rider", "Pedestrian"):
        _emit_group(li, gname, groups[gname])

    return root


def write_pretty(root, path):
    """按真实数据风格（2 空格缩进）写出。"""
    raw = ET.tostring(root, encoding="utf-8")
    dom = minidom.parseString(raw)
    txt = dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    # 去掉 minidom 产生的空行
    lines = [ln for ln in txt.split("\n") if ln.strip()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def xml_to_objects(ann, cam_h=1.5):
    """把我们的仿真标注转成 XML Object 列表。

    我们的 annotation 里 objects 形如：
      {'class': 'car', 'bbox': [...], 'world': {'x':..,'y':..},
       'size': {'L':..,'W':..,'H':..}, 'yaw': .., ...}
    """
    out = []
    for i, o in enumerate(ann.get("objects") or []):
        cls = str(o.get("class") or o.get("cls") or "car").lower()
        world = o.get("world") or {}
        size = o.get("size") or {}
        L = float(size.get("L") or size.get("length") or 4.5)
        W = float(size.get("W") or size.get("width") or 1.8)
        H = float(size.get("H") or size.get("height") or 1.5)
        x = float(world.get("x", o.get("x", 0.0)))
        y = float(world.get("y", o.get("y", 0.0)))
        yaw = float(o.get("yaw", 0.0))
        # objWz = 车辆中心离地高度
        z = H / 2.0
        out.append({
            "cls": cls,
            "kind": KIND_MAP.get(cls, "car"),
            "x": x, "y": y, "z": z,
            "l": L, "w": W, "h": H,
            "yaw": yaw,
            "vx": float(o.get("vx", 0.0)),
            "vy": float(o.get("vy", 0.0)),
            "vz": 0.0,
            "certainty": float(o.get("score", o.get("certainty", 0.9))),
            "motion": 1 if o.get("synthetic") else 0,
            "pts": int(o.get("lidar_pts", 0)),
            "visible": 1,
            "tracking_id": int(o.get("tracking_id", i)),
            "uuid": o.get("uuid"),
        })
    return out


def build_objects_from_items(ann, cam_h=1.5):
    """BUILD-OBJ-FROM-ITEMS: 兼容多种输入结构的对象转换。

    支持:
      {'objects': [{'class','world':{x,y},'size':{L,W,H},'yaw',...}]}
      [{'cls':'car','x':..,'y':..,'L':..,...}]
    """
    if isinstance(ann, dict):
        items = ann.get("objects") or []
    elif isinstance(ann, list):
        items = ann
    else:
        items = []

    out = []
    for i, o in enumerate(items):
        cls = str(o.get("class") or o.get("cls") or "car").lower()
        world = o.get("world") or {}
        size = o.get("size") or {}
        L = float(size.get("L") or o.get("L") or
                  o.get("length") or 4.5)
        W = float(size.get("W") or o.get("W") or o.get("width") or 1.8)
        H = float(size.get("H") or o.get("H") or o.get("height") or 1.5)
        x = float(world.get("x", o.get("x", 0.0)))
        y = float(world.get("y", o.get("y", 0.0)))
        z = float(o.get("z", H / 2.0))
        yaw = float(o.get("yaw", 0.0))
        out.append({
            "cls": cls,
            "kind": KIND_MAP.get(cls, "car"),
            "x": x, "y": y, "z": z,
            "l": L, "w": W, "h": H,
            "yaw": yaw,
            "vx": float(o.get("vx", 0.0)),
            "vy": float(o.get("vy", 0.0)),
            "vz": 0.0,
            "certainty": float(o.get("score", o.get("certainty", 0.9))),
            "motion": 1 if o.get("synthetic") else 0,
            "pts": int(o.get("lidar_pts", 0)),
            "visible": 1,
            "tracking_id": int(o.get("tracking_id", i)),
            "uuid": o.get("uuid"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="仿真标注 json 路径")
    # OUTROOT-LOCKED: 默认输出到网盘 zhj_datasim
    ap.add_argument("--out", default="/mnt/Data_Platform/zhj_datasim/OBJ",
                    help="输出目录（默认网盘 zhj_datasim/OBJ）")
    ap.add_argument("--ts", default=None, help="时间戳（默认用文件名）")
    ap.add_argument("--ego", default=None, help="ego pose json")
    ap.add_argument("--k", default=None, help="相机内参 json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ann = json.load(open(args.ann, encoding="utf-8"))

    ts = args.ts or os.path.splitext(os.path.basename(args.ann))[0]

    # ego pose
    if args.ego and os.path.isfile(args.ego):
        ego = json.load(open(args.ego))
    else:
        ego = {"wx": 0.0, "wy": 0.0, "wz": 0.0,
               "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
               "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

    # 内参
    if args.k and os.path.isfile(args.k):
        kj = json.load(open(args.k))
        fx, fy = kj["fx"], kj["fy"]
        cx, cy = kj["cx"], kj["cy"]
    else:
        fx, fy, cx, cy = 1319.71, 1319.36, 953.25, 540.01
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    distort = [0.0] * 8

    # LiDAR->Cam（LiDAR 在车身原点，相机在前上方）
    # 近似：绕 X 轴 -90 度（LiDAR 坐标系 z 向上 -> 相机 z 向前）
    rotvec = [math.pi / 2, 0.0, 0.0]
    tvec = [0.0, 1.5, 1.2]

    objs = xml_to_objects(ann)
    root = build_xml(ego, K, distort, rotvec, tvec, objs,
                     vehicle_info=None, timestamp=ts)

    fp = os.path.join(args.out, "%s.xml" % ts)
    write_pretty(root, fp)
    print("写出: %s  (%d 个对象)" % (fp, len(objs)))

    # 同时写 _ctrl.txt（真实数据里有）
    ctrl = os.path.join(args.out, "%s_ctrl.txt" % ts)
    with open(ctrl, "w", encoding="utf-8") as f:
        f.write("")     # 空白，真实样本也是空的
    print("写出: %s" % ctrl)


if __name__ == "__main__":
    main()
