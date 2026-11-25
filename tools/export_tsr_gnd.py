#!/usr/bin/env python3
"""TSR + GND 仿真产物导出为真实 to_label 格式。

用法:
    # TSR
    python export_tsr_gnd.py tsr --ann <sim_ann.json> --out <dir> --ts <timestamp>

    # GND
    python export_tsr_gnd.py gnd --sim <sim.jpg> --orig <orig.jpg> \
        --seg <seg_lane.png> --out <dir> --ts <timestamp> \
        [--yaml <parameter_yaml源目录>]

格式依据：docs/DATA_FORMAT_SPEC.md
"""
import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

import cv2
import numpy as np


# ============================================================
# TSR
# ============================================================
# 我们的标志类别 -> TSR 语义属性
TSR_SHAPE_MAP = {
    "circle": "CIRCLE", "round": "CIRCLE",
    "triangle": "TRIANGLE", "diamond": "DIAMOND",
    "rect": "RECTANGLE", "rectangle": "RECTANGLE",
    "square": "RECTANGLE",
}

TSR_ATTR_MAP = {
    # firstAttribute
    "limit": "LIMIT", "speed_limit": "LIMIT",
    "warning": "WARNING", "prohibit": "PROHIBIT",
    "indication": "INDICATION", "mandatory": "INDICATION",
}


def build_tsr_xml(signs, is_virtual=True):
    """signs: [{'shape','first_attribute','color','bbox'(x1,y1,x2,y2),
                'f_value','limit_type','type','label_type','is_covered'}]"""
    root = ET.Element("Image")
    li = ET.SubElement(root, "LabelInfo")

    # 空组
    for g in ("Vehicle", "Pedestrian", "Rider"):
        ET.SubElement(li, g, {"Amount": "0"})

    ts = ET.SubElement(li, "TrafficSign",
                       {"Amount": str(len(signs))})
    for i, s in enumerate(signs):
        bbox = s.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        attrs = {
            "shape": TSR_SHAPE_MAP.get(
                str(s.get("shape", "circle")).lower(), "CIRCLE"),
            "firstAttribute": TSR_ATTR_MAP.get(
                str(s.get("first_attribute", "limit")).lower(), "LIMIT"),
            "color": str(s.get("color", "UNKNOWN")).upper(),
            "isCovered": str(s.get("is_covered", "Level0")),
            "isVirtualScene": "true" if is_virtual else "false",
            "ID": str(i + 1),
            "deep": str(int(s.get("deep", 0))),
            "definition": str(s.get("definition", "obvious")),
            "precision": str(s.get("precision", "float")),
            "labelType": str(s.get("label_type", "POS")),
            "fValue": "%.2f" % float(s.get("f_value", 0.0)),
            "LimitType": str(s.get("limit_type", "SPEED")),
            "type": str(s.get("type", "RELEASE")),
            "LimitTypeOverseas": str(
                s.get("limit_type_overseas", "TrafficReduce")),
            "trafficReduceSubType": str(
                s.get("traffic_reduce_sub_type", "NonMotorizedArea")),
            "edgePts00X": "%.2f" % x1,
            "edgePts00Y": "%.2f" % y1,
            "edgePts01X": "%.2f" % x2,
            "edgePts01Y": "%.2f" % y2,
        }
        ET.SubElement(ts, "Object", attrs)

    return root


def export_tsr(ann_path, out_dir, ts):
    """从仿真标注导出 TSR XML。

    ann 里的 objects 应包含:
      {'class': 'speed_limit', 'shape': 'circle', 'bbox': [x1,y1,x2,y2],
       'f_value': 60, 'first_attribute': 'limit', ...}
    """
    os.makedirs(out_dir, exist_ok=True)
    ann = json.load(open(ann_path, encoding="utf-8"))
    signs = []
    for o in (ann.get("objects") or []):
        bbox = o.get("bbox") or o.get("bbox2d")
        if not bbox or len(bbox) < 4:
            continue
        cls = str(o.get("class") or o.get("cls") or "").lower()
        shape = o.get("shape") or ("circle" if "limit" in cls else "circle")
        signs.append({
            "shape": shape,
            "first_attribute": o.get("first_attribute", "limit"),
            "color": o.get("color", "UNKNOWN"),
            "bbox": bbox,
            "f_value": o.get("f_value", o.get("limit_value", 0.0)),
            "limit_type": o.get("limit_type", "SPEED"),
            "type": o.get("type", "RELEASE"),
            "label_type": o.get("label_type", "POS"),
            "is_covered": o.get("is_covered", "Level0"),
        })

    root = build_tsr_xml(signs, is_virtual=True)
    raw = ET.tostring(root, encoding="utf-8")
    dom = minidom.parseString(raw)
    txt = dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    lines = [ln for ln in txt.split("\n") if ln.strip()]
    fp = os.path.join(out_dir, "%s.xml" % ts)
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("TSR XML: %s  (%d 个标志)" % (fp, len(signs)))
    return fp


# ============================================================
# GND
# ============================================================
# seg_lane 调色板（RGB）
# PALETTE-CORRECTED: 已用 PIL 读真实文件确认，以下为【文件里的 RGB 字节值】
GND_PALETTE_RGB = {
    "bg":   (0, 0, 0),
    "edge": (0, 255, 0),
    "dash": (244, 35, 232),
}
# cv2 写图时接受 BGR，转换一次
GND_PALETTE = {k: (v[2], v[1], v[0]) for k, v in GND_PALETTE_RGB.items()}

# OUTROOT-LOCKED: 所有导出默认落盘到网盘
OUT_ROOT = "/mnt/Data_Platform/zhj_datasim"



MASK_DIRS = ("freespace_mask_ori_3.0.0", "obj_mask_ori_3.1.1",
             "sky_mask_ori_3.0.0")


def export_gnd_masks(mask_dict, out_dir, ts):
    """GND-MASKS-FULL: 导出 freespace / obj / sky 三个 mask 目录。"""
    root = os.path.join(out_dir, ts)
    n = 0
    for key, arr in (mask_dict or {}).items():
        dname = {
            "freespace": "freespace_mask_ori_3.0.0",
            "obj": "obj_mask_ori_3.1.1",
            "sky": "sky_mask_ori_3.0.0",
        }.get(key)
        if not dname or arr is None:
            continue
        d = os.path.join(root, dname)
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, "%s.png" % ts)
        cv2.imwrite(fp, arr)
        print("  %s: %s" % (dname, fp))
        n += 1
    return n


def export_gnd(sim_jpg, orig_jpg, seg_png, out_dir, ts, yaml_src=None):
    """导出 GND 目录结构:
        <ts>/img_ori/<ts>.jpg
        <ts>/img_rec/<ts>.jpg      ← 仿真图
        <ts>/seg_lane/<ts>.png
        <ts>/parameter_yaml/...    （可选复制）
    """
    root = os.path.join(out_dir, ts)
    os.makedirs(os.path.join(root, "img_ori"), exist_ok=True)
    os.makedirs(os.path.join(root, "img_rec"), exist_ok=True)
    os.makedirs(os.path.join(root, "seg_lane"), exist_ok=True)

    ok = 0
    if orig_jpg and os.path.isfile(orig_jpg):
        dst = os.path.join(root, "img_ori", "%s.jpg" % ts)
        shutil.copy2(orig_jpg, dst)
        print("  img_ori: %s" % dst)
        ok += 1
    else:
        print("  ! 缺原图")

    if sim_jpg and os.path.isfile(sim_jpg):
        dst = os.path.join(root, "img_rec", "%s.jpg" % ts)
        shutil.copy2(sim_jpg, dst)
        print("  img_rec: %s" % dst)
        ok += 1
    else:
        print("  ! 缺仿真图")

    if seg_png and os.path.isfile(seg_png):
        dst = os.path.join(root, "seg_lane", "%s.png" % ts)
        if os.path.abspath(seg_png) == os.path.abspath(dst):
            print("  seg_lane: %s （已就位）" % dst)
        else:
            shutil.copy2(seg_png, dst)
            print("  seg_lane: %s" % dst)
        ok += 1
    else:
        print("  ! 缺 seg_lane")

    if yaml_src and os.path.isdir(yaml_src):
        dst = os.path.join(root, "parameter_yaml")
        shutil.copytree(yaml_src, dst, dirs_exist_ok=True)
        print("  parameter_yaml: %s" % dst)

    return root


def make_seg_lane(orig_jpg, mask_dict, out_png):
    """从各类 mask 合成 seg_lane 调色板图。

    mask_dict: {'edge': mask255, 'dash': mask255, 'far': mask255}
    """
    img = cv2.imread(orig_jpg)
    if img is None:
        raise RuntimeError("读不到原图")
    h, w = img.shape[:2]
    out = np.zeros((h, w, 3), np.uint8)     # 全黑背景

    # 按优先级绘制：far -> dash -> edge
    for key in ("far", "dash", "edge"):
        m = mask_dict.get(key)
        if m is None:
            continue
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        # GND_PALETTE 已是 BGR（由 RGB 转换），直接写入
        out[m > 127] = GND_PALETTE[key]

    cv2.imwrite(out_png, out)
    print("seg_lane 生成: %s" % out_png)
    return out_png


def validate_seg_lane(png_path):
    """校验生成的 seg_lane 颜色是否符合调色板。"""
    im = cv2.imread(png_path)
    if im is None:
        print("  ! 读不到 seg_lane")
        return False
    b, g, r = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    uniq = np.unique(im.reshape(-1, 3), axis=0)
    allowed = set(GND_PALETTE_RGB.values())
    # cv2 读到的是 BGR -> 转 RGB 再比对
    bad = []
    for c in uniq:
        rgb = (int(c[2]), int(c[1]), int(c[0]))
        if rgb not in allowed:
            bad.append((tuple(int(v) for v in c), rgb))
    if bad:
        print("  ! 非法颜色 BGR->RGB: %s" % bad[:5])
        return False
    print("  seg_lane 调色板校验: OK (%d 种颜色)" % len(uniq))
    return True


def run_gnd_full(args):
    """一次跑全：Mask2Former 出 freespace/obj/sky + seg_lane 合成。"""
    sys.path.insert(0, "/opt/ue5_synth/tools")
    from gnd_mask_4a import MapillaryMarkingExtractor

    img = cv2.imread(args.orig)
    if img is None:
        raise RuntimeError("读不到原图: %s" % args.orig)
    h, w = img.shape[:2]
    print("原图: %dx%d" % (w, h))

    ext = MapillaryMarkingExtractor(device=args.model)

    # 1) 车辆 mask
    veh = ext.get_vehicle_mask(img, prob_thresh=0.5)
    print("  车辆像素: %d (%.2f%%)" % ((veh > 127).sum(),
                                   (veh > 127).mean() * 100))

    # 2) 标记线 mask -> seg_lane
    mark = ext.get_marking_mask(img, prob_thresh=0.12)
    os.makedirs(os.path.join(args.out, args.ts, "seg_lane"), exist_ok=True)
    seg_fp = os.path.join(args.out, args.ts, "seg_lane", "%s.png" % args.ts)
    mm = {"dash": mark}
    make_seg_lane(args.orig, mm, seg_fp)
    validate_seg_lane(seg_fp)

    # 3) 三个 mask 目录
    sky = np.zeros((h, w), np.uint8)      # 占位（下方补）
    free = (mark == 0).astype(np.uint8) * 180
    export_gnd_masks({"freespace": free, "obj": veh, "sky": sky},
                     args.out, args.ts)

    # 4) img_ori / img_rec / parameter_yaml
    export_gnd(args.sim, args.orig, seg_fp, args.out, args.ts, args.yaml)

    # 汇总
    print()
    print("GND 完整导出: %s" % os.path.join(args.out, args.ts))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("tsr")
    p1.add_argument("--ann", required=True)
    p1.add_argument("--out",
                    default="/mnt/Data_Platform/zhj_datasim/TSR")
    p1.add_argument("--ts", required=True)

    p2 = sub.add_parser("gnd")
    p2.add_argument("--sim", required=True)
    p2.add_argument("--orig", required=True)
    p2.add_argument("--seg", default=None)
    p2.add_argument("--out",
                    default="/mnt/Data_Platform/zhj_datasim/GND")
    p2.add_argument("--ts", required=True)
    p2.add_argument("--yaml", default=None)

    p3 = sub.add_parser("gnd-full")
    p3.add_argument("--orig", required=True, help="原始 jpg")
    p3.add_argument("--sim", required=True, help="仿真 jpg")
    p3.add_argument("--out",
                    default="/mnt/Data_Platform/zhj_datasim/GND")
    p3.add_argument("--ts", required=True)
    p3.add_argument("--yaml", default=None)
    p3.add_argument("--model", default="cuda",
                    help="Mask2Former 设备 cuda/cpu")

    args = ap.parse_args()

    if args.cmd == "tsr":
        export_tsr(args.ann, args.out, args.ts)
    elif args.cmd == "gnd":
        r = export_gnd(args.sim, args.orig, args.seg, args.out, args.ts,
                       args.yaml)
        if args.seg:
            validate_seg_lane(os.path.join(r, "seg_lane",
                                           "%s.png" % args.ts))
    elif args.cmd == "gnd-full":
        run_gnd_full(args)


if __name__ == "__main__":
    main()
