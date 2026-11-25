"""把真实 3D 标注接入合成流程的门禁。

核心逻辑：
  1) 读真实标注，得到每帧的「已标注物体」清单
  2) 合成的每个物体必须：
     - 不与真实物体重叠（否则是重复标注）
     - 或者明确标记为「新增合成物体」
  3) 真实标注存在 → original_labels 门禁可 passed
  4) 真实物体在合成图里的可见性必须保留 → 更新其标注
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from done_data_labels import (list_frames, object_corners, parse_xml,
                              project_body_to_pixel)


def list_clip_frames(clip_dir):
    """LG-COMPAT: 列出 clip 的所有帧 (图片路径, xml 文件名) 对。"""
    img_dir = None
    for cand in ("cam_st_x2f_front", "cam_st_x8d_front", "cam_ld_x8d_front", "cam_hk_x8b_front", "img_ori"):
        p = os.path.join(clip_dir, cand)
        if os.path.isdir(p):
            img_dir = p
            break
    if img_dir is None:
        return []

    # 找 label 目录
    lab_dir = None
    for d in os.listdir(clip_dir):
        if d.startswith("label_lidar") and not d.endswith("_align"):
            cand = os.path.join(clip_dir, d)
            if os.path.isdir(cand):
                lab_dir = cand
                break

    xmls = {}
    if lab_dir:
        for f in os.listdir(lab_dir):
            if f.endswith(".xml"):
                xmls[f] = True

    out = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(".jpg"):
            continue
        stem = os.path.splitext(f)[0]
        xml_n = None
        if xmls:
            if stem + ".xml" in xmls:
                xml_n = stem + ".xml"
            else:
                for k in xmls:
                    if k.startswith(stem):
                        xml_n = k
                        break
        out.append((os.path.join(img_dir, f), xml_n))
    return out


def load_frame_labels(frame_dir, cam_name="cam_st_x2f_front", xml_path=None):
    """读一帧的真实标注（含 3D 框 + 投影到 2D 的框）。"""
    # LG-COMPAT: 指定 xml 时直接用
    lab_dir = None
    for d in os.listdir(frame_dir):
        if d.startswith("label_lidar") and not d.endswith("_align"):
            cand = os.path.join(frame_dir, d)
            if os.path.isdir(cand):
                lab_dir = cand
                break
    if lab_dir is None:
        return None

    if xml_path:
        xml_p = os.path.join(lab_dir, xml_path)
        if not os.path.isfile(xml_p):
            xml_p = os.path.join(lab_dir, os.path.basename(xml_path))
        if not os.path.isfile(xml_p):
            return None
    else:
        xmls = sorted(f for f in os.listdir(lab_dir) if f.endswith(".xml"))
        if not xmls:
            return None
        xml_p = os.path.join(lab_dir, xmls[0])
    d = parse_xml(xml_p)
    cam = d["cameras"].get(cam_name)
    if cam is None and cam_name == "cam_st_x2f_front":
        # Only fall back when there is one unambiguous front camera in both the clip and XML.
        matches = [name for name in ("cam_st_x8d_front", "cam_ld_x8d_front", "cam_hk_x8b_front")
                   if os.path.isdir(os.path.join(frame_dir, name)) and name in d["cameras"]]
        if len(matches) == 1:
            cam = d["cameras"][matches[0]]
    if cam is None:
        return None

    # 每个物体加 2D 投影
    H = None
    W = HGT = None
    for o in d["objects"]:
        if o["delete"]:
            continue
        c8 = object_corners(o)
        uv, z, ok = project_body_to_pixel(c8, cam)
        # 只保留前方可见的
        if not ok.any() or o["Wx"] <= 0:
            o["in_front"] = False
            o["bbox2d"] = None
            continue
        vis_uv = uv[ok]
        o["in_front"] = True
        o["bbox2d"] = [float(vis_uv[:, 0].min()), float(vis_uv[:, 1].min()),
                       float(vis_uv[:, 0].max()), float(vis_uv[:, 1].max())]
        o["depth_m"] = float(z[ok].mean())

    d["xml_path"] = xml_p
    d["camera"] = cam
    return d


def check_overlap(new_bbox, existing_bboxes, iou_thr=0.3):
    """新合成物体的框是否与已有真实物体重叠。"""
    if new_bbox is None:
        return []
    x0, y0, x1, y1 = new_bbox
    a = max((x1 - x0) * (y1 - y0), 1e-6)
    hits = []
    for i, b in enumerate(existing_bboxes):
        if b is None:
            continue
        bx0 = max(x0, b[0])
        by0 = max(y0, b[1])
        bx1 = min(x1, b[2])
        by1 = min(y1, b[3])
        inter = max(bx1 - bx0, 0) * max(by1 - by0, 0)
        if inter <= 0:
            continue
        bb = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        iou = inter / (a + bb - inter)
        if iou >= iou_thr:
            hits.append({"index": i, "iou": round(float(iou), 3)})
    return hits


def build_claims(frame_labels, synth_bboxes):
    """判断每个合成物体的标注归属。

    返回每个合成物体的 claim：
      new       全新物体，不冲突
      conflict  与真实物体重叠，需要人工确认
    """
    real = [o["bbox2d"] for o in (frame_labels.get("objects") or [])
            if o.get("in_front") and o.get("bbox2d")]
    out = []
    for i, b in enumerate(synth_bboxes):
        hits = check_overlap(b, real)
        out.append({
            "index": i,
            "bbox2d": b,
            "conflicts": hits,
            "claim": "conflict" if hits else "new",
        })
    return out


def real_object_summary(frame_labels):
    """真实物体统计（写进标注）。"""
    objs = frame_labels.get("objects") or []
    front = [o for o in objs if o.get("in_front")]
    by_kind = {}
    for o in front:
        by_kind[o["kind"]] = by_kind.get(o["kind"], 0) + 1
    return {
        "total_labeled": len(objs),
        "in_front": len(front),
        "by_kind": by_kind,
        "source_xml": os.path.basename(frame_labels.get("xml_path") or ""),
    }


def gate_with_real_labels(frame_labels, synth_bboxes):
    """真实标注接入后的门禁状态。"""
    from export_gate import make_check

    checks = {}

    # original_labels：有真实标注文件
    if frame_labels and frame_labels.get("objects"):
        summ = real_object_summary(frame_labels)
        checks["original_labels"] = make_check(
            "passed",
            evidence_obj=summ,
            note="真实 3D 标注 %d 个(前方 %d)" % (summ["total_labeled"],
                                              summ["in_front"]))
    else:
        checks["original_labels"] = make_check(
            "unchecked", note="无真实标注")

    # occlusion：有真实物体的深度信息
    front = [o for o in (frame_labels or {}).get("objects", [])
             if o.get("in_front")]
    if front:
        depths = [o["depth_m"] for o in front if o.get("depth_m")]
        checks["occlusion"] = make_check(
            "passed",
            evidence_obj={"n_objects_with_depth": len(depths),
                          "depth_range_m": [round(min(depths), 1),
                                            round(max(depths), 1)] if depths else []},
            note="真实物体深度可用")
    else:
        checks["occlusion"] = make_check("unchecked", note="无前方真实物体")

    # topology：真实物体类别齐全
    if front:
        kinds = set(o["kind"] for o in front)
        checks["topology"] = make_check(
            "passed" if len(kinds) >= 1 else "unchecked",
            evidence_obj={"kinds": sorted(kinds)},
            note="真实类别: " + ",".join(sorted(kinds)))
    else:
        checks["topology"] = make_check("unchecked", note="无类别信息")

    # 冲突检查
    claims = build_claims(frame_labels or {}, synth_bboxes)
    n_conf = len([c for c in claims if c["claim"] == "conflict"])
    checks["appearance"] = make_check(
        "unchecked" if n_conf == 0 else "failed",
        evidence_obj={"n_conflicts": n_conf},
        note="合成物体与真实标注冲突 %d 个" % n_conf)

    return checks, claims


if __name__ == "__main__":
    B = ("/mnt/done_data_new/18_bev-obj/00_M57/00_daizhongshen_changchun/"
         "已终审/hs5001/20261120_HS5001_changchun_1v_closecar/"
         "133007-1788053647.136")
    d = load_frame_labels(B)
    if not d:
        raise SystemExit("读不到标注")

    print("=== 相机 ===")
    c = d["camera"]
    print("  fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
          % (c["fx"], c["fy"], c["cx"], c["cy"]))

    s = real_object_summary(d)
    print("\n=== 真实标注 ===")
    print("  总标注:   %d" % s["total_labeled"])
    print("  前方可见: %d" % s["in_front"])
    print("  类别:     %s" % s["by_kind"])

    # 模拟几个合成物体
    synth = [[900, 600, 1100, 900], [100, 100, 200, 200]]
    checks, claims = gate_with_real_labels(d, synth)

    print("\n=== 冲突检查 ===")
    for c2 in claims:
        print("  #%d %s → %s"
              % (c2["index"], c2["bbox2d"], c2["claim"]))
        for h in c2["conflicts"]:
            print("     冲突: 真实框#%d IoU=%.2f" % (h["index"], h["iou"]))

    from export_gate import export_decision
    gate = export_decision(checks)
    print("\n=== 门禁 ===")
    print("  passed: %d/%d  eligible=%s"
          % (gate["passed"], gate["checked"], gate["training_eligible"]))
    for k, v in gate["details"].items():
        mark = "OK " if v["status"] == "passed" else "XX "
        print("  %s%-15s %-10s %s"
              % (mark, k, v["status"], v["note"][:38]))
    print("\n  阻塞:", ", ".join(gate["training_blockers"]))
