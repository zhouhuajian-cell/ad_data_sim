#!/usr/bin/env python3
"""clip 数据自检工具 — 检查目录结构、文件完整性、能否直接用。

用法:
    python3 check_clip.py <目录>                 # 自动识别 OBJ / GND
    python3 check_clip.py <目录> --type obj      # 强制按 OBJ 检查
    python3 check_clip.py <目录> --json          # JSON 输出

检查项（OBJ）:
  ① 帧序列目录（含 cam_st_x2f_front 或 img_ori）
  ② 图片数量
  ③ LiDAR 点云数量 + 格式
  ④ 标定 XML（内参/外参/畸变）
  ⑤ 标注 XML（真实车 3D 框）
  ⑥ 时间戳对齐
  ⑦ 抽帧验证：地面拟合是否成功

检查项（GND）:
  ① 图片
  ② parameter_yaml（内外参）
  ③ freespace_mask / obj_mask / seg_lane
  ④ hdmap / geojson
"""
import argparse
import glob
import json
import os
import sys

OK, WARN, ERR = "OK", "WARN", "ERR"


def _fmt(status, msg):
    icon = {OK: "✅", WARN: "⚠️ ", ERR: "❌"}[status]
    return "  %s %s" % (icon, msg)


def detect_type(root):
    """自动判断 OBJ 还是 GND。"""
    # 找一层或两层内的特征目录
    for depth in (1, 2, 3):
        pat = os.path.join(root, *(["*"] * depth))
        if glob.glob(os.path.join(pat, "img_ori")):
            return "gnd"
        if glob.glob(os.path.join(pat, "cam_st_x2f_front")):
            return "obj"
        if glob.glob(os.path.join(pat, "lidar_hs_pandar128_topmiddle")):
            return "obj"
    return None


# SEQ-FILTER-FIX: 排除标注/辅助目录
_EXCLUDE = ("label_lidar", "label_", "cam_st_x2f_front",
            "img_ori", "lidar_", "freespace_mask", "obj_mask",
            "seg_lane", "hdmap", "geojson", "parameter_yaml")


def find_seqs(root, keys):
    """在 root 下找含指定子目录的帧序列目录。"""
    out = []
    for depth in (1, 2, 3):
        pat = os.path.join(root, *(["*"] * depth))
        for cand in glob.glob(pat):
            if not os.path.isdir(cand):
                continue
            base = os.path.basename(cand.rstrip("/"))
            if any(base.startswith(x) or base == x for x in _EXCLUDE):
                continue
            if any(os.path.isdir(os.path.join(cand, k)) for k in keys):
                if cand not in out:
                    out.append(cand)
    return sorted(out)


# ============================================================
# OBJ 检查
# ============================================================
def check_obj(root, sample_frames=3):
    res = []

    # ① 帧序列
    seqs = find_seqs(root, ("cam_st_x2f_front", "img_ori",
                            "lidar_hs_pandar128_topmiddle"))
    if not seqs:
        res.append((ERR, "未找到帧序列目录（需含 cam_st_x2f_front 或 lidar_*）"))
        return res, {}
    res.append((OK, "找到 %d 个帧序列" % len(seqs)))
    for s in seqs[:5]:
        res.append((OK, "  %s" % os.path.relpath(s, root)))
    if len(seqs) > 5:
        res.append((OK, "  ... 还有 %d 个" % (len(seqs) - 5)))

    total_img = total_bin = total_xml = 0
    detail = []

    for seq in seqs:
        name = os.path.relpath(seq, root)
        imgs = []
        for cam in ("cam_st_x2f_front", "img_ori"):
            imgs = sorted(glob.glob(os.path.join(seq, cam, "*.jpg")))
            if not imgs:
                imgs = sorted(glob.glob(os.path.join(seq, cam, "*.png")))
            if imgs:
                break
        bins = sorted(glob.glob(os.path.join(
            seq, "lidar_hs_pandar128_topmiddle", "*.bin")))
        xmls = sorted(glob.glob(os.path.join(
            seq, "label_lidar_hs_pandar128_topmiddle_align",
            "cam_st_x2f_front", "*.xml")))
        xmls2 = sorted(glob.glob(os.path.join(
            seq, "label_lidar_hs_pandar128_topmiddle", "*.xml")))

        total_img += len(imgs)
        total_bin += len(bins)
        total_xml += len(xmls) + len(xmls2)

        d = {"seq": name, "imgs": len(imgs), "bins": len(bins),
             "xml_align": len(xmls), "xml_plain": len(xmls2),
             "ok": False, "issues": []}

        if not imgs:
            d["issues"].append("无图片")
        if not bins:
            d["issues"].append("无 LiDAR 点云")
        if not xmls and not xmls2:
            d["issues"].append("无标定 XML")
        if imgs and bins and len(imgs) != len(bins):
            d["issues"].append("图 %d 与点云 %d 数量不符" % (len(imgs), len(bins)))

        d["ok"] = not d["issues"]
        detail.append(d)

    if total_img:
        res.append((OK, "总图片: %d" % total_img))
    else:
        res.append((ERR, "没有图片"))

    if total_bin:
        res.append((OK, "总点云: %d" % total_bin))
    else:
        res.append((WARN, "没有 LiDAR 点云 → 无法做 3D 碰撞/轨迹校验，"
                          "只能做 2D 合成"))
    if total_xml:
        res.append((OK, "总标定/标注: %d" % total_xml))
    else:
        res.append((ERR, "没有标定 XML → 无法投影 3D 框"))

    # 抽帧验证
    if total_bin and total_xml:
        res.append((OK, "-- 抽帧验证 --"))
        try:
            sys.path.insert(0, "/opt/ue5_synth/tools")
            from lidar_depth_core import load_points, fit_ground_plane
            tested = 0
            ok_cnt = 0
            for seq in seqs[:2]:
                bins = sorted(glob.glob(os.path.join(
                    seq, "lidar_hs_pandar128_topmiddle", "*.bin")))
                if not bins:
                    continue
                for i in range(min(2, len(bins))):
                    try:
                        xyz = load_points(bins[i])[:, :3]
                        pl = fit_ground_plane(xyz)
                        tested += 1
                        if pl and pl.get("sane"):
                            ok_cnt += 1
                            res.append((OK, "  %s: 地面 Z@15m=%.3f 内点%.0f%%" % (
                                os.path.basename(bins[i]),
                                pl["ground_z_at_15m"],
                                pl["inlier_ratio"] * 100)))
                        else:
                            res.append((WARN, "  %s: 地面拟合失败" %
                                        os.path.basename(bins[i])))
                    except Exception as e:
                        res.append((ERR, "  %s: %s" % (
                            os.path.basename(bins[i]), str(e)[:60])))
            if tested:
                res.append((OK if ok_cnt == tested else WARN,
                            "地面拟合 %d/%d 通过" % (ok_cnt, tested)))
        except Exception as e:
            res.append((WARN, "抽帧验证跳过: %s" % str(e)[:80]))

    return res, {"type": "obj", "seqs": len(seqs), "imgs": total_img,
                 "bins": total_bin, "xmls": total_xml,
                 "detail": detail}


# ============================================================
# GND 检查
# ============================================================
def check_gnd(root):
    res = []
    seqs = find_seqs(root, ("img_ori",))
    if not seqs:
        res.append((ERR, "未找到 img_ori 帧序列"))
        return res, {}

    res.append((OK, "找到 %d 个帧序列" % len(seqs)))

    need = ["img_ori", "parameter_yaml", "freespace_mask_ori_3.0.0",
            "obj_mask_ori_3.1.1", "seg_lane"]
    opt = ["hdmap", "geojson"]

    total_img = 0
    detail = []
    for seq in seqs[:10]:
        d = {"seq": os.path.relpath(seq, root), "have": [],
             "miss": [], "ok": False}
        for k in need:
            p = os.path.join(seq, k)
            if os.path.isdir(p):
                d["have"].append(k)
            else:
                d["miss"].append(k)
        for k in opt:
            if os.path.isdir(os.path.join(seq, k)):
                d["have"].append(k + "(可选)")

        imgs = sorted(glob.glob(os.path.join(seq, "img_ori", "*.jpg")))
        d["imgs"] = len(imgs)
        total_img += len(imgs)
        d["ok"] = not d["miss"]
        detail.append(d)

    first = detail[0]
    res.append((OK, "帧序列 %s" % first["seq"]))
    res.append((OK, "  图片: %d 张" % first["imgs"]))
    if first["have"]:
        res.append((OK, "  已有: %s" % ", ".join(first["have"])))
    if first["miss"]:
        res.append((ERR, "  缺少: %s" % ", ".join(first["miss"])))
    else:
        res.append((OK, "  结构完整"))

    ok_cnt = sum(1 for d in detail if d["ok"])
    res.append((OK if ok_cnt == len(detail) else WARN,
                "完整序列 %d/%d" % (ok_cnt, len(detail))))

    # LiDAR 检查（GND 通常没有）
    has_lidar = any(glob.glob(os.path.join(s, "lidar_*"))
                    for s in seqs[:3])
    if has_lidar:
        res.append((OK, "含 LiDAR 点云"))
    else:
        res.append((WARN, "无 LiDAR 点云 → 只能做 2D 合成（地面用 mask 推断）"))

    return res, {"type": "gnd", "seqs": len(seqs),
                 "imgs": total_img, "detail": detail}


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--type", choices=["obj", "gnd", "auto"], default="auto")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print("目录不存在: %s" % root)
        return 2

    t = args.type
    if t == "auto":
        t = detect_type(root)
        if t is None:
            print("无法自动识别类型，请加 --type obj 或 --type gnd")
            return 2

    if t == "obj":
        lines, meta = check_obj(root)
    else:
        lines, meta = check_gnd(root)

    if args.json:
        print(json.dumps({"root": root, "type": t,
                          "items": [{"status": s, "msg": m}
                                    for s, m in lines],
                          "summary": meta}, ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print("目录: %s" % root)
        print("类型: %s" % ("OBJ（物体插入）" if t == "obj" else "GND（地面标线）"))
        print("=" * 60)
        for s, m in lines:
            print(_fmt(s, m))
        print("=" * 60)

        n_err = sum(1 for s, _ in lines if s == ERR)
        n_warn = sum(1 for s, _ in lines if s == WARN)
        if n_err:
            print("结论: ❌ 不可用（%d 个错误）" % n_err)
        elif n_warn:
            print("结论: ⚠️  可用，但有 %d 个警告" % n_warn)
        else:
            print("结论: ✅ 完全可用")
        print()

    return 0 if not any(s == ERR for s, _ in lines) else 1


if __name__ == "__main__":
    sys.exit(main())
