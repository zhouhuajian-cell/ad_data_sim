#!/usr/bin/env python3
"""多车型批量渲染（canonical 3D + 真实相机）。

用法:
    python batch_canonical.py --clip <clip> --frame <fr> --n 10
    python batch_canonical.py --batch <clip_name> --count 10
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

CLIPS_ROOT = "/data/test_data/real_clips"
CANON_DIR = "/data/assets_global/OBJ_3D/canonical"
INDEX = "/data/assets_global/OBJ_3D/asset_index_obj3d.json"
BLENDER = "/data/blender-4.2.23-linux-x64/blender"
RENDER_PY = "/opt/ue5_synth/tools/render_canonical_one.py"
OUT_ROOT = "/mnt/Data_Platform/zhj_datasim/OBJ_batch"

# 采样权重：超跑降权（用户要求 <5%）
SPORT_KEYWORDS = ["4e99a536", "81a50236", "9633b90f", "1cc7517b",
                  "7c5c1410", "5678d407", "1a795645", "6d764524",
                  "700d0a62", "1f714b68", "9409145e", "16d097f5",
                  "16071357", "110a2cb6", "74faa2b7", "6d0c5063",
                  "9d01d048", "a1d9f096", "29b8ce20"]
SPORT_WEIGHT = 0.04


def rodrigues(rv):
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def zero_roll(R_in):
    """消掉相机 roll，只保留 pitch/yaw，转为 OpenCV 相机系 T_w2c。"""
    fwd = R_in[2, :]          # 相机前向（世界系）
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    up0 = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up0)
    n = np.linalg.norm(right)
    if n < 1e-6:
        right = R_in[0, :]
    else:
        right = right / n
    up = np.cross(right, fwd)
    R_new = np.vstack([right, -up, fwd])   # OpenCV: +Y down
    return R_new


def parse_calib(xml_path):
    root = ET.parse(xml_path).getroot()
    cam = root.find(".//cam_st_x2f_front")
    iv = [float(cam.find("intrinsic_matrix%d" % i).text)
          for i in range(9)]
    K = np.array(iv).reshape(3, 3)
    rv = [float(cam.find("rotation_vector%d" % i).text)
          for i in range(3)]
    tv = np.array([float(cam.find("translation_vector%d" % i).text)
                   for i in range(3)])
    R_in = rodrigues(rv)
    R = zero_roll(R_in)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tv
    return {"fx": K[0, 0], "fy": K[1, 1],
            "cx": K[0, 2], "cy": K[1, 2],
            "W": 1920, "H": 1080}, T.tolist()


def load_index():
    with open(INDEX, encoding="utf-8") as f:
        return json.load(f)["assets"]


def weight_of(uid):
    return SPORT_WEIGHT if any(k in uid for k in SPORT_KEYWORDS) else 1.0


# AVOID-RELAX: 只避让近距离真实车（远处可部分重叠，视觉上不冲突）
MIN_GAP_LONG = 2.0
MIN_GAP_LAT = 1.6
AVOID_DIST = 20.0      # 只避让 20m 内的真实车


def _is_free(x, y, boxes, own_l=4.6):
    """AVOID-REAL: 与近距离真实车辆做碰撞检查。"""
    for (bx, by, bl, bw) in boxes:
        # 只看 20m 内的（远处车小、重叠不明显）
        if abs(bx) > AVOID_DIST:
            continue
        dl = abs(x - bx) - (own_l + bl) / 2.0
        dw = abs(y - by) - (2.0 + bw) / 2.0
        if dl < MIN_GAP_LONG and dw < MIN_GAP_LAT:
            return False
    return True


def sample_vehicles(assets, n, rng, avoid=None):
    """按权重采样 N 辆车，位置在 [12,30]m × ±3.5m，避让真实车辆。"""
    w = np.array([weight_of(a["uid"]) for a in assets], dtype=float)
    w = w / w.sum()
    idxs = np.random.RandomState(rng.randint(0, 99999)).choice(
        len(assets), size=min(n, len(assets)), replace=False, p=w)
    boxes = list(avoid or [])
    out = []
    for i in idxs:
        a = assets[i]
        L = (a.get("lwh") or [4.6])[0]
        placed = False
        for _ in range(40):
            x = rng.uniform(12.0, 30.0)
            y = rng.uniform(-5.0, 5.0)
            if not _is_free(x, y, boxes, L):
                continue
            yaw = rng.choice([0.0, 180.0]) + rng.uniform(-8, 8)
            out.append({"uid": a["uid"], "x": round(x, 2),
                        "y": round(y, 2), "yaw": round(yaw, 1),
                        "class": a["true_class"]})
            boxes.append((x, y, L, 2.0))
            placed = True
            break
        if not placed:
            continue
    out.sort(key=lambda v: -v["x"])
    return out


def composite(bg_path, fg_path, out_path):
    bg = cv2.imread(bg_path)
    fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
    if fg is None or bg is None:
        return False
    if fg.shape[2] == 3:
        cv2.imwrite(out_path, fg, [cv2.IMWRITE_JPEG_QUALITY, 94])
        return True
    h, w = bg.shape[:2]
    if fg.shape[:2] != (h, w):
        fg = cv2.resize(fg, (w, h))
    a = fg[:, :, 3:4].astype(np.float32) / 255.0
    comp = (fg[:, :, :3].astype(np.float32) * a +
            bg.astype(np.float32) * (1 - a))
    cv2.imwrite(out_path, np.clip(comp, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 94])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="20261120_HS5001_changchun_1v_closecar")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--per-frame", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--out", default=OUT_ROOT)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    assets = load_index()
    print("可用资产: %d" % len(assets))

    clip_dir = os.path.join(CLIPS_ROOT, args.batch)
    frames = sorted([d for d in os.listdir(clip_dir)
                     if os.path.isdir(os.path.join(clip_dir, d))])
    print("clip %s: %d 帧目录" % (args.batch, len(frames)))
    if not frames:
        return

    out_dir = os.path.join(args.out, args.batch)
    os.makedirs(out_dir, exist_ok=True)

    n_done = 0
    for fi in range(min(args.count, len(frames))):
        fr = frames[fi]
        d = os.path.join(clip_dir, fr)
        imgs = sorted(glob.glob(os.path.join(d, "cam_st_x2f_front",
                                             "*.jpg")))
        if not imgs:
            continue
        img = imgs[0]
        stem = os.path.splitext(os.path.basename(img))[0]
        # 相机标定
        xmls = glob.glob(os.path.join(
            d, "label_lidar_hs_pandar128_topmiddle_align",
            "cam_st_x2f_front", "*.xml"))
        if not xmls:
            xmls = glob.glob(os.path.join(
                d, "label_lidar_hs_pandar128_topmiddle", "*.xml"))
        if not xmls:
            print("  [%d] no calib" % fi)
            continue
        K, T = parse_calib(xmls[0])

        # AVOID-REAL: 读取真实车辆标注，采样时避让
        real_boxes = []
        try:
            lab_xml = sorted(glob.glob(os.path.join(
                d, "label_lidar_hs_pandar128_topmiddle", "*.xml")))
            if lab_xml:
                rt = ET.parse(lab_xml[0]).getroot()
                for obj in rt.findall(".//Vehicle/Object"):
                    def gv(t, dv=0.0):
                        e = obj.find(t)
                        return float(e.text) if e is not None and e.text else dv
                    real_boxes.append((gv("objWx"), gv("objWy"),
                                       gv("objLength"), gv("objWidth")))
                print("    真实车辆 %d 辆: %s" % (
                    len(real_boxes),
                    ["(%.1f,%.1f)" % (b[0], b[1]) for b in real_boxes[:4]]))
        except Exception as ex:
            print("    读标注失败:", str(ex)[:50])

        veh = sample_vehicles(assets, args.per_frame, rng,
                              avoid=real_boxes)
        cfg = {"bg": img, "K": K, "T": T, "vehicles": veh,
               "out": os.path.join(out_dir, "_tmp_fg_%s.png" % stem),
               "seed": args.seed + fi}
        cfp = "/tmp/_canon_cfg_%s.json" % stem
        with open(cfp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)

        r = subprocess.run([BLENDER, "--background", "--python",
                            RENDER_PY, "--", cfp],
                           capture_output=True, text=True, timeout=900)
        ok = os.path.isfile(cfg["out"])
        if not ok:
            tail = (r.stdout or "")[-300:]
            print("  [%d] render FAIL %s" % (fi, tail[-120:]))
            continue

        final = os.path.join(out_dir, "%s_obj.jpg" % stem)
        composite(img, cfg["out"], final)
        try:
            os.remove(cfg["out"])
        except Exception:
            pass
        n_done += 1
        print("  [%2d/%d] %s  %d 辆 %s" % (
            n_done, args.count, stem, len(veh),
            ",".join(v["class"] for v in veh)))

    print()
    print("完成 %d 帧 -> %s" % (n_done, out_dir))


if __name__ == "__main__":
    main()
