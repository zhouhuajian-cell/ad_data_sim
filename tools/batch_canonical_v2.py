#!/usr/bin/env python3
"""批量渲染核心：A(Mask2Former 落点校验) + B(EXR 深度遮挡) + 阴影修复。

用法:
    python batch_canonical_v2.py --batch <clip> --count 5 --per-frame 3
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

sys.path.insert(0, "/opt/ue5_synth/tools")

CLIPS_ROOT = "/data/test_data/real_clips"
INDEX = "/data/assets_global/OBJ_3D/asset_index_obj3d.json"
BLENDER = "/data/blender-4.2.23-linux-x64/blender"
RENDER_PY = "/opt/ue5_synth/tools/render_canonical_one.py"
OUT_ROOT = "/mnt/Data_Platform/zhj_datasim/OBJ_batch"

SPORT_KEYWORDS = ["4e99a536", "81a50236", "9633b90f", "1cc7517b",
                  "7c5c1410", "5678d407", "1a795645", "6d764524",
                  "700d0a62", "1f714b68", "9409145e", "16d097f5",
                  "16071357", "110a2cb6", "74faa2b7", "6d0c5063",
                  "9d01d048", "a1d9f096", "29b8ce20"]
SPORT_WEIGHT = 0.04
MIN_VISIBLE = 0.30       # B: 可见率卡控


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
    fwd = R_in[2, :]
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    up0 = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up0)
    n = np.linalg.norm(right)
    if n < 1e-6:
        right = R_in[0, :]
    else:
        right = right / n
    up = np.cross(right, fwd)
    return np.vstack([right, -up, fwd])


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
    return ({"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2],
             "cy": K[1, 2], "W": 1920, "H": 1080}, T, R, tv)


def body_to_pixel(K, R, t, pts_body):
    P = np.asarray(pts_body, np.float64).reshape(-1, 3)
    Pc = P @ R.T + t
    z = Pc[:, 2]
    ok = z > 1e-6
    uv = np.full((len(P), 2), np.nan)
    if ok.any():
        uv[ok, 0] = K["cx"] + K["fx"] * Pc[ok, 0] / z[ok]
        uv[ok, 1] = K["cy"] + K["fy"] * Pc[ok, 1] / z[ok]
    return uv, z, ok


def load_index():
    with open(INDEX, encoding="utf-8") as f:
        return json.load(f)["assets"]


def weight_of(uid):
    return SPORT_WEIGHT if any(k in uid for k in SPORT_KEYWORDS) else 1.0


# ---------- A: Mask2Former 落点校验 ----------
class MaskChecker:
    """用 Mask2Former 路面/车辆掩码校验落点合法性。"""

    def __init__(self, img, K, R, t, extractor=None):
        from gnd_mask_4a import MapillaryMarkingExtractor
        self.K, self.R, self.t = K, R, t
        self.ext = extractor or MapillaryMarkingExtractor()
        # 车辆掩码
        veh = self.ext.get_vehicle_mask(img, prob_thresh=0.5) > 127
        self.veh = veh
        # 这里只提供粗粒度的道路视锥门禁。车辆掩码不能从道路候选中扣除，
        # 否则“自然遮挡”会被误当成 3D 穿模；真实物体碰撞交给 3D 门禁。
        h, w = img.shape[:2]
        free = np.zeros((h, w), bool)
        # 下半部分 + 车辆掩码外，粗略取路面
        free[int(h * 0.45):, :] = True
        self.road = free

    def footprint_ok(self, x, y, L, W, yaw_deg, reject_vehicle=True):
        """校验接地矩形；3D 模式下 reject_vehicle=False。"""
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        pts = []
        for dx in (-L / 2, L / 2):
            for dy in (-W / 2, W / 2):
                wx = x + dx * c - dy * s
                wy = y + dx * s + dy * c
                pts.append([wx, wy, 0.0])
        uv, z, ok = body_to_pixel(self.K, self.R, self.t, pts)
        if not ok.all() or not np.isfinite(uv).all():
            return False, None, "behind_camera"

        h, w = self.road.shape
        # pts 的枚举顺序不是轮廓顺序，直接 fillPoly 会得到蝴蝶结自交多边形。
        poly = cv2.convexHull(np.rint(uv).astype(np.int32)).reshape(-1, 2)
        if poly[:, 0].max() < 0 or poly[:, 0].min() >= w:
            return False, None, "out_of_frame"
        if poly[:, 1].max() < 0 or poly[:, 1].min() >= h:
            return False, None, "out_of_frame"

        fm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(fm, [poly], 1)
        n = int(fm.sum())
        if n < 30:
            return False, None, "too_small"

        road_ratio = float((fm.astype(bool) & self.road).sum()) / n
        veh_ovl = float((fm.astype(bool) & self.veh).sum()) / n
        foot = {"n": n, "road": round(road_ratio, 3),
                "veh": round(veh_ovl, 3), "bbox": fm}
        if road_ratio < 0.98:
            return False, foot, "off_road %.2f" % road_ratio
        if reject_vehicle and veh_ovl > 0.0:
            return False, foot, "on_vehicle %.2f" % veh_ovl
        return True, foot, "ok"


def sample_vehicles_v2(assets, n, rng, checker):
    """A: 用 Mask2Former 校验落点。"""
    w = np.array([weight_of(a["uid"]) for a in assets], dtype=float)
    w = w / w.sum()
    rs = np.random.RandomState(rng.randint(0, 99999))
    idxs = rs.choice(len(assets), size=min(len(assets), n * 6),
                     replace=False, p=w)
    out = []
    tried = 0
    for i in idxs:
        if len(out) >= n:
            break
        a = assets[i]
        lwh = a.get("lwh") or [4.6, 1.9, 1.5]
        L, Wd = lwh[0], lwh[1]
        placed = False
        for _ in range(50):
            tried += 1
            x = rng.uniform(10.0, 28.0)
            y = rng.uniform(-4.0, 4.0)
            # YAW-ONEWAY: 同向车道锁定朝向（车尾朝自车）
            #   canonical 模型车头 +X，放到世界坐标系后 yaw=180 让车头朝外
            yaw = 180.0 + rng.uniform(-8.0, 8.0)
            ok, foot, why = checker.footprint_ok(x, y, L, Wd, yaw)
            if ok:
                out.append({"uid": a["uid"], "x": round(x, 2),
                            "y": round(y, 2),
                            "yaw": round(yaw, 1),
                            "class": a["true_class"]})
                placed = True
                break
        if not placed:
            continue
    out.sort(key=lambda v: -v["x"])
    return out, tried


# ---------- 合并 + 阴影修复 ----------
def composite_v2(bg_path, fg_path, depth_npy, out_path, shadow_max=0.35,
                 dark_mode=False):
    """合成 + 阴影修复 + 暗光适配。"""
    bg = cv2.imread(bg_path)
    fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
    if fg is None or bg is None:
        return False, 0.0
    h, w = bg.shape[:2]
    if fg.shape[:2] != (h, w):
        fg = cv2.resize(fg, (w, h))
    if fg.shape[2] == 3:
        cv2.imwrite(out_path, fg, [cv2.IMWRITE_JPEG_QUALITY, 94])
        return True, 1.0

    a = fg[:, :, 3:4].astype(np.float32) / 255.0
    rgb = fg[:, :, :3].astype(np.float32)
    bgf = bg.astype(np.float32)
    # LIGHT-ADAPT: 暗光帧把前景压暗，匹配背景
    if dark_mode:
        _bg_luma = float(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).mean())
        # 目标亮度 ~ 背景亮度 * 1.1（略亮于人眼可辨）
        _body = a > 0.5
        if _body.any():
            _fg_luma = float(rgb[_body.squeeze(-1)].mean())
            _ratio = min(1.0, (_bg_luma * 1.15) / max(_fg_luma, 1.0))
            _ratio = max(_ratio, 0.15)
            rgb = rgb * _ratio
            print("    暗光适配: 背景亮度 %.0f 前景 %.0f -> 系数 %.2f" % (
                _bg_luma, _fg_luma, _ratio))

    # 车身区域：直接覆盖
    body = a > 0.98
    # 半透明边缘：混合
    comp = rgb * a + bgf * (1 - a)

    # 阴影修复：Shadow Catcher 输出的阴影是暗部，
    # 但完全透明区(alpha=0)不该被压暗
    # 找出 alpha 很小但比背景暗的地方 = 阴影
    # BROADCAST-FIX: a 是 (H,W,1)，需扩展成 (H,W,3) 与 comp 对齐
    shadow_zone = np.repeat(a < 0.02, 3, axis=2)
    darker = comp < bgf
    max_darken = shadow_max
    ratio = np.clip(comp / np.maximum(bgf, 1e-3), 0, 1)
    ratio = np.maximum(ratio, 1.0 - max_darken)
    comp_fix = np.where(shadow_zone & darker, bgf * ratio, comp)

    cv2.imwrite(out_path, np.clip(comp_fix, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 94])
    vis = float((a > 0.5).sum()) / max(float((a > 0.02).sum()), 1)
    return True, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="20261120_HS5001_changchun_1v_extremely_closetruck")
    ap.add_argument("--count", type=int, default=5)
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
    out_dir = os.path.join(args.out, args.batch)
    os.makedirs(out_dir, exist_ok=True)

    n_ok = n_skip = 0
    for fi in range(min(args.count, len(frames))):
        fr = frames[fi]
        d = os.path.join(clip_dir, fr)
        imgs = sorted(glob.glob(os.path.join(d, "cam_st_x2f_front",
                                             "*.jpg")))
        if not imgs:
            continue
        img_p = imgs[0]
        stem = os.path.splitext(os.path.basename(img_p))[0]
        xmls = glob.glob(os.path.join(
            d, "label_lidar_hs_pandar128_topmiddle_align",
            "cam_st_x2f_front", "*.xml"))
        if not xmls:
            xmls = glob.glob(os.path.join(
                d, "label_lidar_hs_pandar128_topmiddle", "*.xml"))
        if not xmls:
            print("  [%d] no calib, skip" % fi)
            continue

        K, T, R, t = parse_calib(xmls[0])
        img = cv2.imread(img_p)
        if img is None:
            continue

        # A: Mask2Former 校验
        chk = MaskChecker(img, K, R, t)
        # TUPLE-FIX: sample_vehicles_v2 返回 (list, tried)
        veh, _tried = sample_vehicles_v2(assets, args.per_frame, rng, chk)
        if not veh:
            print("  [%d] %s  A: 无合法空位，跳过" % (fi, stem))
            n_skip += 1
            continue
        print("  [%d] %s  A通过 %d 辆: %s" % (
            fi, stem, len(veh),
            ",".join(v["class"] for v in veh)))

        # 渲染
        Tlist = T.tolist()
        # LOCAL-TMP: 中间文件全部放本地，网盘只放最终 jpg
        _tmp = "/tmp/_canon_work"
        os.makedirs(_tmp, exist_ok=True)
        fg_p = os.path.join(_tmp, "fg_%s.png" % stem)
        dep_p = os.path.join(_tmp, "dep_%s" % stem)
        cfg = {"bg": img_p, "K": K, "T": Tlist, "vehicles": veh,
               "out": fg_p, "depth_npy": dep_p,
               "seed": args.seed + fi}
        cfp = "/tmp/_cv2_%s.json" % stem
        with open(cfp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)

        r = subprocess.run([BLENDER, "--background", "--python",
                            RENDER_PY, "--", cfp],
                           capture_output=True, text=True, timeout=1200)
        if not os.path.isfile(fg_p):
            print("    渲染失败")
            continue

        # B + 阴影修复
        # LIGHT-ADAPT: 按背景平均亮度判定昼夜
        _bg = cv2.imread(img_p)
        _luma = float(cv2.cvtColor(_bg, cv2.COLOR_BGR2GRAY).mean())
        is_dark = (_luma < 85) or ("night" in args.batch.lower())
        smax = 0.15 if is_dark else 0.35
        final = os.path.join(out_dir, "%s_obj.jpg" % stem)
        ok, vis = composite_v2(img_p, fg_p, dep_p, final,
                               shadow_max=smax, dark_mode=is_dark)
        print("    合成 ok=%s 可见率=%.2f 阴影上限=%.2f" % (ok, vis, smax))

        try:
            os.remove(fg_p)
        except Exception:
            pass
        if ok and vis >= MIN_VISIBLE:
            n_ok += 1
        else:
            print("    可见率不足 %.0f%%，丢弃" % (MIN_VISIBLE * 100))
            try:
                os.remove(final)
            except Exception:
                pass

    print()
    print("完成 %d 帧，跳过 %d 帧" % (n_ok, n_skip))


if __name__ == "__main__":
    main()
