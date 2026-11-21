#!/usr/bin/env python3
"""用点云生成高质量 OBJ 素材（替代 GrabCut）。

优势：
  - 点云精确勾出物体轮廓，不包含背景
  - 3D 框提供准确的尺寸和位置
  - 无需猜测

流程：
  1) 读点云 + XML 标注
  2) 3D 框内的点云 → 投到图像 → 掩码
  3) 掩码 × 原图 = 素材
  4) 严格质量校验
"""
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_guard import guard_all_mnt_ops
from label_gate import load_frame_labels
from lidar_io import load_lidar
from real_calib import body_to_pixel_real, load_cam_real

guard_all_mnt_ops()

OUT = "/data/obj_assets_pc"
MAX_PER_CLASS = 100
FRAMES_PER_CLIP = 10
MIN_POINTS = 30          # 点云最少点数
MIN_COVER = 0.25         # 掩码需覆盖 bbox 的比例
MIN_AREA_PX = 6000


def points_in_box(pts, o):
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    cx, cy, cz = (o.get("Wx") or 0.0), (o.get("Wy") or 0.0), (o.get("Wz") or 0.0)
    L, W, H = (o.get("L") or 2.0), (o.get("W") or 1.8), (o.get("H") or 1.5)
    yaw = o.get("yaw") or 0.0
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx, dy = X - cx, Y - cy
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    return ((np.abs(lx) <= L / 2 * 1.15) & (np.abs(ly) <= W / 2 * 1.15)
            & (Z >= cz - H / 2 * 1.15) & (Z <= cz + H / 2 * 1.15))


def mask_from_points(pts, o, cam, shape, dilate=2):
    """3D 框内点云 → 像素掩码。"""
    h, w = shape[:2]
    m = points_in_box(pts, o)
    n_pts = int(m.sum())
    if n_pts < MIN_POINTS:
        return None, n_pts

    bp = pts[m][:, :3]
    uv, zc, ok = body_to_pixel_real(cam, bp)
    valid = ok & np.isfinite(uv).all(axis=1)
    if valid.sum() < MIN_POINTS:
        return None, n_pts

    u, v = uv[valid, 0], uv[valid, 1]
    inb = (u >= -10) & (u < w + 10) & (v >= -10) & (v < h + 10)
    u, v = u[inb], v[inb]
    if len(u) < MIN_POINTS:
        return None, n_pts

    mask = np.zeros((h, w), np.uint8)
    for uu, vv in zip(u, v):
        cv2.circle(mask, (int(uu), int(vv)), dilate, 255, -1)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, n_pts
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8) * 255
    return mask, n_pts


def crop_rgba(bg, mask, pad=2):
    ys, xs = np.where(mask > 0)
    if len(ys) < 50:
        return None
    h, w = mask.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(h - 1, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w - 1, xs.max() + pad)
    if y1 - y0 < 20 or x1 - x0 < 20:
        return None
    a = cv2.GaussianBlur(mask[y0:y1 + 1, x0:x1 + 1], (3, 3), 0)
    return np.dstack([bg[y0:y1 + 1, x0:x1 + 1], a])


def box_sanity(bg, mask, bbox, kind):
    """BOX-SANITY: 校验 3D 框位置是否真的圈到了目标。

    依据：
      1) 框内 vs 框外的颜色差异（框错了 → 几乎一样）
      2) 框内的边缘密度（真物体有轮廓）
      3) 框内的颜色多样性（树叶/杂物也多样，但我们需要形状特征）
    """
    h, w = bg.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w - 1, int(bbox[2]))
    y2 = min(h - 1, int(bbox[3]))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return False, "tiny"

    roi = bg[y1:y2, x1:x2]
    # 环状外框（1.5 倍范围）
    pad = int(max(x2 - x1, y2 - y1) * 0.4)
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)
    ring = bg[oy1:oy2, ox1:ox2].astype(np.float32)
    # 挖掉中心
    ix1, iy1 = x1 - ox1, y1 - oy1
    ix2, iy2 = x2 - ox1, y2 - oy1
    rm = np.ones(ring.shape[:2], np.float32)
    rm[max(0, iy1):iy2, max(0, ix1):ix2] = 0
    outer = ring[rm > 0]

    inner = roi.reshape(-1, 3).astype(np.float32)
    if len(outer) < 100 or len(inner) < 100:
        return False, "few"

    # 1) 颜色差异
    inner_m = inner.mean(axis=0)
    outer_m = outer.mean(axis=0)
    diff = float(np.abs(inner_m - outer_m).mean())
    # 2) 纹理差异（标准差）
    inner_s = float(inner.std(axis=0).mean())
    outer_s = float(outer.std(axis=0).mean())
    # 3) 掩码占比（点云圈到的面积应该适中）
    mfrac = float((mask[y1:y2, x1:x2] > 0).mean())
    # 4) 边缘密度
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edge = float(cv2.Canny(g, 50, 150).mean() / 255.0)

    info = {"diff": round(diff, 1), "inner_std": round(inner_s, 1),
            "mfrac": round(mfrac, 2), "edge": round(edge, 3)}

    # 车辆：框内应与框外有明显差异（车是亮的/有颜色）
    if kind in ("car", "truck", "van", "bus", "special"):
        if diff < 12:
            return False, "flat_diff(%.0f)" % diff
        if inner_s < 14:
            return False, "flat_tex(%.0f)" % inner_s
        if edge < 0.008:
            return False, "no_edge"

    # 行人/骑手：也应与背景有差异
    if kind in ("pedestrian", "rider"):
        if diff < 8:
            return False, "flat_diff"

    # 点云圈到的面积不能过大（说明框太小或点散）
    if mfrac > 0.92:
        return False, "mask_full"

    return True, info


def vehicle_shape_check(rgba, kind):
    """VEHICLE-SHAPE: 车辆形状特征校验。

    车辆特征（树叶/路面不具备）：
      1) 底部更暗（阴影/轮胎）—— 底部亮度 < 中部亮度
      2) 有长的水平线（车身腰线、保险杠）
      3) 颜色数量有限（车漆 1-3 主色）
    """
    if kind not in ("car", "truck", "van", "bus", "special"):
        return True, "skip"

    a = rgba[..., 3]
    rgb = rgba[..., :3]
    h, w = a.shape
    filled = a > 140
    if filled.sum() < 500:
        return False, "few"

    v = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    masked_v = v[filled]
    if len(masked_v) < 200:
        return False, "few"

    # 1) 底部 vs 中部亮度
    rows = np.where(filled.any(axis=1))[0]
    if len(rows) < 20:
        return False, "thin"
    top = rows.min()
    bot = rows.max()
    mid = (top + bot) // 2
    # 底部 25%
    b0 = int(bot - (bot - top) * 0.25)
    bottom_v = v[b0:bot + 1][filled[b0:bot + 1]]
    mid_v = v[mid:b0][filled[mid:b0]]
    if len(bottom_v) < 50 or len(mid_v) < 50:
        return False, "thin2"
    b_mean = float(np.median(bottom_v))
    m_mean = float(np.median(mid_v))
    # 车辆底部有阴影/深色轮胎，通常比中部暗
    if b_mean > m_mean * 1.25:
        return False, "no_shadow"

    # 2) 颜色集中度：主色占比
    px = rgb[filled].reshape(-1, 3).astype(np.uint8)
    # 量化到 32 级
    q = (px // 32).astype(np.int32)
    codes = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    _, cnts = np.unique(codes, return_counts=True)
    top3 = np.sort(cnts)[-3:].sum() if len(cnts) >= 3 else cnts.sum()
    conc = float(top3) / max(len(codes), 1)
    if conc < 0.30:
        return False, "scattered_%.2f" % conc

    # 3) 长水平线（车身边缘）
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                            minLineLength=max(w // 4, 15), maxLineGap=8)
    n_horiz = 0
    if lines is not None:
        arr = np.asarray(lines).reshape(-1, 4)
        for l in arr:
            x1_, y1_, x2_, y2_ = int(l[0]), int(l[1]), int(l[2]), int(l[3])
            dx = abs(x2_ - x1_)
            dy = abs(y2_ - y1_)
            if dx > 0 and dy / max(dx, 1) < 0.25:
                n_horiz += 1
    if n_horiz < 2:
        return False, "no_lines_%d" % n_horiz

    return True, "ok(b=%.0f/m=%.0f,conc=%.2f,nl=%d)" % (
        b_mean, m_mean, conc, n_horiz)


def quality(rgba):
    a = rgba[..., 3]
    h, w = a.shape
    filled = a > 140
    npx = int(filled.sum())
    if npx < MIN_AREA_PX:
        return False, "small"
    fill = filled.mean()
    if fill < 0.45 or fill > 0.95:
        return False, "fill_%.2f" % fill
    cnts, _ = cv2.findContours(filled.astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "nocnt"
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    compact = (peri * peri) / max(area, 1.0)
    if compact > 50:
        return False, "frag"
    px = rgba[..., :3][filled]
    if len(px) < 100:
        return False, "few"
    if px.reshape(-1, 3).std(axis=0).mean() < 10:
        return False, "flat"
    return True, "ok"


def main():
    roots = ["/data/test_data/real_clips",
             "/data/test_data/real_clips_extra"]
    clips = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for b in sorted(glob.glob(os.path.join(r, "*"))):
            if not os.path.isdir(b):
                continue
            for c in sorted(glob.glob(os.path.join(b, "*"))):
                if (os.path.isdir(os.path.join(c, "cam_st_x2f_front"))
                        and os.path.isdir(os.path.join(
                            c, "lidar_hs_pandar128_topmiddle"))):
                    clips.append((os.path.basename(b), c))

    print("可用 clip: %d（有点云）" % len(clips))
    if not clips:
        raise SystemExit("无点云数据")

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    counts = {}
    rej = {}
    saved = 0

    for ci, (bname, cdir) in enumerate(clips):
        cam = load_cam_real(bname, os.path.basename(cdir))
        if cam is None:
            continue

        ld = os.path.join(cdir, "lidar_hs_pandar128_topmiddle")
        bins = sorted(f for f in os.listdir(ld) if f.endswith(".bin"))[:FRAMES_PER_CLIP]

        for bf in bins:
            stem = bf[:-4]
            ip = os.path.join(cdir, "cam_st_x2f_front", stem + ".jpg")
            xp = os.path.join(cdir, "label_lidar_hs_pandar128_topmiddle", stem + ".xml")
            if not (os.path.isfile(ip) and os.path.isfile(xp)):
                continue

            img = cv2.imread(ip)
            if img is None:
                continue
            pts = load_lidar(os.path.join(ld, bf))
            lab = load_frame_labels(cdir, xml_path=xp)
            if lab is None:
                continue
            h, w = img.shape[:2]

            for o in (lab.get("objects") or []):
                if o.get("delete") or not o.get("in_front"):
                    continue
                kind = o.get("kind") or ""
                if kind not in ("car", "truck", "van", "bus",
                                "pedestrian", "rider", "three", "special"):
                    continue
                if counts.get(kind, 0) >= MAX_PER_CLASS:
                    continue

                bbox = o.get("bbox2d")
                if not bbox:
                    continue
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bw * bh < MIN_AREA_PX:
                    rej["area"] = rej.get("area", 0) + 1
                    continue

                mask, npts = mask_from_points(pts, o, cam, img.shape)
                if mask is None:
                    rej["fewpts"] = rej.get("fewpts", 0) + 1
                    continue

                # BOX-SANITY: 校验框位置
                okb, whyb = box_sanity(img, mask, bbox, kind)
                if not okb:
                    k3 = whyb.split("(")[0]
                    rej["box_" + k3] = rej.get("box_" + k3, 0) + 1
                    continue

                # 覆盖率
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(w - 1, int(bbox[2]))
                y2 = min(h - 1, int(bbox[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                cov = (mask[y1:y2, x1:x2] > 0).mean()
                if cov < MIN_COVER:
                    rej["cover"] = rej.get("cover", 0) + 1
                    continue

                rgba = crop_rgba(img, mask)
                if rgba is None:
                    rej["crop"] = rej.get("crop", 0) + 1
                    continue

                ok, why = quality(rgba)
                if not ok:
                    k2 = why.split("_")[0]
                    rej[k2] = rej.get(k2, 0) + 1
                    continue

                # VEHICLE-SHAPE
                oks, whys = vehicle_shape_check(rgba, kind)
                if not oks:
                    k4 = whys.split("_")[0]
                    rej["shape_" + k4] = rej.get("shape_" + k4, 0) + 1
                    continue

                d = os.path.join(OUT, kind)
                os.makedirs(d, exist_ok=True)
                name = "%s_%s_%d" % (bname[:16], stem[:12],
                                     counts.get(kind, 0))
                cv2.imwrite(os.path.join(d, name + ".png"), rgba)
                with open(os.path.join(d, name + ".json"), "w",
                          encoding="utf-8") as f:
                    json.dump({
                        "src": ip, "kind": kind,
                        "3d": {"x": o.get("Wx"), "y": o.get("Wy"),
                               "z": o.get("Wz"),
                               "L": o.get("L"), "W": o.get("W"),
                               "H": o.get("H")},
                        "n_points": npts,
                        "coverage": round(float(cov), 3),
                        "size_px": [int(rgba.shape[1]), int(rgba.shape[0])],
                    }, f, ensure_ascii=False)
                counts[kind] = counts.get(kind, 0) + 1
                saved += 1

        if (ci + 1) % 10 == 0:
            print("  [%d/%d] %d 张  %s" % (ci + 1, len(clips), saved,
                                           dict(sorted(counts.items()))))

    print("\n=== 完成 ===")
    print("保存 %d 张 → %s" % (saved, OUT))
    for k in sorted(counts):
        print("  %-12s %d" % (k, counts[k]))
    print("\n剔除:")
    for k, v in sorted(rej.items(), key=lambda x: -x[1])[:6]:
        print("  %-10s %d" % (k, v))

    with open(os.path.join(OUT, "_index.json"), "w", encoding="utf-8") as f:
        json.dump({"counts": counts, "rejects": rej, "total": saved},
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
