"""用 Maxieye 真实标定批量跑 OBJ + GND。

输出：
  <out>/<frame>_composed.jpg     合成图
  <out>/<frame>_orig.jpg         原图（对照）
  <out>/annotations/<frame>.json 标注 + 门禁
  <out>/contact_sheet.jpg        缩略对比

用法：
  python batch_real.py --frames 20 --out /data/render/real_batch --module both
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from maxieye_calib import (body_to_pixel, find_dataset, ground_homography,
                           list_frames, load_camera, pixel_to_ground)

try:
    from export_gate import build_checks_from_artifacts, export_decision, make_check
    HAS_GATE = True
except ImportError:
    HAS_GATE = False

try:
    from ground_masks import road_mask, foreground_mask, hood_mask
    HAS_MASKS = True
except ImportError:
    HAS_MASKS = False


# 物体真实尺寸（宽, 高）米
ASSET_SIZE_M = {
    "cone": (0.28, 0.70),
    "car": (1.80, 1.50),
    "suv": (1.90, 1.75),
    "truck": (2.50, 3.20),
    "pedestrian": (0.55, 1.70),
}


def sprite_for(kind, h=256):
    """生成精灵图（没有预渲染资产时用占位）。"""
    w = max(int(h * 0.62), 8)
    im = np.zeros((h, w, 4), np.uint8)
    if kind == "cone":
        pts = np.array([[w // 2, 8], [w - 6, h - 6], [6, h - 6]], np.int32)
        cv2.fillPoly(im, [pts], (30, 130, 240, 255))
        cv2.rectangle(im, (3, h - 12), (w - 3, h - 3), (25, 25, 25, 255), -1)
    elif kind in ("car", "suv", "truck"):
        cv2.rectangle(im, (4, int(h * 0.32)), (w - 4, h - 6), (150, 150, 150, 255), -1)
        cv2.rectangle(im, (int(w * 0.16), int(h * 0.10)),
                      (int(w * 0.84), int(h * 0.36)), (110, 110, 110, 255), -1)
    else:
        cv2.rectangle(im, (int(w * 0.30), 16), (int(w * 0.70), int(h * 0.55)),
                      (90, 140, 200, 255), -1)
        cv2.rectangle(im, (int(w * 0.28), int(h * 0.55)), (int(w * 0.72), h - 6),
                      (60, 60, 160, 255), -1)
    return im


def project_footprint(cam, x, y, w_m, h_m):
    """地面点 (x 前, y 左) + 尺寸 → 像素四边形。"""
    pts = []
    for fx, fy in ((x, y + w_m / 2), (x, y - w_m / 2)):
        uv, z, ok = body_to_pixel([[fx, fy, 0.0]], cam)
        if not ok[0] or not np.isfinite(uv[0]).all():
            return None
        pts.append(uv[0])

    # 1 米横向 = 多少像素（用来换算高度）
    uv_l, _, _ = body_to_pixel([[x, y + 0.5, 0.0]], cam)
    uv_r, _, _ = body_to_pixel([[x, y - 0.5, 0.0]], cam)
    px_per_m = abs(uv_l[0][0] - uv_r[0][0])
    top_dy = h_m * px_per_m

    left, right = pts[0], pts[1]
    tl = (left[0], left[1] - top_dy)
    tr = (right[0], right[1] - top_dy)
    return np.array([tl, tr, right, left], np.float32)


def draw_ground_line(bg, cam, x0, x1, y, w_m, color=(240, 240, 240)):
    """在路面上画一条标线。"""
    pts = []
    for x in np.arange(x0, x1, 1.0):
        uv, z, ok = body_to_pixel([[x, y, 0.0]], cam)
        if ok[0] and np.isfinite(uv[0]).all():
            pts.append(uv[0])
    if len(pts) < 2:
        return None
    pts = np.array(pts, np.float32)
    # 沿线生成四边形（左右各半宽）
    left, right = [], []
    for (px, py), (x, _) in zip(pts, [(x, 0) for x in np.arange(x0, x1, 1.0)]):
        uvl, _, ok1 = body_to_pixel([[x, y + w_m / 2, 0.0]], cam)
        uvr, _, ok2 = body_to_pixel([[x, y - w_m / 2, 0.0]], cam)
        if ok1[0] and ok2[0]:
            left.append(uvl[0])
            right.append(uvr[0])
    if len(left) < 2:
        return None
    poly = np.array(left + right[::-1], np.float32)
    return poly


def process_frame(img_path, cam, module, out_dir, stem):
    """处理一帧。"""
    bg = cv2.imread(img_path)
    if bg is None:
        return None
    h, w = bg.shape[:2]

    out = bg.copy()
    anns = []
    masks_info = {}

    # ---- GND：车道线 ----
    if module in ("gnd", "both"):
        for (y, w_m) in ((-5.25, 0.20), (-1.75, 0.15), (1.75, 0.15), (5.25, 0.20)):
            poly = draw_ground_line(bg, cam, 6.0, 45.0, y, w_m)
            if poly is None:
                continue
            cv2.fillPoly(out, [np.rint(poly).astype(np.int32)],
                         (245, 245, 245))
            anns.append({"module": "GND", "class": "lane_line",
                         "world": {"y_m": y, "x0_m": 6.0, "x1_m": 45.0},
                         "width_m": w_m})

        # 前景遮罩：被车挡住的线要扣掉
        if HAS_MASKS:
            road = road_mask(bg)
            fg = foreground_mask(bg, road)
            hood = hood_mask(bg, frac=0.18)
            protect = np.clip(np.maximum(fg, hood), 0, 1)
            diff = np.abs(out.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
            paint = np.clip(diff / 3.0, 0, 1)
            keep = paint * (1.0 - protect)
            out = (bg.astype(np.float32) * (1 - keep[..., None])
                   + out.astype(np.float32) * keep[..., None])
            out = np.clip(out, 0, 255).astype(np.uint8)
            masks_info = {"fg_mean": round(float(fg.mean()), 4),
                          "hood_mean": round(float(hood.mean()), 4),
                          "blocked": round(float(1 - keep.sum() / max(paint.sum(), 1e-6)), 4)}

    # ---- OBJ：锥桶 + 车 ----
    if module in ("obj", "both"):
        # 按距离从远到近画
        objs = [
            ("cone", 12.0, -1.0), ("cone", 14.0, 0.0), ("cone", 16.0, 1.0),
            ("car", 25.0, -3.5), ("cone", 18.0, -2.0),
        ]
        objs.sort(key=lambda o: -o[1])

        if HAS_MASKS:
            road = road_mask(bg)
            fg = foreground_mask(bg, road)
            hood = hood_mask(bg, frac=0.18)
            protect = np.clip(np.maximum(fg, hood), 0, 1)
        else:
            protect = np.zeros((h, w), np.float32)

        for kind, x, y in objs:
            w_m, h_m = ASSET_SIZE_M[kind]
            quad = project_footprint(cam, x, y, w_m, h_m)
            if quad is None:
                continue
            if quad[:, 0].max() < 0 or quad[:, 0].min() > w:
                continue
            if quad[:, 1].max() < 0 or quad[:, 1].min() > h:
                continue

            sp = sprite_for(kind)
            sh, sw = sp.shape[:2]
            src = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32)
            Pm = cv2.getPerspectiveTransform(src, quad.astype(np.float32))

            a = sp[..., 3].astype(np.float32) / 255.0
            rgb = sp[..., :3].astype(np.float32)
            prem = rgb * a[..., None]

            a_w = cv2.warpPerspective(a, Pm, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            p_w = cv2.warpPerspective(prem, Pm, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
            a_w = np.clip(a_w, 0, 1)
            straight = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
            straight[a_w <= 1e-4] = 0

            # 接地阴影
            base = quad.astype(np.int32)
            cx = int((base[2][0] + base[3][0]) / 2)
            cy = int((base[2][1] + base[3][1]) / 2)
            rw = max(int(abs(base[1][0] - base[0][0]) * 0.55), 4)
            rh = max(int(rw * 0.25), 2)
            shm = np.zeros((h, w), np.float32)
            cv2.ellipse(shm, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)
            shm = cv2.GaussianBlur(shm, (13, 13), 0) * 0.35
            out = (out.astype(np.float32) * (1 - shm[..., None])).astype(np.uint8)

            keep = a_w * (1.0 - protect)
            out = (out.astype(np.float32) * (1 - keep[..., None])
                   + straight * keep[..., None])
            out = np.clip(out, 0, 255).astype(np.uint8)

            xs, ys = quad[:, 0], quad[:, 1]
            anns.append({"module": "OBJ", "class": kind,
                         "bbox": [round(float(xs.min()), 1), round(float(ys.min()), 1),
                                  round(float(xs.max()), 1), round(float(ys.max()), 1)],
                         "world": {"x_m": x, "y_m": y, "z_m": 0.0},
                         "size_m": [w_m, h_m]})

    # ---- 门禁 ----
    if HAS_GATE:
        checks = {
            "calibration": make_check("passed",
                                      evidence_obj={"source": "maxieye_calib",
                                                    "reproj_err": cam.get("reproj_err")},
                                      note="厂家真实标定"),
            "original_labels": make_check("unchecked", note="未接入原始标注"),
            "geometry": make_check("passed" if anns else "unchecked",
                                   evidence_obj={"n": len(anns)} if anns else None,
                                   note="真实标定几何"),
            "occlusion": make_check("passed" if masks_info else "unchecked",
                                    evidence_obj=masks_info if masks_info else None,
                                    note="前景遮罩" if masks_info else "无遮罩"),
        }
        for nm in ("appearance", "multi_camera", "temporal", "topology", "split"):
            checks[nm] = make_check("unchecked", note="需人工复核")
        gate = export_decision(checks)
    else:
        gate = {"policy": "unavailable", "training_eligible": False,
                "training_blockers": ["gate_module_missing"], "details": {}}

    # ---- 落盘 ----
    out_p = os.path.join(out_dir, stem + "_composed.jpg")
    orig_p = os.path.join(out_dir, stem + "_orig.jpg")
    cv2.imwrite(out_p, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(orig_p, bg, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ann = {
        "image": os.path.basename(out_p),
        "source_image": os.path.basename(img_path),
        "source_path": img_path,
        "width": w, "height": h,
        "module": module.upper(),
        "camera": cam["camera_id"],
        "calibration": {"source": "maxieye",
                        "reproj_err": cam.get("reproj_err"),
                        "fx": cam["fx"], "fy": cam["fy"],
                        "cx": cam["cx"], "cy": cam["cy"]},
        "objects": anns,
        "masks": masks_info,
        "gate": gate,
        "training_eligible": gate["training_eligible"],
        "training_blockers": gate["training_blockers"],
    }
    ad = os.path.join(out_dir, "annotations")
    os.makedirs(ad, exist_ok=True)
    with open(os.path.join(ad, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=1)

    return {"stem": stem, "out": out_p, "orig": orig_p,
            "n_obj": len([a for a in anns if a.get("module") == "OBJ"]),
            "n_gnd": len([a for a in anns if a.get("module") == "GND"]),
            "gate": gate}


def make_sheet(results, out_dir):
    """对比图：上原图，下合成。"""
    if not results:
        return None
    TH_W, TH_H = 480, 195
    rows = []
    for r in results[:12]:
        a = cv2.imread(r["orig"])
        b = cv2.imread(r["out"])
        if a is None or b is None:
            continue
        a = cv2.resize(a, (TH_W, TH_H), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (TH_W, TH_H), interpolation=cv2.INTER_AREA)
        sep = np.full((4, TH_W, 3), 40, np.uint8)
        rows.append(np.vstack([a, sep, b]))
    if not rows:
        return None
    sheet = np.hstack(rows) if len(rows) <= 3 else np.vstack(
        [np.hstack(rows[i:i + 3] + [np.full((rows[0].shape[0], 4, 3), 40, np.uint8)]
                   * (3 - len(rows[i:i + 3])))
         for i in range(0, min(len(rows), 6), 3)])
    p = os.path.join(out_dir, "contact_sheet.jpg")
    cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default="/data/render/real_batch")
    ap.add_argument("--module", default="both", choices=["obj", "gnd", "both"])
    ap.add_argument("--cam", default="cam_st_x2f_front")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    ds = find_dataset("141705")
    if not ds:
        raise SystemExit("找不到数据集")
    cam = load_camera(ds, a.cam)
    frames = list_frames(ds, a.cam)
    if not frames:
        raise SystemExit("无图片")

    print("数据集:", os.path.basename(ds))
    print("相机:   %s (%s)  %.1fx%.1f  reproj=%.3fpx"
          % (cam["camera_id"], a.cam, cam["fx"], cam["fy"],
             cam.get("reproj_err") or -1))
    print("总帧数: %d，本次处理 %d 帧" % (len(frames), a.frames))

    os.makedirs(a.out, exist_ok=True)
    sel = frames[a.start:a.start + a.frames]

    t0 = time.time()
    results = []
    for i, p in enumerate(sel):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            r = process_frame(p, cam, a.module, a.out, stem)
        except Exception as e:
            print("  [%d/%d] %s 失败: %s" % (i + 1, len(sel), stem, e))
            continue
        if r:
            results.append(r)
            print("  [%d/%d] %s  OBJ=%d GND=%d  eligible=%s"
                  % (i + 1, len(sel), stem, r["n_obj"], r["n_gnd"],
                     r["gate"]["training_eligible"]))

    dt = time.time() - t0
    print("\n完成 %d 帧，耗时 %.1fs（%.2fs/帧）"
          % (len(results), dt, dt / max(len(results), 1)))

    sheet = make_sheet(results, a.out)
    if sheet:
        print("对比图:", sheet)

    # 汇总
    if results:
        g = results[0]["gate"]
        print("\n门禁（首帧）: eligible=%s  passed=%d/%d"
              % (g["training_eligible"], g.get("passed", 0), g.get("checked", 0)))
        print("阻塞:", ", ".join(g["training_blockers"][:6]))
    print("产出目录:", a.out)


if __name__ == "__main__":
    main()
