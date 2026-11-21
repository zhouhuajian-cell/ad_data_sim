"""用 done_data 的真实标注 + 真实标定批量跑 OBJ。

关键：合成物体放在真实标注之间的**空隙**，不与真实物体重叠。

输出：
  <out>/<frame>_composed.jpg
  <out>/<frame>_orig.jpg
  <out>/annotations/<frame>.json
  <out>/contact_sheet.jpg
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
from done_data_labels import (list_frames, object_corners, parse_xml,
                              project_body_to_pixel)
from label_gate import (build_claims, gate_with_real_labels,
                        load_frame_labels, real_object_summary)

try:
    from export_gate import export_decision
    HAS_GATE = True
except ImportError:
    HAS_GATE = False

try:
    from ground_masks import road_mask, foreground_mask, hood_mask
    HAS_MASKS = True
except ImportError:
    HAS_MASKS = False


ASSET_SIZE_M = {
    "cone": (0.28, 0.70),
    "car": (1.80, 1.50),
    "truck": (2.50, 3.20),
    "pedestrian": (0.55, 1.70),
}


def sprite_for(kind, h=256):
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
    pts = []
    for fy in (y + w_m / 2, y - w_m / 2):
        uv, z, ok = project_body_to_pixel([[x, fy, 0.0]], cam)
        if not ok[0] or not np.isfinite(uv[0]).all():
            return None
        pts.append(uv[0])
    uv_l, _, _ = project_body_to_pixel([[x, y + 0.5, 0.0]], cam)
    uv_r, _, _ = project_body_to_pixel([[x, y - 0.5, 0.0]], cam)
    px_per_m = abs(uv_l[0][0] - uv_r[0][0])
    top_dy = h_m * px_per_m
    left, right = pts[0], pts[1]
    return np.array([(left[0], left[1] - top_dy), (right[0], right[1] - top_dy),
                     right, left], np.float32)


def find_gaps(frame_labels, cam, img_shape):
    """在真实物体之间找空隙，返回可放置位置列表 [(x, y, 类型)]。"""
    h, w = img_shape
    real = [o["bbox2d"] for o in (frame_labels.get("objects") or [])
            if o.get("in_front") and o.get("bbox2d")]

    # 候选位置：前方 8~30m，横向 -3~3m
    cands = []
    for x in (8.0, 11.0, 14.0, 18.0, 22.0, 26.0):
        for y in (-2.5, -1.5, -0.8, 0.0, 0.8, 1.5, 2.5):
            quad = project_footprint(cam, x, y, 0.28, 0.70)
            if quad is None:
                continue
            bx = [quad[:, 0].min(), quad[:, 1].min(),
                  quad[:, 0].max(), quad[:, 1].max()]
            if bx[2] < 0 or bx[0] > w or bx[3] < 0 or bx[1] > h:
                continue
            # 与真实物体重叠？
            a = max((bx[2] - bx[0]) * (bx[3] - bx[1]), 1e-6)
            conflict = False
            for rb in real:
                ix0 = max(bx[0], rb[0])
                iy0 = max(bx[1], rb[1])
                ix1 = min(bx[2], rb[2])
                iy1 = min(bx[3], rb[3])
                inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
                if inter <= 0:
                    continue
                ra = max((rb[2] - rb[0]) * (rb[3] - rb[1]), 1e-6)
                if inter / (a + ra - inter) > 0.15:
                    conflict = True
                    break
            if not conflict:
                cands.append((x, y, quad))
    return cands, real


def process_frame(frame_dir, out_dir):
    stem = os.path.basename(frame_dir)
    img_p = os.path.join(frame_dir, "cam_st_x2f_front", stem + ".jpg")
    if not os.path.isfile(img_p):
        # 试别的文件名
        js = glob.glob(os.path.join(frame_dir, "cam_st_x2f_front", "*.jpg"))
        if not js:
            return None
        img_p = sorted(js)[0]

    bg = cv2.imread(img_p)
    if bg is None:
        return None
    h, w = bg.shape[:2]

    labels = load_frame_labels(frame_dir)
    if labels is None:
        return None
    cam = labels["camera"]

    # 找空隙放锥桶
    cands, real = find_gaps(labels, cam, (h, w))
    if not cands:
        return None

    # 最多放 4 个，从远到近
    cands.sort(key=lambda c: -c[0])
    picked = cands[:4]

    out = bg.copy()

    # 前景保护
    if HAS_MASKS:
        road = road_mask(bg)
        fg = foreground_mask(bg, road)
        hood = hood_mask(bg, frac=0.18)
        protect = np.clip(np.maximum(fg, hood), 0, 1)
    else:
        protect = np.zeros((h, w), np.float32)

    synth_anns = []
    for x, y, quad in picked:
        w_m, h_m = ASSET_SIZE_M["cone"]
        sp = sprite_for("cone")
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

        # 阴影
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
        synth_anns.append({
            "class": "cone",
            "bbox": [round(float(xs.min()), 1), round(float(ys.min()), 1),
                     round(float(xs.max()), 1), round(float(ys.max()), 1)],
            "world": {"x_m": x, "y_m": y, "z_m": 0.0},
            "source": "synthetic",
        })

    # 门禁（含真实标注）
    synth_boxes = [a["bbox"] for a in synth_anns]
    if HAS_GATE:
        checks, claims = gate_with_real_labels(labels, synth_boxes)
        gate = export_decision(checks)
    else:
        gate = {"policy": "unavailable", "training_eligible": False,
                "training_blockers": ["gate_missing"], "details": {}}
        claims = []

    # 落盘
    out_p = os.path.join(out_dir, stem + "_composed.jpg")
    orig_p = os.path.join(out_dir, stem + "_orig.jpg")
    cv2.imwrite(out_p, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(orig_p, bg, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ann = {
        "image": os.path.basename(out_p),
        "source_path": img_p,
        "width": w, "height": h,
        "module": "OBJ",
        "camera": cam.get("name"),
        "calibration": {"source": "xml", "fx": cam["fx"], "fy": cam["fy"],
                        "cx": cam["cx"], "cy": cam["cy"]},
        "real_labels": real_object_summary(labels),
        "synthetic": synth_anns,
        "claims": claims,
        "gate": gate,
        "training_eligible": gate["training_eligible"],
        "training_blockers": gate["training_blockers"],
    }
    ad = os.path.join(out_dir, "annotations")
    os.makedirs(ad, exist_ok=True)
    with open(os.path.join(ad, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=1)

    return {"stem": stem, "out": out_p, "orig": orig_p,
            "n_synth": len(synth_anns), "gate": gate,
            "real": ann["real_labels"]}


def make_sheet(results, out_dir):
    if not results:
        return None
    TW, TH = 480, 270
    rows = []
    for r in results[:6]:
        a = cv2.imread(r["orig"])
        b = cv2.imread(r["out"])
        if a is None or b is None:
            continue
        a = cv2.resize(a, (TW, TH), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (TW, TH), interpolation=cv2.INTER_AREA)
        rows.append(np.vstack([a, np.full((3, TW, 3), 40, np.uint8), b]))
    if not rows:
        return None
    while len(rows) % 3:
        rows.append(np.full_like(rows[0], 30))
    sheet = np.vstack([np.hstack(rows[i:i + 3]) for i in range(0, len(rows), 3)])
    p = os.path.join(out_dir, "contact_sheet.jpg")
    cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=(
        "/mnt/done_data_new/18_bev-obj/00_M57/00_daizhongshen_changchun/"
        "已终审/hs5001/20261120_HS5001_changchun_1v_closecar/"
        "133007-1788053647.136"))
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--out", default="/data/render/real_label_batch")
    a = ap.parse_args()

    frames = list_frames(a.batch)
    print("批次:", os.path.basename(a.batch))
    print("帧数:", len(frames))

    os.makedirs(a.out, exist_ok=True)
    sel = frames[:a.frames]

    t0 = time.time()
    results = []
    for i, fd in enumerate(sel):
        try:
            r = process_frame(fd, a.out)
        except Exception as e:
            import traceback
            print("  [%d/%d] %s 失败: %s" % (i + 1, len(sel),
                                            os.path.basename(fd), e))
            traceback.print_exc()
            continue
        if r:
            results.append(r)
            print("  [%d/%d] %s  合成=%d 真实=%s  eligible=%s"
                  % (i + 1, len(sel), r["stem"], r["n_synth"],
                     r["real"]["by_kind"], r["gate"]["training_eligible"]))

    dt = time.time() - t0
    print("\n完成 %d/%d 帧，%.1fs（%.2fs/帧）"
          % (len(results), len(sel), dt, dt / max(len(results), 1)))

    sheet = make_sheet(results, a.out)
    if sheet:
        print("对比图:", sheet)

    if results:
        g = results[0]["gate"]
        print("\n门禁: passed=%d/%d  eligible=%s"
              % (g.get("passed", 0), g.get("checked", 0),
                 g["training_eligible"]))
        print("阻塞:", ", ".join(g["training_blockers"]))
    print("产出:", a.out)


if __name__ == "__main__":
    main()
