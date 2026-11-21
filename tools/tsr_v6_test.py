#!/usr/bin/env python3
"""用真实系统的素材选择逻辑做对照测试。

关键：按 render_selected.py 的三级回退选素材
  1. 同类别 + 同形状
  2. 同类别（任意形状）
  3. 同大类前缀

对比：
  A = pipeline_v411.SignRealismV411Renderer（现有最终版）
  B = V5（本体裁切 + 多方法形状 + ramp 羽化）
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from batch_replace import asset_shape, list_signs, std_shape
from replace_sign import detect_signs, load_sign
from tsr_shape import normalize_sign
from tsr_shape_v2 import detect_shape_mask, detect_shape_vote
from tsr_v5 import composite_v5

OUT = "/data/render/tsr_v6"


def pick_asset(table, cls, want):
    """三级回退（复制 render_selected.py 的逻辑）。"""
    cands = [(nm, pth) for nm, pth in table.items()
             if (nm == cls or nm.startswith(cls + "_"))
             and asset_shape(nm) == want]
    if cands:
        return cands[0], "同类别+同形状"
    cands = [(nm, pth) for nm, pth in table.items()
             if (nm == cls or nm.startswith(cls + "_"))]
    if cands:
        return cands[0], "同类别"
    base = cls.split("_")[0]
    cands = [(nm, pth) for nm, pth in table.items()
             if nm.split("_")[0] == base]
    if cands:
        return cands[0], "同大类"
    return None, "无"


def infer_shape(cls, detected):
    """小标牌形状不可靠时按类别推断（同 render_selected）。"""
    if detected != "unknown":
        return detected
    if cls.startswith("speed_limit") or cls.startswith("prohibition"):
        return "circle"
    if cls.startswith("warning"):
        return "triangle"
    return "rect"


def coverage(omask, pmask):
    a = (omask > 0)
    b = (pmask > 0)
    n = int(a.sum())
    if n == 0:
        return 0.0
    return float(int(np.logical_and(a, b).sum())) / float(n)


def iou(m1, m2):
    a = (m1 > 0)
    b = (m2 > 0)
    i = int(np.logical_and(a, b).sum())
    u = int(np.logical_or(a, b).sum())
    return float(i) / max(u, 1)


def seam(comp, box):
    h, w = comp.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    g = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x1, y1 = max(4, x1), max(4, y1)
    x2, y2 = min(w - 5, x2), min(h - 5, y2)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return 0.0
    vals = [
        np.abs(np.diff(g[y1 - 3:y1 + 4, x1:x2], axis=0)).mean(),
        np.abs(np.diff(g[y2 - 3:y2 + 4, x1:x2], axis=0)).mean(),
        np.abs(np.diff(g[y1:y2, x1 - 3:x1 + 4], axis=1)).mean(),
        np.abs(np.diff(g[y1:y2, x2 - 3:x2 + 4], axis=1)).mean(),
    ]
    return round(float(np.mean(vals)), 2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-img", type=int, default=6)
    ap.add_argument("--min-area", type=int, default=2000)
    ap.add_argument("--country", default="AD")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    # 素材表
    table = list_signs("/data/tsr_library_v2", a.country)
    print("素材表: %d 项 (country=%s)" % (len(table), a.country))

    # 渲染器
    try:
        from pipeline_v411 import SignRealismV411Renderer
        renderer = SignRealismV411Renderer()
        has_v411 = True
        print("pipeline_v411 可用")
    except Exception as e:
        has_v411 = False
        print("pipeline_v411 不可用:", str(e)[:80])

    # 找图
    cands = []
    for clip in sorted(glob.glob("/data/test_data/real_clips*/*/*")):
        d = os.path.join(clip, "cam_st_x2f_front")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d))[:3]:
            if f.endswith(".jpg"):
                cands.append(os.path.join(d, f))
        if len(cands) >= 60:
            break

    picked = []
    for p in cands:
        try:
            signs = detect_signs(p, conf=0.35)
        except Exception:
            continue
        for s in signs:
            if s["w"] * s["h"] >= a.min_area:
                picked.append((s["w"] * s["h"], p, s))
                break
        if len(picked) >= a.max_img:
            break

    print("选中 %d 张\n" % len(picked))
    if not picked:
        raise SystemExit("无样本")

    rows = []
    results = []

    for area, img_p, s in picked:
        bbox = s["bbox"]
        cls = s["class"]
        orig = cv2.imread(img_p)
        if orig is None:
            continue
        h, w = orig.shape[:2]

        # 形状检测
        sh_det, _, detail = detect_shape_vote(orig, bbox)
        want = infer_shape(cls, sh_det)

        # 选素材
        got, why = pick_asset(table, cls, want)

        rec = {"image": os.path.basename(img_p), "cls": cls,
               "bbox": bbox, "area": area,
               "shape_detected": sh_det, "shape_want": want,
               "pick": got[0] if got else None, "pick_why": why,
               "shape_detail": detail}

        if got is None:
            rec["skip"] = "无素材"
            results.append(rec)
            print("  %-22s %-22s 无素材" % (rec["image"][:22], cls))
            continue

        nm, path = got
        asset = load_sign(path)

        # A: 现有 V411
        A = None
        A_box = bbox
        if has_v411:
            try:
                A, bb46, _, _ = renderer.render(orig.copy(), bbox, asset)
                A_box = bb46
            except Exception as e:
                rec["v411_error"] = str(e)[:80]
                A = None

        # B: V5
        if asset.ndim == 3 and asset.shape[2] == 4:
            sb, sa = asset[..., :3], asset[..., 3]
        else:
            sb = asset[..., :3]
            g2 = cv2.cvtColor(sb, cv2.COLOR_BGR2GRAY)
            sa = np.where(g2 > 248, 0, 255).astype(np.uint8)
            if sa.mean() > 250:
                sa = np.full(g2.shape, 255, np.uint8)

        B, infoB = composite_v5(orig.copy(), sb, sa, bbox, use_lama=True,
                                feather_mode="ramp")
        B_box = infoB.get("placed", bbox)

        # 指标
        omask = detect_shape_mask(orig, bbox, want, pad=0)
        hh, ww = orig.shape[:2]
        mA = np.zeros((hh, ww), np.uint8)
        x1, y1, x2, y2 = [int(v) for v in A_box]
        mA[max(0, y1):y2, max(0, x1):x2] = 255
        mB = np.zeros((hh, ww), np.uint8)
        x1, y1, x2, y2 = [int(v) for v in B_box]
        mB[max(0, y1):y2, max(0, x1):x2] = 255

        rec["A"] = {"cov": round(coverage(omask, mA), 3),
                    "iou": round(iou(omask, mA), 3),
                    "seam": seam(A, A_box) if A is not None else None}
        rec["B"] = {"cov": round(coverage(omask, mB), 3),
                    "iou": round(iou(omask, mB), 3),
                    "seam": seam(B, B_box)}
        results.append(rec)

        print("  %-22s cls=%-22s 形状=%-8s 素材=%-32s (%s)"
              % (rec["image"][:22], cls[:22], want,
                 (got[0] or "")[:32], why))

        for name, im in (("A", A), ("B", B)):
            if im is None:
                continue
            cv2.imwrite(os.path.join(
                OUT, "%s_%s.jpg" % (os.path.splitext(rec["image"])[0], name)),
                im, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 拼图
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = max(int((x2 - x1) * 0.9), 60)
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        tiles = []
        for name, im in (("orig", orig), ("A_v411", A), ("B_v5", B)):
            if im is None:
                continue
            t = im[cy1:cy2, cx1:cx2]
            sc = 340.0 / max(t.shape[0], 1)
            t = cv2.resize(t, (max(int(t.shape[1] * sc), 10), 340),
                           interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, name, (5, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            tiles.append(t)
        if tiles:
            hh2 = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, hh2 - t.shape[0], 0, 5,
                                        cv2.BORDER_CONSTANT,
                                        value=(40, 40, 40)) for t in tiles]
            rows.append(np.hstack(tiles))

    if rows:
        ww2 = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, ww2 - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(40, 40, 40))
                for r in rows]
        cv2.imwrite(os.path.join(OUT, "compare_sheet.jpg"),
                    np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("\n=== 素材选择 ===")
    for r in results:
        if r.get("skip"):
            print("  %-22s 跳过: %s" % (r["image"][:22], r["skip"]))
            continue
        print("  %-22s %-20s → %-34s [%s]"
              % (r["image"][:22], r["cls"][:20],
                 (r["pick"] or "")[:34], r["pick_why"]))

    print("\n=== 贴合质量 ===")
    print("  %-22s %-6s %8s %8s %8s" %
          ("图", "方案", "覆盖率", "IoU", "接缝"))
    for r in results:
        if r.get("skip"):
            continue
        for k in ("A", "B"):
            m = r.get(k)
            if not m or m.get("seam") is None:
                continue
            print("  %-22s %-6s %7.1f%% %8.3f %8.2f"
                  % (r["image"][:22], k, m["cov"] * 100, m["iou"], m["seam"]))

    print("\n产出:", OUT)


if __name__ == "__main__":
    main()
