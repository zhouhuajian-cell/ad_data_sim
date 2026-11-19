#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_walk.py — 遍历式批量替换：递归扫描目录树，自动挑图跑批量

背景：
  真实数据是两层结构 images/<序列名>/<图片>.jpg，
  而且有大量空目录（151276 个序列里大部分没图）。
  原来的 batch_replace.py 只处理单层目录，需要先手写文件清单。

本脚本做的事：
  1) 递归遍历目录树，收集所有 jpg/png
  2) 可按「最少检出数」「最小标牌尺寸」过滤（只跑有标牌、够大的帧）
  3) 输出到扁平目录：<序列名>__<帧名>，避免不同序列同名覆盖
  4) 产出统一标注 + summary.json

用法：
  python3 batch_walk.py --images <根目录> --country BR --out <输出目录> \
      --limit 1000 --min-sign 40
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def walk_images(root, max_scan=2000000):
    """递归收集图片路径。"""
    exts = (".jpg", ".jpeg", ".png")
    out = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
                n += 1
                if n >= max_scan:
                    return out
    return out


def main():
    import cv2
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="根目录（递归）")
    ap.add_argument("--country", default="BR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1000, help="成功生成的目标张数")
    ap.add_argument("--max-scan", type=int, default=200000,
                    help="最多扫描多少个文件就停（防止扫全盘太慢）")
    ap.add_argument("--min-sign", type=int, default=40,
                    help="标牌短边最小像素（太小看不出效果，默认 40）")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--signs-dir", default="/data/tsr_library_v2")
    ap.add_argument("--vary-class", action="store_true")
    for name in ("bright-gain", "haze-gain", "contrast-gain", "satur-keep", "feather-ratio"):
        ap.add_argument("--" + name, type=float, default=None)
    # 实测检测框/原牌红圈的比值在 0.76~1.39 之间乱跳，任何固定倍数都会在
    # 某些帧上偏大、另一些帧上偏小。1.12 在多数帧上能盖住原牌边缘
    # （偏小会让原牌描边露出来，比偏大更明显），所以默认用 1.12。
    ap.add_argument("--grow", type=float, default=1.12,
                    help="粘贴区域放大倍数（默认 1.12，盖住原牌描边；"
                         "1.0 按检测框原尺寸，实测会露出原牌边缘）")
    ap.add_argument("--realism", default="v3", choices=["v2", "v3"],
                    help="劣化算法版本：v3（P5-P95黑白点映射+迭代模糊匹配）或 v2（旧版）")
    ap.add_argument("--jobs-progress", default="",
                    help="进度写到这个文件，面板可轮询")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    ann_dir = os.path.join(a.out, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    from replace_sign import detect_signs, load_sign, sign_mask
    from sign_realism2 import paste_v2, set_params
    effective_params = set_params(bright_gain=a.bright_gain, haze_gain=a.haze_gain,
                                  contrast_gain=a.contrast_gain, satur_keep=a.satur_keep,
                                  feather_ratio=a.feather_ratio)
    from sign_occlusion import detect_occluder, add_backing_plate
    from batch_replace import (fit_bbox, pick_varied_asset, list_signs,
                               fit_to_sign_contour, std_shape, asset_shape,
                               fit_to_inner, measure_face)

    t0 = time.time()
    print("[1/3] 扫描目录…", flush=True)
    files = walk_images(a.images, a.max_scan)
    print("  找到 %d 个图片文件（%.1fs）" % (len(files), time.time() - t0), flush=True)

    table = list_signs(a.signs_dir, a.country)
    if not table:
        print("ERROR: 国家 %s 在素材库里没有资产" % a.country)
        return 2
    print("  素材库 %s: %d 个类别" % (a.country, len(table)), flush=True)

    ok = skip_no_sign = skip_small = fail = 0
    summary = []
    total = len(files)

    def write_progress(phase, cur, tot, extra=None):
        if not a.jobs_progress:
            return
        d = {"phase": phase, "done": cur, "total": tot,
             "ok": ok, "skip_no_sign": skip_no_sign,
             "skip_small": skip_small, "fail": fail,
             "elapsed": round(time.time() - t0, 1)}
        if extra:
            d.update(extra)
        try:
            with open(a.jobs_progress, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
        except Exception:
            pass

    print("[2/3] 逐帧检测与合成…", flush=True)
    for i, p in enumerate(files):
        if ok >= a.limit:
            break
        if i % 20 == 0:
            write_progress("running", i, total)
            print("  [%d/%d] 成功 %d，跳过(无标牌) %d，跳过(太小) %d，失败 %d"
                  % (i, total, ok, skip_no_sign, skip_small, fail), flush=True)

        try:
            dets = detect_signs(p, a.conf)
            if not dets:
                skip_no_sign += 1
                continue
            # 只处理足够大的标牌（太小合成看不出效果，也检不准）
            dets = [d for d in dets if min(d["w"], d["h"]) >= a.min_sign]
            if not dets:
                skip_small += 1
                continue

            # A detector may return several classes for the same physical sign.
            chosen = []
            for det in sorted(dets, key=lambda item: item["conf"], reverse=True):
                x1, y1, x2, y2 = det["bbox"]
                area = max(0, x2 - x1) * max(0, y2 - y1)
                duplicate = False
                for kept in chosen:
                    a1, b1, a2, b2 = kept["bbox"]
                    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
                    other_area = max(0, a2 - a1) * max(0, b2 - b1)
                    if min(area, other_area) > 0 and intersection / min(area, other_area) >= 0.65:
                        duplicate = True
                        break
                if not duplicate:
                    chosen.append(det)
            dets = chosen

            im = cv2.imread(p)
            if im is None:
                fail += 1
                continue

            # 严格形状过滤：只处理标准圆/方/三角，且素材形状必须一致。
            # 实测形状不匹配是穿帮的主因（圆贴三角、椭圆贴圆都会露边），
            # 宁可跳过，也不贴错。
            keep_dets = []
            for d in dets:
                sh = std_shape(im, d["bbox"])
                if sh == "unknown":
                    continue
                want = "circle" if sh == "circle" else ("triangle" if sh == "triangle" else "rect")
                keep_dets.append((d, want))
            if not keep_dets:
                skip_small += 1     # 统计上并入「不适用形状」
                continue

            seq = os.path.basename(os.path.dirname(p))
            base = os.path.splitext(os.path.basename(p))[0]
            stem = "%s__%s" % (seq, base)
            H, W = im.shape[:2]
            objects = []
            out_im = im

            for d, want in keep_dets:
                # 在该大类下找「形状匹配」的素材
                cands = [(nm, pp) for nm, pp in table.items()
                         if (nm == d["class"] or nm.startswith(d["class"] + "_"))
                         and asset_shape(nm) == want]
                if not cands:
                    continue
                if a.vary_class:
                    import hashlib
                    h = int(hashlib.md5((os.path.basename(p) + str(d["bbox"])).encode()).hexdigest()[:8], 16)
                    cls_name, path = cands[h % len(cands)]
                else:
                    cls_name, path = cands[0]
                s = load_sign(path)
                m = sign_mask(s)
                # 先把检测框收缩到原标牌实际轮廓，避免新牌比原牌大一圈
                # 实测结论（14 个样本，用红色外圈做真值）：
                #   YOLO 框画的是「白色内圈」，不含红色外圈；
                #   红圈比框平均大 (+5.3, +11.0) px。
                #   所以贴图要按「框 × 1.12」放大，而不是缩小 ——
                #   缩小会让原牌红圈整圈露在外面（用户反馈的那种「重影」）。
                # --grow 控制放大倍数，默认 1.12。
                cx = (d["bbox"][0] + d["bbox"][2]) * 0.5
                cy = (d["bbox"][1] + d["bbox"][3]) * 0.5
                hw = (d["bbox"][2] - d["bbox"][0]) * 0.5 * a.grow
                hh = (d["bbox"][3] - d["bbox"][1]) * 0.5 * a.grow
                grown = [cx - hw, cy - hh, cx + hw, cy + hh]
                bbox = fit_bbox(grown, m.shape)
                # 底板方案试过并放弃：
                #   从原牌四角采样底板颜色，但四角往往是背景而不是底板，
                #   贴上去多出一块明显的灰方块，比不带底板更突兀。
                #   现在保持纯圆形贴图，退化到「只盖住牌面」。
                out_im, cinfo = paste_v2(out_im, s[:, :, :3], m, bbox,
                                         use_v3=a.realism == "v3")
                if not cinfo.get("ok"):
                    continue
                objects.append({
                    "class": cls_name,
                    "category": cls_name.split("_")[0],
                    "bbox": [round(float(v), 1) for v in bbox],
                    "score": round(float(d["conf"]), 3),
                    "material": os.path.basename(path),
                    "occluder_ratio": cinfo.get("occluder_ratio", 0),
                })

            if not objects:
                skip_no_sign += 1
                continue

            out_jpg = os.path.join(a.out, stem + "_composed.jpg")
            cv2.imwrite(out_jpg, out_im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rec = {
                "image": os.path.basename(out_jpg),
                "source_image": os.path.basename(p),
                "source_dir": seq,
                "source_path": p,
                "width": W, "height": H,
                "simulation_params": dict(effective_params),
                "bbox_grow": a.grow,
                "objects": objects,
            }
            with open(os.path.join(ann_dir, stem + ".json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            summary.append({"image": rec["image"], "n": len(objects),
                            "classes": [o["class"] for o in objects]})
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print("    [ERR] %s: %s" % (os.path.basename(p), str(e)[:70]), flush=True)

    dt = time.time() - t0
    print("[3/3] 汇总…", flush=True)
    cls_count = {}
    for s in summary:
        for c in s["classes"]:
            cls_count[c] = cls_count.get(c, 0) + 1
    rep = {
        "country": a.country,
        "root": a.images,
        "scanned_files": total,
        "limit": a.limit,
        "ok": ok, "skip_no_sign": skip_no_sign,
        "skip_small": skip_small, "fail": fail,
        "elapsed_sec": round(dt, 1),
        "min_sign_px": a.min_sign,
        "vary_class": a.vary_class,
        "simulation_params": dict(effective_params),
        "bbox_grow": a.grow,
        "class_distribution": dict(sorted(cls_count.items(), key=lambda kv: -kv[1])),
        "items": summary,
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    write_progress("done", total, total)

    print("-" * 56)
    print("扫描 %d 个文件，生成 %d 张（%.0f 秒）" % (total, ok, dt))
    print("跳过：无标牌 %d，标牌太小 %d；失败 %d"
          % (skip_no_sign, skip_small, fail))
    print("类别分布 Top10:")
    for c, n in list(rep["class_distribution"].items())[:10]:
        print("  %-28s %4d" % (c, n))
    print("输出: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
