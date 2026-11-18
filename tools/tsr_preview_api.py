#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tsr_preview_api.py — 给面板用的「单图替换预览」，结果以 JSON 输出到 stdout

用法：
  python3 tsr_preview_api.py <实拍图> <输出jpg> --library <素材库> [--country BR]
                             [--target-class speed_limit_60] [--bbox x1,y1,x2,y2]
                             [--conf 0.3]
输出：
  {"ok":true,"out":"...","objects":[{class,bbox,scene_shape}],"material":"..."}
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "modules"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--library", required=True)
    ap.add_argument("--country", default="")
    ap.add_argument("--target-class", default="")
    ap.add_argument("--shape", default="", choices=["", "circle", "triangle", "rect"])
    ap.add_argument("--strict-shape", action="store_true")
    ap.add_argument("--bbox", default="")
    ap.add_argument("--conf", type=float, default=0.30)
    a = ap.parse_args()

    res = {"ok": False, "objects": []}
    try:
        import cv2
        from replace_sign import detect_signs, load_sign, sign_mask
        from sign_realism import paste_realistic, make_realistic
        from sign_occlusion import paste_with_occlusion
        import batch_replace as _br
        from batch_replace import (detect_shape, shape_matches, fit_bbox,
                                   list_signs, pick_neutral_asset)
        _br.STRICT_SHAPE = bool(a.strict_shape)

        im = cv2.imread(a.image)
        if im is None:
            res["error"] = "读不到图片"
            print(json.dumps(res, ensure_ascii=False))
            return 1

        table = list_signs(a.library, a.country)
        if not table:
            res["error"] = "素材库为空（%s / %s）" % (a.library, a.country)
            print(json.dumps(res, ensure_ascii=False))
            return 1

        # 手动框优先，否则自动检测
        if a.bbox:
            try:
                bb = [float(v) for v in a.bbox.split(",")]
                w = bb[2] - bb[0]
                dets = [{"bbox": bb, "class": "manual", "conf": 1.0, "w": w, "h": bb[3] - bb[1]}]
            except Exception:
                res["error"] = "bbox 格式错误，应为 x1,y1,x2,y2"
                print(json.dumps(res, ensure_ascii=False))
                return 1
        else:
            dets = [d for d in detect_signs(a.image, a.conf) if d["w"] >= 16]
        if not dets:
            res["error"] = "没有检测到标牌"
            print(json.dumps(res, ensure_ascii=False))
            return 1

        composed = im
        for d in dets:
            shape = detect_shape(im, d["bbox"])

            # 选素材：指定类别 > 中性代表
            if a.target_class:
                cls = a.target_class
                path = table.get(cls)
            else:
                if d["class"] == "manual":
                    cls, path = None, None
                else:
                    cls, path = pick_neutral_asset(d["class"], table)
            if not path:
                res["objects"].append({"class": d["class"], "bbox": d["bbox"],
                                       "scene_shape": shape, "skipped": "无匹配素材"})
                continue

            s = load_sign(path)
            m = sign_mask(s)
            if not m.any():
                continue
            # 形状约束：形状不符就不贴（避免圆图案塞进三角牌露角）
            from build_library_mtsd import shape_for
            if a.strict_shape and not shape_matches(shape_for(cls), shape, a.shape):
                res["objects"].append({"class": cls, "bbox": d["bbox"],
                                       "scene_shape": shape,
                                       "skipped": "形状不符(%s vs %s)"
                                                  % (shape_for(cls), shape)})
                continue

            bbox = fit_bbox(d["bbox"], m.shape, keep_ratio=True)
            out_im, cinfo = paste_with_occlusion(
                composed, s[:, :, :3], m, bbox,
                degrade_fn=lambda sig, alp, ref: make_realistic(sig, alp, ref, strength=1.0),
                shadow=True, strength=1.0)
            if not cinfo.get("ok"):
                continue
            composed = out_im
            res["objects"].append({"class": cls, "bbox": [round(v, 1) for v in bbox],
                                   "scene_shape": shape, "material": os.path.basename(path)})

        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        cv2.imwrite(a.out, composed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        res["ok"] = bool([o for o in res["objects"] if "skipped" not in o])
        res["out"] = a.out
        if not res["ok"]:
            res["error"] = "所有检测都因素材/形状限制跳过"
    except Exception as e:
        import traceback
        res["error"] = str(e)[:300]
        res["trace"] = traceback.format_exc()[-400:]

    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
