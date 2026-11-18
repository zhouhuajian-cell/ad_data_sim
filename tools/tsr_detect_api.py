#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tsr_detect_api.py — 给面板用的单图检测接口（命令行形式，输出 JSON 到 stdout）

为什么单独一个脚本：面板是纯 stdlib 的 python3，而 YOLO 只在 ad_mining 的 venv 里，
所以用「面板起子进程调本脚本」的方式隔离依赖。

用法：python3 tsr_detect_api.py <图片路径> [conf]
输出：{"ok":true,"detections":[{class,conf,bbox,w,h,scene_shape}],"candidates":[...]}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "modules"))


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "缺少图片路径"}))
        return 2
    img = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30

    out = {"ok": False, "detections": [], "candidates": []}
    try:
        import cv2
        from replace_sign import detect_signs
        from batch_replace import detect_shape, NEUTRAL_SUBTYPE

        im = cv2.imread(img)
        if im is None:
            out["error"] = "读不到图片"
            print(json.dumps(out, ensure_ascii=False))
            return 1

        dets = detect_signs(img, conf)
        for d in dets:
            shape = detect_shape(im, d["bbox"])
            # 该大类下的代表性素材类别名（前端展示用）
            sub = NEUTRAL_SUBTYPE.get(d["class"], "")
            out["detections"].append({
                "class": d["class"],
                "conf": d["conf"],
                "bbox": [round(v, 1) for v in d["bbox"]],
                "w": d["w"], "h": d["h"],
                "scene_shape": shape,
                "suggest": ("%s_%s" % (d["class"], sub)) if sub else d["class"],
                "usable": d["w"] >= 24 and shape != "unknown",
            })
        out["ok"] = True
        out["count"] = len(dets)
    except Exception as e:
        out["error"] = str(e)[:300]

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
