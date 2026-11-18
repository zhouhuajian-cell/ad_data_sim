#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prescreen_yolo.py — 批量预筛：找出目录里「有标牌」的图片

为什么需要：
  实拍数据里大部分帧其实没有标牌（或标牌太小看不清），
  逐张人工翻图太慢。先用 YOLO 跑一遍，只保留「检出可用标牌」的帧，
  后续替换和人工确认的工作量能降到 1/5 以下。

用法：
  python3 prescreen_yolo.py <图片目录> [--out 结果.json] [--conf 0.30]
                            [--min-px 24] [--limit 0] [--copy-to 目录]

输出：
  {
    "dir": "...", "total": 200, "hits": 43,
    "results": [
      {"file": "xxx.jpg", "n": 2, "usable": 2,
       "dets": [{"class":"speed_limit","conf":0.88,"w":59,"h":61, "shape":"circle"}]}
    ]
  }
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "modules"))

EXTS = (".jpg", ".jpeg", ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min-px", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--copy-to", default="", help="把命中的图片复制到该目录")
    a = ap.parse_args()

    if not os.path.isdir(a.dir):
        print(json.dumps({"ok": False, "error": "目录不存在: %s" % a.dir},
                         ensure_ascii=False))
        return 2

    files = sorted(f for f in os.listdir(a.dir) if f.lower().endswith(EXTS))
    if a.limit > 0:
        files = files[:a.limit]

    result = {"ok": True, "dir": a.dir, "total": len(files), "hits": 0,
              "usable_hits": 0, "results": []}

    try:
        from replace_sign import detect_signs
        from batch_replace import detect_shape
        import cv2
    except Exception as e:
        print(json.dumps({"ok": False, "error": "依赖缺失: %s" % e}, ensure_ascii=False))
        return 1

    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)

    for i, fn in enumerate(files):
        p = os.path.join(a.dir, fn)
        rec = {"file": fn, "n": 0, "usable": 0, "dets": []}
        try:
            dets = detect_signs(p, a.conf)
            rec["n"] = len(dets)
            if dets:
                im = cv2.imread(p)
                for d in dets:
                    shape = detect_shape(im, d["bbox"]) if im is not None else "unknown"
                    ok = d["w"] >= a.min_px and shape != "unknown"
                    rec["dets"].append({"class": d["class"], "conf": d["conf"],
                                        "w": d["w"], "h": d["h"], "shape": shape,
                                        "usable": ok})
                    if ok:
                        rec["usable"] += 1
        except Exception as e:
            rec["error"] = str(e)[:120]
        if rec["n"] > 0:
            result["hits"] += 1
        if rec["usable"] > 0:
            result["usable_hits"] += 1
            if a.copy_to:
                try:
                    import shutil
                    shutil.copy2(p, os.path.join(a.copy_to, fn))
                except Exception:
                    pass
        result["results"].append(rec)

        if (i + 1) % 20 == 0:
            print("  [%d/%d] 已有标牌 %d 张" % (i + 1, len(files), result["usable_hits"]),
                  file=sys.stderr, flush=True)

    out = a.out or os.path.join("/tmp", "prescreen_%d.json" % os.getpid())
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    result["out"] = out
    print(json.dumps({k: result[k] for k in
                      ("ok", "total", "hits", "usable_hits", "out")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
