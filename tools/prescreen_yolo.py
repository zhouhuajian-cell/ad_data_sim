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
    ap.add_argument("--progress", default="", help="进度文件路径（前端轮询用）")
    ap.add_argument("--recursive", action="store_true",
                    help="递归子目录（网盘数据是 images/<序列>/<帧>.jpg）")
    ap.add_argument("--thumb-dir", default="", help="缩略图输出目录")
    ap.add_argument("--pin-class", default="", help="只保留这一类")
    a = ap.parse_args()

    if not os.path.isdir(a.dir):
        print(json.dumps({"ok": False, "error": "目录不存在: %s" % a.dir},
                         ensure_ascii=False))
        return 2

    if a.recursive:
        # 用带进度回调的遍历，边扫边报，避免前端一直 0/0
        try:
            from batch_walk import walk_images
        except Exception:
            walk_images = None
        if walk_images is not None:
            def _on_scan(nfile, ndir, cur):
                if a.progress:
                    try:
                        json.dump({"done": 0, "total": 0, "usable": 0, "hits": 0,
                                   "finished": False, "phase": "scan",
                                   "scanned": nfile, "dirs": ndir},
                                  open(a.progress, "w", encoding="utf-8"))
                    except Exception:
                        pass
                if nfile and nfile % 2000 == 0:
                    print("扫描中：%d 目录 / %d 图" % (ndir, nfile),
                          file=sys.stderr, flush=True)
            files = walk_images(a.dir, 2000000, _on_scan, every=100)
        else:
            files = []
            for rt, dns, fns in os.walk(a.dir):
                dns[:] = [d for d in dns if not d.startswith(".")]
                for fn in fns:
                    if fn.lower().endswith(EXTS):
                        files.append(os.path.join(rt, fn))
            files.sort()
    else:
        files = sorted(os.path.join(a.dir, f) for f in os.listdir(a.dir)
                       if f.lower().endswith(EXTS))
    if a.limit > 0:
        files = files[:a.limit]
    print("共 %d 张图" % len(files), file=sys.stderr, flush=True)
    # 扫描完成，立刻告诉前端总量（否则前端一直显示 3%）
    if a.progress:
        try:
            json.dump({"done": 0, "total": len(files), "usable": 0, "hits": 0,
                       "finished": False, "phase": "running",
                       "scanned": len(files)},
                      open(a.progress, "w", encoding="utf-8"))
        except Exception:
            pass

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

    def write_progress(done, total, usable, hits):
        if not a.progress:
            return
        try:
            with open(a.progress, "w", encoding="utf-8") as pf:
                json.dump({"done": done, "total": total, "usable": usable,
                           "hits": hits, "finished": False}, pf)
        except Exception:
            pass

    if a.thumb_dir:
        os.makedirs(a.thumb_dir, exist_ok=True)

    for i, p in enumerate(files):
        fn = os.path.basename(p)
        seq = os.path.basename(os.path.dirname(p))
        stem = "%s__%s" % (seq, os.path.splitext(fn)[0])
        rec = {"file": fn, "stem": stem, "path": p, "seq": seq,
               "n": 0, "usable": 0, "dets": []}
        try:
            dets = detect_signs(p, a.conf)
            rec["n"] = len(dets)
            if dets:
                im = cv2.imread(p)
                for d in dets:
                    cls = d.get("class") or d.get("cls") or ""
                    # 指定标牌：类族过滤
                    if a.pin_class:
                        parts = a.pin_class.split("_")
                        fam = ["_".join(parts[:k]) for k in range(min(3, len(parts)), 0, -1)]
                        if not any(cls.startswith(f) for f in fam):
                            continue
                    shape = detect_shape(im, d["bbox"]) if im is not None else "unknown"
                    ok = d["w"] >= a.min_px and shape != "unknown"
                    rec["dets"].append({"class": cls, "conf": d["conf"],
                                        "w": d["w"], "h": d["h"], "shape": shape,
                                        "bbox": d["bbox"], "usable": ok})
                    if ok:
                        rec["usable"] += 1
        except Exception as e:
            rec["error"] = str(e)[:120]
        # 生成缩略图（整图 + 标牌特写）
        if a.thumb_dir and rec["usable"] > 0:
            try:
                im2 = cv2.imread(p)
                if im2 is not None:
                    h, w = im2.shape[:2]
                    sc = 320.0 / max(w, 1)
                    small = cv2.resize(im2, (int(w * sc), int(h * sc)))
                    cv2.imwrite(os.path.join(a.thumb_dir, stem + ".jpg"), small,
                                [cv2.IMWRITE_JPEG_QUALITY, 72])
                    best = max([x for x in rec["dets"] if x["usable"]],
                               key=lambda x: x["w"] * x["h"])
                    x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
                    crop = im2[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        cv2.imwrite(os.path.join(a.thumb_dir, stem + "_sign.jpg"),
                                    cv2.resize(crop, (72, 72)),
                                    [cv2.IMWRITE_JPEG_QUALITY, 82])
            except Exception:
                pass

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

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            write_progress(i + 1, len(files), result["usable_hits"], result["hits"])
            print("  [%d/%d] 有标牌 %d 张" % (i + 1, len(files), result["usable_hits"]),
                  file=sys.stderr, flush=True)

    out = a.out or os.path.join("/tmp", "prescreen_%d.json" % os.getpid())
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    result["out"] = out
    if a.progress:
        try:
            with open(a.progress, "w", encoding="utf-8") as pf:
                json.dump({"done": len(files), "total": len(files),
                           "usable": result["usable_hits"], "hits": result["hits"],
                           "finished": True, "out": out}, pf)
        except Exception:
            pass
    print(json.dumps({k: result[k] for k in
                      ("ok", "total", "hits", "usable_hits", "out")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
