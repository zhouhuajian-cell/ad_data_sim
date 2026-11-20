#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""render_selected.py — 对指定的一组帧做仿真（复用 V4.11 渲染器）"""
import argparse
import json
import os
import cv2
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--country", default="BR")
    ap.add_argument("--signs-dir", default="/data/tsr_library_v2")
    ap.add_argument("--pin-class", default="")
    ap.add_argument("--jobs-progress", default="")
    a = ap.parse_args()

    items = json.load(open(a.list, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    ann = os.path.join(a.out, "annotations")
    os.makedirs(ann, exist_ok=True)

    from quality_contract import check_frame
    from replace_sign import detect_signs, load_sign, sign_mask
    from batch_replace import list_signs, std_shape, asset_shape
    from pipeline_v411 import SignRealismV411Renderer

    renderer = SignRealismV411Renderer()
    table = list_signs(a.signs_dir, a.country)
    print("素材表 %d 项" % len(table), flush=True)

    total = len(items)
    ok = fail = 0
    qc_unchecked = 0
    t0 = time.time()

    def wp(cur, extra=None):
        if not a.jobs_progress:
            return
        d = {"phase": "running", "done": cur, "total": total,
             "ok": ok, "fail": fail, "qc_unchecked": qc_unchecked,
             "elapsed": round(time.time() - t0, 1), "alive": True}
        if extra:
            d.update(extra)
        try:
            json.dump(d, open(a.jobs_progress, "w", encoding="utf-8"),
                      ensure_ascii=False)
        except Exception:
            pass

    wp(0)

    for i, it in enumerate(items):
        p = it.get("path") or ""
        stem = it.get("stem") or os.path.basename(p)
        if not p or not os.path.isfile(p):
            fail += 1
            wp(i + 1); continue

        im = cv2.imread(p)
        if im is None:
            fail += 1
            wp(i + 1); continue
        H, W = im.shape[:2]

        try:
            dets = detect_signs(p, conf=0.25)
        except Exception as e:
            print("  detect fail %s: %s" % (stem[:30], e), flush=True)
            fail += 1
            wp(i + 1); continue

        objects = []
        out_im = im

        for d in dets:
            cls = d.get("class") or d.get("cls") or ""
            if a.pin_class:
                parts = a.pin_class.split("_")
                fam = ["_".join(parts[:k]) for k in range(min(3, len(parts)), 0, -1)]
                if not any(cls.startswith(f) for f in fam):
                    continue

            sh = std_shape(im, d["bbox"])
            print("    shape=%s bbox=%s" % (sh, d["bbox"]), flush=True)
            # 形状识别在小标牌上不可靠：unknown 时按类别推断
            if sh == "unknown":
                if cls.startswith("speed_limit") or cls.startswith("prohibition"):
                    sh = "circle"
                elif cls.startswith("warning"):
                    sh = "triangle"
                else:
                    sh = "rect"
            want = "circle" if sh == "circle" else ("triangle" if sh == "triangle" else "rect")

            if a.pin_class:
                pin_nm = a.pin_class
                pp = table.get(pin_nm)
                if not pp:
                    for nm, pth in table.items():
                        if nm.rsplit(".", 1)[0] == pin_nm:
                            pp, pin_nm = pth, nm
                            break
                if not pp:
                    continue
                cls_name, path = pin_nm.rsplit(".", 1)[0], pp
            else:
                # 三级回退：同形状 → 同族任意形状 → 同大类前缀
                cands = [(nm, pth) for nm, pth in table.items()
                         if (nm == cls or nm.startswith(cls + "_"))
                         and asset_shape(nm) == want]
                if not cands:
                    cands = [(nm, pth) for nm, pth in table.items()
                             if (nm == cls or nm.startswith(cls + "_"))]
                if not cands:
                    base = cls.split("_")[0]
                    cands = [(nm, pth) for nm, pth in table.items()
                             if nm.split("_")[0] == base]
                if not cands:
                    print("    无素材 cls=%s want=%s" % (cls, want), flush=True)
                    continue
                cls_name, path = cands[0]

            try:
                asset = load_sign(path)
                out_im, bb46, _, _ = renderer.render(out_im, d["bbox"], asset)
                objects.append({
                    "class": cls_name,
                    "category": cls_name.split("_")[0],
                    # P0-FIX: 最终输出框必须对应实际渲染位置
                    "bbox": [round(float(v), 1) for v in bb46],
                    # 原检测框单独保留，用于回溯
                    "source_bbox": [round(float(v), 1) for v in d["bbox"]],
                    "score": round(float(d.get("conf") or 0), 3),
                    "material": os.path.basename(path),
                    "renderer": "v4.11",
                    "orig_class": cls,
                    "orig_conf": round(float(d.get("conf") or 0), 3),
                })
            except Exception as e:
                print("  render fail %s: %s" % (stem[:30], e), flush=True)
                continue

        if not objects:
            fail += 1
            wp(i + 1); continue

        out_jpg = os.path.join(a.out, stem + "_composed.jpg")

        # ---- 先编码，检查实际将交付的像素，避免检查通过后再次压缩改变结果 ----
        encode_ok, encoded = cv2.imencode(
            ".jpg", out_im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not encode_ok:
            raise RuntimeError("JPEG encoding failed")

        delivered = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if delivered is None:
            raise RuntimeError("JPEG decoding failed")

        qstatus, quality_records = check_frame(im, delivered, objects)

        if qstatus == "failed":
            fail += 1
            print(json.dumps({"source": p,
                              "quality_reject": quality_records},
                             ensure_ascii=False), flush=True)
            wp(i + 1)
            continue

        # 此后保存的就是刚刚验收过的字节，不再二次编码
        with open(out_jpg, "wb") as output_file:
            output_file.write(encoded.tobytes())

        # unchecked 单独计数：不判失败，转人工复核
        if qstatus == "unchecked":
            qc_unchecked += 1

        json.dump({
            "image": stem + "_composed.jpg",
            "source_image": os.path.basename(p),
            "source_dir": it.get("seq") or "",
            "source_path": p,
            "width": W, "height": H,
            "objects": objects,
            "label_completeness": "synthetic_targets_only",
            "quality": {
                "policy": "pixel_and_position_v1",
                "passed": qstatus == "passed",
                "status": qstatus,
                "semantic_status": "not_checked",
                "objects": quality_records,
            },
            # 明确声明训练资格：像素/位置通过 ≠ 可训练。
            # 还需要：原真实标签完整合并、标定来源、多相机时序验收。
            "training_eligible": False,
            "training_blockers": [
                "no_real_label_merge",
                "no_multi_camera_temporal",
                "semantic_not_checked",
            ],
        }, open(os.path.join(ann, stem + ".json"), "w", encoding="utf-8"),
            ensure_ascii=False, indent=1)
        ok += 1
        print("  [%d/%d] %s -> %s" % (i + 1, total, stem[:40], objects[0]["class"]),
              flush=True)
        wp(i + 1)

    wp(total, {"phase": "done", "alive": False})
    print("完成 产出 %d / %d（未检查 %d）" % (ok, total, qc_unchecked), flush=True)


if __name__ == "__main__":
    main()
