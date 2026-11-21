"""用本地副本跑完整验证：扫所有测试包，看门禁通过率。"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_source import assert_writable, list_clips
from done_data_labels import project_body_to_pixel
from export_gate import export_decision
from gate_extra import augment_gate
from label_gate import (gate_with_real_labels, list_clip_frames,
                        load_frame_labels, real_object_summary)
from batch_real_labels import ASSET_SIZE_M, find_gaps, project_footprint, sprite_for


def process_clip(clip_dir, out_dir, max_frames=10, all_clips=None):
    """处理一个 clip，返回每帧结果。"""
    frames = list_clip_frames(clip_dir)
    if not frames:
        return []

    clip_name = os.path.basename(clip_dir)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, (img_p, xml_name) in enumerate(frames[:max_frames]):
        try:
            r = process_one(clip_dir, img_p, xml_name, out_dir)
        except Exception as e:
            print("    帧 %s 失败: %s" % (os.path.basename(img_p), e))
            continue
        if r:
            results.append(r)
    return results


def process_one(clip_dir, img_p, xml_name, out_dir):
    stem = os.path.splitext(os.path.basename(img_p))[0]
    bg = cv2.imread(img_p)
    if bg is None:
        return None
    h, w = bg.shape[:2]

    labels = load_frame_labels(clip_dir, xml_path=xml_name)
    if labels is None:
        return None
    cam = labels["camera"]

    cands, _ = find_gaps(labels, cam, (h, w))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    picked = cands[:4]

    out = bg.copy()
    synth = []

    try:
        from ground_masks import road_mask, foreground_mask, hood_mask
        road = road_mask(bg)
        fg = foreground_mask(bg, road)
        hood = hood_mask(bg, frac=0.18)
        protect = np.clip(np.maximum(fg, hood), 0, 1)
    except ImportError:
        protect = np.zeros((h, w), np.float32)

    for x, y, quad in picked:
        w_m, h_m = ASSET_SIZE_M["cone"]
        sp = sprite_for("cone")
        sh, sw = sp.shape[:2]
        src = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]],
                       np.float32)
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

        keep = a_w * (1.0 - protect)
        out = (out.astype(np.float32) * (1 - keep[..., None])
               + straight * keep[..., None])
        out = np.clip(out, 0, 255).astype(np.uint8)

        xs, ys = quad[:, 0], quad[:, 1]
        synth.append({
            "class": "cone",
            "bbox": [float(xs.min()), float(ys.min()),
                     float(xs.max()), float(ys.max())],
            "world": {"x_m": x, "y_m": y},
        })

    out_p = os.path.join(out_dir, stem + "_composed.jpg")
    cv2.imwrite(out_p, out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    synth_boxes = [s["bbox"] for s in synth]
    checks, claims = gate_with_real_labels(labels, synth_boxes)
    return {"stem": stem, "out": out_p, "synthetic": synth,
            "checks": checks, "real": real_object_summary(labels),
            "clip": clip_dir}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", type=int, default=10)
    ap.add_argument("--out", default="/data/render/verify_batch")
    a = ap.parse_args()

    assert_writable(a.out)

    clips = list_clips()
    if not clips:
        raise SystemExit("本地无测试数据")

    print("=" * 66)
    print("本地验证：%d 个数据包" % len(clips))
    print("=" * 66)

    all_clip_dirs = []
    for name, p, src in clips:
        for c in sorted(glob.glob(os.path.join(p, "*"))):
            if os.path.isdir(os.path.join(c, "cam_st_x2f_front")):
                all_clip_dirs.append(c)

    t0 = time.time()
    summary = []

    for name, p, src in clips:
        print("\n### %s  [%s]" % (name, src))
        clip_subdirs = [c for c in sorted(glob.glob(os.path.join(p, "*")))
                        if os.path.isdir(os.path.join(c, "cam_st_x2f_front"))]
        for c in clip_subdirs:
            t1 = time.time()
            out_dir = os.path.join(a.out, name)
            rs = process_clip(c, out_dir, a.per_clip, all_clip_dirs)
            dt = time.time() - t1
            if not rs:
                print("  %-26s  跳过（无空隙/无标注）" % os.path.basename(c))
                continue

            # 三项追加
            extra = augment_gate(c, rs, all_clip_dirs)
            merged = dict(rs[0]["checks"])
            merged.update(extra)
            gate = export_decision(merged)

            print("  %-26s  %2d帧  %5.1fs  passed=%d/%d"
                  % (os.path.basename(c), len(rs), dt,
                     gate["passed"], gate["checked"]))
            summary.append({"clip": c, "n": len(rs), "gate": gate,
                            "checks": merged})

    print("\n" + "=" * 66)
    print("总耗时 %.1fs" % (time.time() - t0))

    if summary:
        g = summary[0]["gate"]
        print("\n门禁明细（第一个 clip）:")
        for k, v in g["details"].items():
            mark = "OK " if v["status"] == "passed" else (
                "XX " if v["status"] == "failed" else "-- ")
            print("  %s%-14s %-10s %s"
                  % (mark, k, v["status"], v["note"][:44]))

        passed_counts = [s["gate"]["passed"] for s in summary]
        print("\n通过率: 平均 %.1f/9  最好 %d/9  最差 %d/9"
              % (np.mean(passed_counts), max(passed_counts),
                 min(passed_counts)))
        print("可训练: %d/%d clip"
              % (sum(1 for s in summary if s["gate"]["training_eligible"]),
                 len(summary)))

    # 汇总 JSON
    sp = os.path.join(a.out, "summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump([{"clip": os.path.basename(s["clip"]), "n": s["n"],
                    "passed": s["gate"]["passed"],
                    "eligible": s["gate"]["training_eligible"],
                    "blockers": s["gate"]["training_blockers"]}
                   for s in summary], f, ensure_ascii=False, indent=1)
    print("\n汇总:", sp)


if __name__ == "__main__":
    main()
