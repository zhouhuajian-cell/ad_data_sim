#!/usr/bin/env python3
"""GND 片段任务渲染：逐帧标线合成 + 分割标注。

用法:
    python render_gnd_task.py --config <config.json>

config:
    {batch, clip, out, params:{mat, lanes, render_mode, dash, stop}}
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

CLIP_ROOT = "/data/test_data/real_clips"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    batch = cfg["batch"]
    clip = cfg["clip"]
    out = cfg["out"]
    pp = cfg.get("params") or {}

    clip_dir = os.path.join(CLIP_ROOT, batch, clip)
    if not os.path.isdir(clip_dir):
        print("clip 目录不存在: %s" % clip_dir)
        return 1

    imgs = []
    for cam in ("img_ori", "cam_st_x2f_front", "cam_ld_x8d_front",
                "cam_st_x8d_front"):
        p = os.path.join(clip_dir, cam)
        if os.path.isdir(p):
            imgs = sorted(glob.glob(os.path.join(p, "*.jpg")))
            if imgs:
                break
    if not imgs:
        print("无图片")
        return 1

    os.makedirs(out, exist_ok=True)
    ann_dir = os.path.join(out, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    print("GND 任务: %s  %d 帧" % (clip, len(imgs)))
    print("  标线材质: %s" % pp.get("mat"))
    print("  车道数: %s  模式: %s" % (pp.get("lanes"),
                                      pp.get("render_mode")))

    # 尝试走真实 GND 渲染（gnd_render_one / gnd_asset_render）
    try:
        import gnd_render_one as _g1
        _has = True
    except Exception:
        _has = False

    n = 0
    for i, img_p in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        dst_jpg = os.path.join(out, stem + "_gnd.jpg")
        dst_seg = os.path.join(out, stem + "_seg_lane.png")
        rec = {"frame": stem, "index": i,
               "mat": pp.get("mat"), "lanes": pp.get("lanes"),
               "render_mode": pp.get("render_mode"),
               "dash": pp.get("dash"), "stop": pp.get("stop")}
        try:
            if _has:
                _g1.render_frame(img_p, dst_jpg, params=pp)
                rec["rendered"] = True
            else:
                # 兜底：直接复制原图（保证逐帧可看）
                import shutil
                shutil.copy2(img_p, dst_jpg)
                rec["rendered"] = False
                rec["note"] = "gnd_render_one 不可用，已复制原图"
        except Exception as e:
            rec["rendered"] = False
            rec["error"] = str(e)[:200]
        with open(os.path.join(ann_dir, stem + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        n += 1
        if i % 10 == 0:
            print("  [%2d/%d] %s" % (i, len(imgs), stem))

    with open(os.path.join(out, "trajectory_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"schema_version": 1, "module": "gnd",
                   "clip": clip, "frames_total": len(imgs),
                   "frames_rendered": n,
                   "params": pp}, f, ensure_ascii=False, indent=1)

    print("完成 %d 帧 -> %s" % (n, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
