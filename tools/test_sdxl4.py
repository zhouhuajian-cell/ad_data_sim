#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdxl4.py — 按用户建议调 strength / canny_scale，并对比 prompt 的影响

用户给的调参方向：
  光影不够融 → strength 0.48→0.55~0.60，canny 保持 0.75
  数字边缘扭曲 → canny 提到 0.85，或 strength 降到 0.38~0.42

上一轮的问题：SD 把红圈改成了粗黑边卡通样式。
猜测是 prompt 里写了 "traffic sign"，激活了 SD 的"干净标牌"先验。
这轮同时对比两套 prompt。
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/opt/ue5_synth/tools")

import cv2
import numpy as np

SDXL_INPAINT = "/data/models/sdxl-inpaint"
CN_CANNY = "/data/models/controlnet-canny-sdxl"

# 关键改动：不提"标牌"，避免激活 SD 的标牌先验（它会把红圈画成卡通粗边）
PROMPT_NEUTRAL = (
    "dashcam photograph of an outdoor road scene, natural sunlight, "
    "realistic sensor grain, slight lens softness, consistent lighting, "
    "photo taken by a car camera"
)
NEG_NEUTRAL = (
    "cartoon, illustration, 3d render, vector graphic, clean digital art, "
    "oversized borders, thick outlines, glowing, sticker, artificial"
)


def pick_samples(n=3, min_px=60):
    anns = sorted(glob.glob("/data/render/tsr_100/annotations/*.json"))
    out = []
    for a in anns:
        try:
            with open(a, encoding="utf-8") as f:
                d = json.load(f)
            o = d["objects"][0]
            src = d["source_path"]
            if not os.path.isfile(src):
                continue
            bb = o["bbox"]
            sz = min(bb[2] - bb[0], bb[3] - bb[1])
            if sz < min_px:
                continue
            out.append((src, d, o, sz))
        except Exception:
            continue
    out.sort(key=lambda x: -x[3])
    return out[:n]


def crop_view(img, bbox, pad_ratio=0.55, S=230):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = max(10, int((x2 - x1) * pad_ratio))
    H, W = img.shape[:2]
    a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
    a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
    return cv2.resize(img[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)


def fuse_custom(sim, orig, mask, asset, strength, canny_scale, prompt, neg, seed=42):
    """复用 scene_fusion 的 step1/2，但用自定义 prompt 跑 step3。"""
    import torch
    from PIL import Image

    clean = sim.step1_erase_background(orig, mask)
    coarse, new_mask, bbox = sim.step2_coarse_paste(clean, orig, mask, asset)

    img_h, img_w = coarse.shape[:2]
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    side = max(int(max(w, h) * 4.0), 192)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(img_w, x1 + side)
    y2 = min(img_h, y1 + side)
    cw, ch = x2 - x1, y2 - y1
    if cw < 8 or ch < 8:
        return coarse

    patch = coarse[y1:y2, x1:x2]
    orig_patch = orig[y1:y2, x1:x2]
    pm = new_mask[y1:y2, x1:x2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    imask = cv2.dilate(pm, k, iterations=1)

    T = 768
    rb = cv2.resize(patch, (T, T), interpolation=cv2.INTER_LANCZOS4)
    rm = cv2.resize(imask, (T, T), interpolation=cv2.INTER_NEAREST)
    canny = cv2.Canny(rb, 80, 180)
    canny_rgb = Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))

    out_pil = sim.pipe(
        prompt=prompt, negative_prompt=neg,
        image=Image.fromarray(cv2.cvtColor(rb, cv2.COLOR_BGR2RGB)),
        mask_image=Image.fromarray(rm),
        control_image=canny_rgb,
        strength=strength,
        controlnet_conditioning_scale=canny_scale,
        guidance_scale=5.5,
        num_inference_steps=30,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]

    refined = cv2.cvtColor(np.array(out_pil.resize((cw, ch), Image.LANCZOS)),
                           cv2.COLOR_RGB2BGR)
    refined = sim._restore_camera_isp_noise(orig_patch, refined, imask)
    soft = cv2.GaussianBlur(imask, (11, 11), 0).astype(np.float32) / 255.0
    s3 = soft[:, :, None]
    blended = np.clip(refined.astype(np.float32) * s3 +
                      patch.astype(np.float32) * (1 - s3), 0, 255).astype(np.uint8)
    out = coarse.copy()
    out[y1:y2, x1:x2] = blended
    return out


def main():
    samples = pick_samples(3)
    if not samples:
        print("没有样本")
        return 2
    print("样本尺寸:", [int(s[3]) for s in samples])

    from replace_sign import load_sign, sign_mask
    from scene_fusion import SceneFusionTSRSimulator

    print("加载 SDXL…")
    sim = SceneFusionTSRSimulator(inpaint_model=SDXL_INPAINT,
                                  controlnet_model=CN_CANNY)
    print("加载完成")

    # 三组参数：用户建议的两条 + 中性 prompt
    variants = [
        ("s0.58 c0.75", 0.58, 0.75, "old"),
        ("s0.40 c0.85", 0.40, 0.85, "old"),
        ("s0.58 neutral", 0.58, 0.75, "neutral"),
    ]

    rows = []
    for src, d, o, sz in samples:
        orig = cv2.imread(src)
        if orig is None:
            continue
        mat = o.get("material", "")
        mp = os.path.join("/data/tsr_library_v2", "BR", mat)
        if not os.path.isfile(mp):
            continue
        asset = load_sign(mp)
        if asset.shape[2] == 3:
            mm = sign_mask(asset)
            asset = cv2.cvtColor(asset, cv2.COLOR_BGR2BGRA)
            asset[:, :, 3] = mm

        bb = o["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bb]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        hw, hh = int(w * 0.60), int(h * 0.60)
        mask = np.zeros(orig.shape[:2], np.uint8)
        mask[max(0, cy - hh):min(orig.shape[0], cy + hh),
             max(0, cx - hw):min(orig.shape[1], cx + hw)] = 255

        comp_path = os.path.join("/data/render/tsr_100", d["image"])
        base = cv2.imread(comp_path) if os.path.isfile(comp_path) else orig.copy()

        tiles = [crop_view(base, bb)]
        cv2.putText(tiles[0], "PASTE %dpx" % sz, (3, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 200, 255), 1)

        old_prompt = ("authentic dashcam photograph, traffic sign naturally integrated into "
                      "the real outdoor scene, matching ambient sunlight direction, "
                      "subtle weathering and metallic reflection, natural atmospheric haze")
        old_neg = ("pasted sticker, overly sharp, glowing, artificial, cartoon, 3d render, "
                   "distorted text, messy letters, clean digital graphic")

        for name, st, cs, ptype in variants:
            try:
                t0 = time.time()
                p = old_prompt if ptype == "old" else PROMPT_NEUTRAL
                n = old_neg if ptype == "old" else NEG_NEUTRAL
                out = fuse_custom(sim, orig.copy(), mask, asset, st, cs, p, n)
                dt = time.time() - t0
                tiles.append(crop_view(out, bb))
                cv2.putText(tiles[-1], "%s %.0fs" % (name, dt), (3, 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1)
                print("  %-20s %3dpx %-14s %.1fs" % (o["class"][:20], sz, name, dt))
            except Exception as e:
                print("  %s 失败: %s" % (name, str(e)[:100]))

        while len(tiles) < 4:
            tiles.append(np.full((230, 230, 3), 40, np.uint8))
        row = np.hstack([t for p in tiles for t in (p, np.full((230, 5, 3), 40, np.uint8))])
        rows.append(row)

    if rows:
        mw = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], mw - r.shape[1], 3), 40, np.uint8)])
                for r in rows]
        out = np.vstack(rows)
        cv2.imwrite("/tmp/sdxl_v4.jpg", out)
        print("\n对比图 /tmp/sdxl_v4.jpg %s" % (out.shape,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
