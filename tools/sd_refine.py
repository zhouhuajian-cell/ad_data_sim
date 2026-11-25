#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sd_refine.py — SD 局部重绘精修，让贴上去的标牌与场景光影真正融合

思路来自用户提供的方案（step1 擦除 → step2 贴牌 → step3 SD 精修）：

  纯贴图的问题：素材是"干净的"，场景是"脏的"——光照方向、金属反光、
  空气散射、传感器噪声都对不上，所以一眼看出是贴的。

  SD 局部重绘能解决这件事：只让模型重绘标牌区域，
  用极低的重绘幅度（strength 0.18~0.30），
  模型做的是"把我贴的牌按周围环境重新打光"，而不是"重新画一个牌"。
  幅度足够低时字形基本不变，但光影会跟着场景走。

关键参数（用户实测经验）：
  中近景大牌（>60px）  strength 0.25~0.30  光影融合好、字形稳
  远距离小牌（<40px）  strength 0.18~0.22  防止小数字被重绘糊

依赖：
  diffusers / transformers / accelerate / torch(CUDA)
  模型：stable-diffusion-v1-5/stable-diffusion-inpainting（fp16）

用法：
  python3 sd_refine.py --image a.jpg --mask m.png --out b.jpg --strength 0.25
"""
import argparse
import os
import sys


_MODEL_CACHE = {}


def load_pipeline(model_dir="/data/models/sd15-inpaint", device="cuda"):
    """加载 SD inpaint 管线（整个进程只加载一次）。"""
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device]
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    if not os.path.isdir(model_dir):
        raise FileNotFoundError("模型目录不存在: %s" % model_dir)

    # variant="fp16"：我们下载的是 *.fp16.safetensors，
    # 不指定 variant 的话 diffusers 会去找 model.safetensors 然后报找不到文件
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,          # 关掉安全检查器，省显存也省时间
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    # 省显存：注意力切分
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    _MODEL_CACHE[device] = pipe
    return pipe


def strength_for_size(sign_px):
    """按标牌尺寸给推荐的重绘幅度。

    用户实测：大牌可以重绘多一点（光影融合更好），
    小牌必须轻手（不然数字被重绘糊掉）。
    """
    if sign_px >= 60:
        return 0.28
    if sign_px >= 40:
        return 0.24
    return 0.20


def build_prompt(cls_name, country=""):
    """为交通标志构造正向提示词。

    不用具体的牌子描述（SD 1.5 对交通标志的细分类别理解有限），
    而是描述"物理材质 + 光照条件"，让模型专注在光影融合上。
    """
    base = ("a traffic sign, real photograph, natural lighting, "
            "consistent with surrounding scene, weathered metal surface, "
            "outdoor daylight, sharp details, no text artifacts")
    return base


NEGATIVE = ("blurry, low quality, distorted text, wrong symbols, cartoon, "
            "illustration, painting, watermark, extra signs, duplicate")


def refine(image_bgr, mask_gray, strength=None, sign_px=None,
           model_dir="/data/models/sd15-inpaint",
           prompt=None, seed=0, padding=32):
    """对一块区域做 SD 局部重绘。

    image_bgr  完整的场景图（BGR）
    mask_gray  单通道掩码，255 = 需要重绘的区域（标牌+少量边距）
    strength   重绘幅度；None 时按 sign_px 自动选
    sign_px    标牌短边像素（用于自动选 strength）

    返回 (refined_bgr, info)
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    if mask_gray.max() == 0:
        return image_bgr, {"ok": False, "reason": "empty_mask"}

    # 自动选 strength
    if strength is None:
        strength = strength_for_size(sign_px or 48)

    # 裁出掩码的包围盒，向外扩 padding（SD 需要一点上下文才能判断光照方向）
    ys, xs = np.where(mask_gray > 127)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    H, W = image_bgr.shape[:2]
    cx1 = max(0, x1 - padding)
    cy1 = max(0, y1 - padding)
    cx2 = min(W, x2 + padding)
    cy2 = min(H, y2 + padding)

    crop = image_bgr[cy1:cy2, cx1:cx2]
    crop_mask = mask_gray[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]

    # SD 要求尺寸是 8 的倍数，且不能太小
    tgt_w = max(256, int(round(cw / 8.0)) * 8)
    tgt_h = max(256, int(round(ch / 8.0)) * 8)
    # 限制最大边长，防止显存爆
    mx = 768
    if max(tgt_w, tgt_h) > mx:
        sc = mx / float(max(tgt_w, tgt_h))
        tgt_w = max(256, int(tgt_w * sc / 8) * 8)
        tgt_h = max(256, int(tgt_h * sc / 8) * 8)

    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((tgt_w, tgt_h), Image.LANCZOS)
    mask_pil = Image.fromarray(crop_mask).resize((tgt_w, tgt_h), Image.NEAREST)

    pipe = load_pipeline(model_dir)
    gen = torch.Generator(device="cuda").manual_seed(int(seed))
    with torch.autocast("cuda"):
        res = pipe(
            prompt=prompt or build_prompt(""),
            negative_prompt=NEGATIVE,
            image=img_pil,
            mask_image=mask_pil,
            strength=float(strength),
            guidance_scale=6.0,
            num_inference_steps=28,
            generator=gen,
        ).images[0]

    # 缩回原尺寸
    refined_crop = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
    refined_crop = cv2.resize(refined_crop, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

    # 只把掩码内的部分贴回（掩码外保持原样，避免裁切边界出现接缝）
    out = image_bgr.copy()
    roi = out[cy1:cy2, cx1:cx2]
    soft = cv2.GaussianBlur(crop_mask.astype(np.float32) / 255.0, (5, 5), 1.5)
    soft3 = soft[:, :, None]
    roi[:] = refined_crop.astype(np.float32) * soft3 + roi.astype(np.float32) * (1 - soft3)
    out[cy1:cy2, cx1:cx2] = np.clip(roi, 0, 255).astype(np.uint8)

    return out, {"ok": True, "strength": round(float(strength), 2),
                 "size": [int(cw), int(ch)], "sd_size": [tgt_w, tgt_h]}


def make_sign_mask(image_shape, bbox, pad_ratio=0.06):
    """按检测框生成标牌掩码（矩形 + 少量外扩）。"""
    import numpy as np

    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * pad_ratio)
    m = np.zeros((h, w), np.uint8)
    m[max(0, y1 - pad):min(h, y2 + pad),
      max(0, x1 - pad):min(w, x2 + pad)] = 255
    return m


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--sign-px", type=int, default=48)
    ap.add_argument("--model", default="/data/models/sd15-inpaint")
    a = ap.parse_args()

    im = cv2.imread(a.image)
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if im is None or mask is None:
        print("读图失败")
        return 2

    out, info = refine(im, mask, strength=a.strength, sign_px=a.sign_px,
                       model_dir=a.model)
    cv2.imwrite(a.out, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
