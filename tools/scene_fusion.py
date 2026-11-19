#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scene_fusion.py — SDXL + ControlNet 场景融合

方案来自 code_sim.py（用户提供），解决纯贴图的两个根本问题：

  1) 跨形状残影
     直接用掩码贴图时，原牌的深色描边、残角会从新牌边缘露出来。
     解决：先让 LaMa 把原牌彻底擦掉（掩码外扩后 inpaint），再往上贴。

  2) 光影不融合
     SD 1.5 上 strength 只能给到 0.20~0.24，再高就把数字重绘糊了，
     但这么低的幅度也几乎不做光影融合。
     解决：加 ControlNet Canny 锁住标牌的结构边缘，
     这样就能放心把 strength 提到 0.48 —— 模型可以大改光影，
     但数字、文字、边框被 Canny 锁死不会变形。

四步流水线：
  step1 LaMa 擦除原牌（外扩 11×11 核）
  step2 端正贴图 + LAB 黑白点亮度对齐
  step3 SDXL inpaint + ControlNet canny 精修（strength 0.48 / canny 0.72 / 上下文 4 倍）
  step4 回贴原图 ISP 噪点 + 软羽化贴回

依赖：
  simple-lama-inpainting（模型 196MB，自动下载到 ~/.cache/torch）
  diffusers + SDXL inpainting + controlnet-canny-sdxl-1.0（fp16，约 9.5GB）

用法：
  from scene_fusion import SceneFusionTSRSimulator
  sim = SceneFusionTSRSimulator()
  out = sim.run(bg_bgr, sign_mask, asset_rgba, strength=0.48, canny_scale=0.72)
"""
import os
import sys


class SceneFusionTSRSimulator:
    def __init__(
        self,
        inpaint_model="/data/models/sdxl-inpaint",
        controlnet_model="/data/models/controlnet-canny-sdxl",
        device="cuda",
        lama_dir=None,
    ):
        self.device = device
        self.inpaint_model = inpaint_model
        self.controlnet_model = controlnet_model

        import torch
        from simple_lama_inpainting import SimpleLama
        self._torch = torch
        self.lama = SimpleLama()          # 模型 196MB，首次自动下载

        from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.float16, variant="fp16"
        )
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            inpaint_model,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        # 12GB 显存跑 SDXL：用 cpu offload 把不用的模块换到内存
        self.pipe.enable_model_cpu_offload()
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def step1_erase_background(self, bg_bgr, bg_mask):
        """LaMa 彻底擦除原牌。

        掩码外扩 11×11：原牌的深色描边比检测框略大，
        不扩的话擦不干净，贴新牌时边缘会露出旧描边（跨形状残影）。
        """
        import cv2
        import numpy as np
        from PIL import Image

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(bg_mask, kernel, iterations=1)
        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
        erased = self.lama(Image.fromarray(bg_rgb), Image.fromarray(dilated))
        return cv2.cvtColor(np.array(erased), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    def step2_coarse_paste(self, clean_bg, orig_bg, bg_mask, asset_rgba):
        """端正贴图 + LAB 黑白点亮度对齐。

        亮度对齐是给 SD 一个"接近原图动态范围"的起点 ——
        如果贴上去就过曝，SD 在 strength 0.48 下会试图修正整体亮度，
        反而把标牌改花。
        """
        import cv2
        import numpy as np

        contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return clean_bg, np.zeros(bg_mask.shape, np.uint8), (0, 0, 1, 1)
        main_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_cnt)
        w = max(w, 2)
        h = max(h, 2)

        resized = cv2.resize(asset_rgba, (w, h), interpolation=cv2.INTER_AREA)
        asset_bgr = resized[:, :, :3]
        asset_alpha = resized[:, :, 3] if resized.shape[2] == 4 else np.full((h, w), 255, np.uint8)

        orig_roi = orig_bg[y:y + h, x:x + w]
        valid = asset_alpha > 127
        if valid.sum() > 10 and orig_roi.size:
            bg_lab = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
            as_lab = cv2.cvtColor(asset_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            p5_bg, p95_bg = np.percentile(bg_lab[:, :, 0][valid], [5, 95])
            p5_as, p95_as = np.percentile(as_lab[:, :, 0][valid], [5, 95])
            scale = (p95_bg - p5_bg) / (p95_as - p5_as + 1e-5)
            as_lab[:, :, 0] = np.clip((as_lab[:, :, 0] - p5_as) * scale + p5_bg, 0, 255)
            asset_bgr = cv2.cvtColor(as_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        coarse = clean_bg.copy()
        alpha3 = (asset_alpha.astype(np.float32) / 255.0)[:, :, None]
        roi = coarse[y:y + h, x:x + w].astype(np.float32)
        coarse[y:y + h, x:x + w] = np.clip(
            asset_bgr.astype(np.float32) * alpha3 + roi * (1.0 - alpha3), 0, 255
        ).astype(np.uint8)

        new_mask = np.zeros(clean_bg.shape[:2], np.uint8)
        new_mask[y:y + h, x:x + w] = asset_alpha
        return coarse, new_mask, (x, y, w, h)

    # ------------------------------------------------------------------
    def _restore_camera_isp_noise(self, orig_patch, refined_patch, mask_patch,
                                  noise_weight=0.85):
        """把原图背景的高频噪点回贴到生成区域。

        SD 生成的内容太"干净"，和周围带 CMOS 噪声的实拍放在一起会显得假。
        这里从掩码外的背景估噪点标准差，再叠加回去。
        """
        import cv2
        import numpy as np

        bg_valid = mask_patch < 127
        if bg_valid.sum() < 50:
            return refined_patch
        smooth_bg = cv2.medianBlur(orig_patch, 3)
        residual = (orig_patch.astype(np.float32) - smooth_bg.astype(np.float32))[bg_valid]
        noise_std = float(np.std(residual)) * noise_weight
        if noise_std > 0.8:
            noise = np.random.normal(0, noise_std, refined_patch.shape).astype(np.float32)
            refined_patch = np.clip(
                refined_patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return refined_patch

    # ------------------------------------------------------------------
    def step3_and_4_scene_fusion(self, coarse_img, orig_bg, new_mask, bbox,
                                 strength=0.48, canny_scale=0.72,
                                 crop_scale=4.0, seed=42):
        """SDXL + ControlNet 精修，然后回贴噪点、软羽化贴回原图。

        参数含义（用户实测）：
          strength    0.48  高重绘幅度 —— 让模型真的重构方向光/反光/空气透视
          canny_scale 0.72  强结构锁 —— 锁死数字、文字、边框不变形
          crop_scale  4.0   上下文扩大 4 倍 —— 让模型读懂环境光场
        """
        import cv2
        import numpy as np
        from PIL import Image

        img_h, img_w = coarse_img.shape[:2]
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2

        side = max(int(max(w, h) * crop_scale), 192)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(img_w, x1 + side)
        y2 = min(img_h, y1 + side)
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 8 or crop_h < 8:
            return coarse_img

        patch_bgr = coarse_img[y1:y2, x1:x2]
        orig_patch = orig_bg[y1:y2, x1:x2]
        patch_mask = new_mask[y1:y2, x1:x2]

        # 精修掩码外扩，让牌面边缘与背景光影自然过渡
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        inpaint_mask = cv2.dilate(patch_mask, k, iterations=1)

        target = 768
        resized_bgr = cv2.resize(patch_bgr, (target, target), interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(inpaint_mask, (target, target), interpolation=cv2.INTER_NEAREST)

        # Canny 结构锁：标牌的边缘（圆框、数字笔画）在 Canny 图上很清晰
        canny = cv2.Canny(resized_bgr, 80, 180)
        canny_pil = Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))

        pil_img = Image.fromarray(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(resized_mask)

        prompt = (
            "authentic dashcam photograph, traffic sign naturally integrated into the real outdoor scene, "
            "matching ambient sunlight direction, subtle weathering and metallic reflection, "
            "natural atmospheric haze, realistic lens softness and chromatic aberration"
        )
        neg_prompt = (
            "pasted sticker, overly sharp, glowing, artificial, cartoon, 3d render, "
            "distorted text, messy letters, clean digital graphic"
        )

        gen = self._torch.Generator(device="cuda").manual_seed(seed)
        refined_pil = self.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=pil_img,
            mask_image=pil_mask,
            control_image=canny_pil,
            strength=strength,
            controlnet_conditioning_scale=canny_scale,
            guidance_scale=5.5,
            num_inference_steps=30,
            generator=gen,
        ).images[0]

        refined_bgr = cv2.cvtColor(
            np.array(refined_pil.resize((crop_w, crop_h), Image.LANCZOS)),
            cv2.COLOR_RGB2BGR
        )

        # step4：回贴噪点 + 软羽化
        refined_bgr = self._restore_camera_isp_noise(orig_patch, refined_bgr, inpaint_mask)
        soft = cv2.GaussianBlur(inpaint_mask, (11, 11), 0).astype(np.float32) / 255.0
        soft3 = soft[:, :, None]
        blended = np.clip(
            refined_bgr.astype(np.float32) * soft3 +
            patch_bgr.astype(np.float32) * (1.0 - soft3), 0, 255
        ).astype(np.uint8)

        out = coarse_img.copy()
        out[y1:y2, x1:x2] = blended
        return out

    # ------------------------------------------------------------------
    def run(self, bg_bgr, bg_mask, asset_rgba, strength=0.48, canny_scale=0.72,
            skip_lama=False):
        """完整流水线。

        skip_lama=True 时跳过 LaMa 擦除（我们的素材是带 alpha 的，
        贴图时 alpha 已经盖住了原牌，多数情况不需要先擦）。
        """
        if skip_lama:
            clean = bg_bgr.copy()
        else:
            clean = self.step1_erase_background(bg_bgr, bg_mask)
        coarse, new_mask, bbox = self.step2_coarse_paste(clean, bg_bgr, bg_mask, asset_rgba)
        return self.step3_and_4_scene_fusion(
            coarse, bg_bgr, new_mask, bbox,
            strength=strength, canny_scale=canny_scale
        )


if __name__ == "__main__":
    print("SceneFusionTSRSimulator：LaMa 擦除 → 端正贴图 → SDXL+Canny 精修 → 噪点回贴")
