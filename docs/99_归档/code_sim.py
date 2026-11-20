import cv2
import torch
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline


class SceneFusionTSRSimulator:
    def __init__(
        self,
        inpaint_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0",
        device: str = "cuda"
    ):
        self.device = device
        self.lama = SimpleLama()

        # 加载 Canny 结构锁 + SDXL 局部重绘管线
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.float16, variant="fp16"
        )
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            inpaint_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        # 自动显存调度，12GB 显存可稳定运行
        self.pipe.enable_model_cpu_offload()

    def step1_erase_background(self, bg_bgr: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
        """Step 1: 外扩 Mask 并用 LaMa 彻底擦除原牌，消除跨形状残影"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated_mask = cv2.dilate(bg_mask, kernel, iterations=1)

        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
        erased_pil = self.lama(Image.fromarray(bg_rgb), Image.fromarray(dilated_mask))
        return cv2.cvtColor(np.array(erased_pil), cv2.COLOR_RGB2BGR)

    def step2_coarse_paste(
        self,
        clean_bg: np.ndarray,
        orig_bg: np.ndarray,
        bg_mask: np.ndarray,
        asset_rgba: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, tuple]:
        """Step 2: 水平端正贴图 + 基础亮度对齐（定死位置与比例）"""
        contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_cnt)

        resized_asset = cv2.resize(asset_rgba, (w, h), interpolation=cv2.INTER_AREA)
        asset_bgr = resized_asset[:, :, :3]
        asset_alpha = resized_asset[:, :, 3]

        # 基础 LAB 亮度对齐，给扩散模型一个接近原图动态范围的起点
        orig_roi = orig_bg[y:y+h, x:x+w]
        valid = asset_alpha > 127
        if valid.sum() > 10:
            bg_lab = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
            as_lab = cv2.cvtColor(asset_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            p5_bg, p95_bg = np.percentile(bg_lab[:, :, 0][valid], [5, 95])
            p5_as, p95_as = np.percentile(as_lab[:, :, 0][valid], [5, 95])
            scale = (p95_bg - p5_bg) / (p95_as - p5_as + 1e-5)
            as_lab[:, :, 0] = np.clip((as_lab[:, :, 0] - p5_as) * scale + p5_bg, 0, 255)
            asset_bgr = cv2.cvtColor(as_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        coarse_img = clean_bg.copy()
        alpha_3c = (asset_alpha.astype(np.float32) / 255.0)[:, :, None]
        roi_clean = coarse_img[y:y+h, x:x+w].astype(np.float32)
        coarse_img[y:y+h, x:x+w] = np.clip(
            asset_bgr * alpha_3c + roi_clean * (1.0 - alpha_3c), 0, 255
        ).astype(np.uint8)

        new_mask = np.zeros(clean_bg.shape[:2], dtype=np.uint8)
        new_mask[y:y+h, x:x+w] = asset_alpha
        return coarse_img, new_mask, (x, y, w, h)

    def _restore_camera_isp_noise(
        self,
        orig_patch: np.ndarray,
        refined_patch: np.ndarray,
        mask_patch: np.ndarray,
        noise_weight: float = 0.85
    ) -> np.ndarray:
        """Step 4: 提取原图邻近背景的高频 CMOS 噪点，回贴到生成区域消除 AI 干净感"""
        bg_valid = mask_patch < 127
        if bg_valid.sum() < 50:
            return refined_patch

        # 提取背景高频残差标准差
        smooth_bg = cv2.medianBlur(orig_patch, 3)
        residual = (orig_patch.astype(np.float32) - smooth_bg.astype(np.float32))[bg_valid]
        noise_std = float(np.std(residual)) * noise_weight

        if noise_std > 0.8:
            noise = np.random.normal(0, noise_std, refined_patch.shape).astype(np.float32)
            refined_patch = np.clip(refined_patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return refined_patch

    def step3_and_4_scene_fusion(
        self,
        coarse_img: np.ndarray,
        orig_bg: np.ndarray,
        new_mask: np.ndarray,
        bbox: tuple,
        strength: float = 0.48,              # 高重绘幅度：重构方向光、反光与空气透视
        canny_scale: float = 0.72,           # 强结构锁：锁死数字、文字和边框不形变
        crop_scale: float = 4.0,             # 扩大上下文视野，让模型读懂环境光场
        seed: int = 42
    ) -> np.ndarray:
        img_h, img_w = coarse_img.shape[:2]
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2

        side = max(int(max(w, h) * crop_scale), 192)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(img_w, x1 + side)
        y2 = min(img_h, y1 + side)
        crop_w, crop_h = x2 - x1, y2 - y1

        patch_bgr = coarse_img[y1:y2, x1:x2]
        orig_patch = orig_bg[y1:y2, x1:x2]
        patch_mask = new_mask[y1:y2, x1:x2]

        # 外扩精修 Mask，让牌面边缘与背景光影自然融合
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        inpaint_mask = cv2.dilate(patch_mask, k, iterations=1)

        target_res = 768
        resized_bgr = cv2.resize(patch_bgr, (target_res, target_res), interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(inpaint_mask, (target_res, target_res), interpolation=cv2.INTER_NEAREST)

        # 提取 Canny 边缘图作为强结构锁
        canny_edges = cv2.Canny(resized_bgr, 80, 180)
        canny_pil = Image.fromarray(cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB))

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

        generator = torch.Generator(device=self.device).manual_seed(seed)
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
            generator=generator
        ).images[0]

        refined_bgr = cv2.cvtColor(
            np.array(refined_pil.resize((crop_w, crop_h), Image.LANCZOS)),
            cv2.COLOR_RGB2BGR
        )

        # Step 4: 回贴相机 ISP 噪点 + 软羽化贴回原图
        refined_bgr = self._restore_camera_isp_noise(orig_patch, refined_bgr, inpaint_mask)

        soft_mask = cv2.GaussianBlur(inpaint_mask, (11, 11), 0).astype(np.float32) / 255.0
        soft_mask_3c = soft_mask[:, :, None]
        blended_patch = np.clip(
            refined_bgr * soft_mask_3c + patch_bgr * (1.0 - soft_mask_3c), 0, 255
        ).astype(np.uint8)

        final_img = coarse_img.copy()
        final_img[y1:y2, x1:x2] = blended_patch
        return final_img

    def run(
        self,
        bg_bgr: np.ndarray,
        bg_mask: np.ndarray,
        asset_rgba: np.ndarray,
        strength: float = 0.48,
        canny_scale: float = 0.72
    ) -> np.ndarray:
        clean_bg = self.step1_erase_background(bg_bgr, bg_mask)
        coarse_img, new_mask, bbox = self.step2_coarse_paste(clean_bg, bg_bgr, bg_mask, asset_rgba)
        return self.step3_and_4_scene_fusion(
            coarse_img, bg_bgr, new_mask, bbox, strength=strength, canny_scale=canny_scale
        )


if __name__ == "__main__":
    sim = SceneFusionTSRSimulator()

    bg = cv2.imread("road_sample.jpg")
    mask = cv2.imread("sign_mask.png", cv2.IMREAD_GRAYSCALE)
    asset = cv2.imread("stop_sign.png", cv2.IMREAD_UNCHANGED)

    if bg is not None and mask is not None and asset is not None:
        out = sim.run(bg, mask, asset, strength=0.48, canny_scale=0.72)
        cv2.imwrite("scene_fusion_output.jpg", out)
        print("完成，已保存 scene_fusion_output.jpg")