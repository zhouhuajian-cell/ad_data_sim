import cv2
import random
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable


class SignRealismV4Renderer:
    """
    V4.2 整合版渲染器：
    1. 自动裁切素材透明留白 + LAB-A 红圈精定位（解决贴大/错位）
    2. 物理空气光/雾霾成像映射（彻底根治 255 白底发亮）
    3. 远景小目标分辨极限降采样 + 温和模糊/加噪
    4. 模糊后闭环校准（补偿红圈二次掉色，消除 white_v_diff 与 chroma_s_diff 漂移）
    """
    def __init__(
        self,
        jpeg_quality_range: Tuple[int, int] = (54, 68),
        pad_ratio: float = 1.01,
        fallback_shrink: float = 0.92
    ):
        self.jpeg_quality_range = jpeg_quality_range
        self.pad_ratio = pad_ratio
        self.fallback_shrink = fallback_shrink

    @staticmethod
    def _trim_asset_alpha(asset_rgba: np.ndarray, alpha_thresh: int = 15) -> np.ndarray:
        """Step 1: 裁掉素材 PNG 四周多余透明留白，防止缩放尺寸偏大"""
        if asset_rgba.ndim != 3 or asset_rgba.shape[2] != 4:
            return asset_rgba
        ys, xs = np.where(asset_rgba[:, :, 3] > alpha_thresh)
        if len(xs) == 0:
            return asset_rgba
        return asset_rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def refine_geometry_bbox(
        self,
        bg_img: np.ndarray,
        yolo_bbox: Tuple[int, int, int, int]
    ) -> Tuple[Tuple[int, int, int, int], float]:
        """Step 2: LAB-A 通道红圈精定位（兼容阴天暗红，抗绿色山坡/灰白板干扰）"""
        h_img, w_img = bg_img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in yolo_bbox]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)

        sx1, sy1 = max(0, x1 - int(bw * 0.12)), max(0, y1 - int(bh * 0.12))
        sx2, sy2 = min(w_img, x2 + int(bw * 0.12)), min(h_img, y2 + int(bh * 0.12))
        roi = bg_img[sy1:sy2, sx1:sx2]

        if roi.size > 0:
            # LAB-A 通道：中性灰=128，绿色<128，阴天暗红依然 >133
            lab_a = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32)
            thresh = max(134.0, float(np.mean(lab_a) + 1.2 * np.std(lab_a)))
            red_mask = (lab_a >= thresh).astype(np.uint8) * 255

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_cnt = max(contours, key=cv2.contourArea)
                rx, ry, rw, rh = cv2.boundingRect(main_cnt)
                if bw * 0.55 <= rw <= bw * 1.15 and bh * 0.55 <= rh <= bh * 1.15:
                    cx, cy = sx1 + rx + rw / 2.0, sy1 + ry + rh / 2.0
                    nw, nh = max(4.0, rw * self.pad_ratio), max(4.0, rh * self.pad_ratio)
                    nx1 = max(0, int(round(cx - nw / 2.0)))
                    ny1 = max(0, int(round(cy - nh / 2.0)))
                    nx2 = min(w_img, max(nx1 + 4, int(round(cx + nw / 2.0))))
                    ny2 = min(h_img, max(ny1 + 4, int(round(cy + nh / 2.0))))
                    return (nx1, ny1, nx2, ny2), (nx2 - nx1) / float(max(1, ny2 - ny1))

        # 找不到高置信度红圈时，对宽松 YOLO 框做微缩而非放大
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        nw, nh = max(4.0, bw * self.fallback_shrink), max(4.0, bh * self.fallback_shrink)
        nx1 = max(0, int(round(cx - nw / 2.0)))
        ny1 = max(0, int(round(cy - nh / 2.0)))
        nx2 = min(w_img, max(nx1 + 4, int(round(cx + nw / 2.0))))
        ny2 = min(h_img, max(ny1 + 4, int(round(cy + nh / 2.0))))
        return (nx1, ny1, nx2, ny2), (nx2 - nx1) / float(max(1, ny2 - ny1))

    def align_color_physical_haze(
        self,
        bg_roi: np.ndarray,
        asset_bgr: np.ndarray,
        asset_alpha: np.ndarray
    ) -> np.ndarray:
        """Step 3: 物理空气光/雾霾模型 + LAB-A 红度对齐（数学上杜绝白底拉爆）"""
        valid = asset_alpha > 127
        if valid.sum() < 16:
            return asset_bgr

        # 向内腐蚀掩码取核心区，避开边缘天空/树叶污染
        h, w = asset_alpha.shape[:2]
        erode_px = max(1, int(min(h, w) * 0.12))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        inner = cv2.erode((valid.astype(np.uint8) * 255), k) > 127
        sample_mask = inner if inner.sum() >= 15 else valid

        bg_lab = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_l = bg_lab[:, :, 0]
        bg_a = bg_lab[:, :, 1]

        # 1. 原牌真实白底锚点 W_bg（排除红像素，取亮度 P70~P92 中位数）
        non_red = sample_mask & (bg_a < 138)
        w_pool = non_red if non_red.sum() >= 10 else sample_mask
        p70, p92 = np.percentile(bg_l[w_pool], [70, 92])
        w_sel = w_pool & (bg_l >= p70) & (bg_l <= p92)
        W_bg = (
            np.median(bg_roi[w_sel], axis=0).astype(np.float32)
            if w_sel.sum() > 0
            else np.median(bg_roi[w_pool], axis=0).astype(np.float32)
        )

        # 2. 原牌真实黑字/空气灰雾底噪锚点 K_bg（取亮度 P5~P20 中位数）
        p5, p20 = np.percentile(bg_l[sample_mask], [5, 20])
        k_sel = sample_mask & (bg_l >= p5) & (bg_l <= p20)
        K_bg = (
            np.median(bg_roi[k_sel], axis=0).astype(np.float32)
            if k_sel.sum() > 0
            else np.percentile(bg_roi[sample_mask], 10, axis=0).astype(np.float32)
        )

        # 硬性安全钳位：确保纯白映射绝不超过原图 95 分位
        max_bg_bgr = np.percentile(bg_roi[valid], 95, axis=0).astype(np.float32)
        W_bg = np.minimum(W_bg, max_bg_bgr)
        K_bg = np.clip(np.minimum(K_bg, W_bg - 15.0), 0.0, 240.0)

        # 3. 物理成像映射：I_out = K_bg + (Asset / 255) * (W_bg - K_bg)
        norm_asset = asset_bgr.astype(np.float32) / 255.0
        out_bgr = np.clip(K_bg + norm_asset * (W_bg - K_bg), 0, 255).astype(np.uint8)

        # 4. LAB-A 通道独立微调红度（不改动亮度 L，彻底兼容阴天暗红）
        as_lab = cv2.cvtColor(asset_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        out_lab = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        bg_red = valid & (bg_a > 133)
        as_red = valid & (as_lab[:, :, 1] > 145)
        if bg_red.sum() >= 6 and as_red.sum() >= 6:
            bg_red_delta = float(np.percentile(bg_a[bg_red], 80)) - 128.0
            out_red_delta = float(np.percentile(out_lab[:, :, 1][as_red], 80)) - 128.0
            if out_red_delta > 2.0 and bg_red_delta > 2.0:
                ratio = float(np.clip(bg_red_delta / out_red_delta, 0.45, 1.35))
                w_red = np.clip((as_lab[:, :, 1] - 135.0) / 25.0, 0.0, 1.0)
                new_a = 128.0 + (out_lab[:, :, 1] - 128.0) * ratio
                out_lab[:, :, 1] = np.clip(out_lab[:, :, 1] * (1.0 - w_red) + new_a * w_red, 0, 255)
                out_bgr = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return out_bgr

    @staticmethod
    def calibrate_post_degradation(
        bg_roi: np.ndarray,
        degraded_bgr: np.ndarray,
        clean_asset_bgr: np.ndarray,
        asset_alpha: np.ndarray
    ) -> np.ndarray:
        """
        Step 5: 模糊/加噪后的闭环微调（直接消除模糊导致的二次掉色与白底中位偏移）
        注：使用 clean_asset_bgr 锁定彩色区域位置，防止阴天低饱和压制后找不到红圈
        """
        valid = asset_alpha > 127
        if valid.sum() < 20:
            return degraded_bgr

        bg_hsv = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        sim_hsv = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        clean_hsv = cv2.cvtColor(clean_asset_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 用干净素材的饱和度生成软材质权重，边缘过渡绝不产生色带
        w_chroma = np.clip((clean_hsv[:, :, 1] - 35.0) / 45.0, 0.0, 1.0)

        # 1. 白底亮度中位闭环对齐（对齐口径与 AutoQCGatesV4 完全一致）
        bg_w = valid & (bg_hsv[:, :, 1] < 45) & (bg_hsv[:, :, 2] > 90)
        sim_w = valid & (sim_hsv[:, :, 1] < 45) & (sim_hsv[:, :, 2] > 90)
        if bg_w.sum() >= 10 and sim_w.sum() >= 10:
            bg_v_med = float(np.median(bg_hsv[:, :, 2][bg_w]))
            sim_v_med = float(np.median(sim_hsv[:, :, 2][sim_w]))
            v_gain = float(np.clip(bg_v_med / (sim_v_med + 1e-5), 0.72, 1.28))
            sim_hsv[:, :, 2] = np.clip(
                sim_hsv[:, :, 2] * (1.0 + (v_gain - 1.0) * (1.0 - w_chroma)), 0, 255
            )

        # 2. 彩区 P75 饱和度闭环对齐（兜底阴天低饱和红圈，消灭极端离群值）
        bg_c = valid & (bg_hsv[:, :, 1] >= 50)
        if bg_c.sum() < 10:
            bg_c = valid & (bg_hsv[:, :, 1] >= 20)

        # 优先取当前图里的彩色像素，若被压暗则回退到干净素材的彩色掩码
        sim_c = valid & (sim_hsv[:, :, 1] >= 50)
        if sim_c.sum() < 10:
            sim_c = valid & (clean_hsv[:, :, 1] >= 60)

        if bg_c.sum() >= 6 and sim_c.sum() >= 6:
            bg_s75 = float(np.percentile(bg_hsv[:, :, 1][bg_c], 75))
            sim_s75 = float(np.percentile(sim_hsv[:, :, 1][sim_c], 75))
            s_gain = float(np.clip(bg_s75 / (sim_s75 + 1e-5), 0.25, 1.55))
            sim_hsv[:, :, 1] = np.clip(
                sim_hsv[:, :, 1] * (1.0 + (s_gain - 1.0) * w_chroma), 0, 255
            )

        return cv2.cvtColor(sim_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def match_degradation_and_blend(
        self,
        bg_roi: np.ndarray,
        aligned_bgr: np.ndarray,
        clean_asset_bgr: np.ndarray,
        asset_alpha: np.ndarray
    ) -> np.ndarray:
        """Step 4~6: 小目标分辨极限 -> 温和模糊 -> 加噪 -> 闭环校准 -> 极窄融合 -> JPEG 伪影"""
        valid = asset_alpha > 127
        h, w = bg_roi.shape[:2]
        out = aligned_bgr.copy()

        # 1. 远景小牌 (<35px) 模拟镜头物理分辨极限
        if min(h, w) < 35:
            dw, dh = max(4, int(round(w * 0.75))), max(4, int(round(h * 0.75)))
            out = cv2.resize(out, (dw, dh), interpolation=cv2.INTER_AREA)
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

        # 2. 温和模糊对齐
        bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
        bg_lap = float(np.var(cv2.Laplacian(bg_gray, cv2.CV_64F)[valid])) if valid.sum() > 10 else 100.0
        base_for_blur = out.copy()
        for sigma in np.linspace(0.4, 2.0, 9):
            cur_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            if float(np.var(cv2.Laplacian(cur_gray, cv2.CV_64F)[valid])) <= bg_lap * 1.25:
                break
            ksize = int(2 * round(2 * sigma) + 1)
            out = cv2.GaussianBlur(base_for_blur, (ksize, ksize), sigmaX=sigma)

        # 3. 匹配原图高频噪点
        if valid.sum() > 10:
            residual = (bg_roi.astype(np.float32) - cv2.medianBlur(bg_roi, 3).astype(np.float32))[valid]
            noise_std = float(np.std(residual)) * 0.75
            if noise_std > 0.8:
                out = np.clip(
                    out.astype(np.float32) + np.random.normal(0, noise_std, out.shape), 0, 255
                ).astype(np.uint8)

        # 4. 模糊后闭环校准（核心：拉齐白底中位与红圈 P75 饱和度）
        out = self.calibrate_post_degradation(bg_roi, out, clean_asset_bgr, asset_alpha)

        # 5. 极窄羽化融合（不腐蚀，完全盖死原红边）
        soft_a = (cv2.GaussianBlur(asset_alpha, (3, 3), 0).astype(np.float32) / 255.0)[:, :, None]
        blended = np.clip(
            out.astype(np.float32) * soft_a + bg_roi.astype(np.float32) * (1.0 - soft_a), 0, 255
        ).astype(np.uint8)

        # 6. 车载低码率 JPEG 压缩伪影（仅作用于牌面及外扩 2px）
        q = random.randint(*self.jpeg_quality_range)
        ok, buf = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            comp = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            m = (cv2.GaussianBlur(cv2.dilate(asset_alpha, k), (3, 3), 0).astype(np.float32) / 255.0)[:, :, None]
            blended = np.clip(
                comp.astype(np.float32) * m + blended.astype(np.float32) * (1.0 - m), 0, 255
            ).astype(np.uint8)

        return blended

    def render(
        self,
        bg_img: np.ndarray,
        yolo_bbox: Tuple[int, int, int, int],
        asset_rgba: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray, np.ndarray]:
        """保持与 V4 完全一致的调用签名，可直接无缝替换"""
        trimmed_asset = self._trim_asset_alpha(asset_rgba)
        (nx1, ny1, nx2, ny2), _ = self.refine_geometry_bbox(bg_img, yolo_bbox)
        nw, nh = max(4, nx2 - nx1), max(4, ny2 - ny1)

        resized = cv2.resize(trimmed_asset, (nw, nh), interpolation=cv2.INTER_AREA)
        asset_bgr, asset_alpha = resized[:, :, :3], resized[:, :, 3]

        bg_roi = bg_img[ny1:ny2, nx1:nx2].copy()
        aligned_bgr = self.align_color_physical_haze(bg_roi, asset_bgr, asset_alpha)
        sim_roi = self.match_degradation_and_blend(bg_roi, aligned_bgr, asset_bgr, asset_alpha)

        sim_img = bg_img.copy()
        sim_img[ny1:ny2, nx1:nx2] = sim_roi
        return sim_img, (nx1, ny1, nx2, ny2), bg_roi, sim_roi


class AutoQCGatesV4:
    """保持与上一轮完全一致的统计口径，确保 30 样本 A/B 对比绝对公平"""
    def __init__(
        self,
        max_white_l_diff: float = 12.0,
        max_chroma_s_diff: float = 22.0,
        min_hard_case_h: int = 20,
        orig_conf_thresh: float = 0.50
    ):
        self.max_white_l_diff = max_white_l_diff
        self.max_chroma_s_diff = max_chroma_s_diff
        self.min_hard_case_h = min_hard_case_h
        self.orig_conf_thresh = orig_conf_thresh

    def check_physical_gates(self, orig_roi: np.ndarray, sim_roi: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        orig_hsv = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        sim_hsv = cv2.cvtColor(sim_roi, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 1. 仅白底区域亮度中位差
        orig_w = (orig_hsv[:, :, 1] < 45) & (orig_hsv[:, :, 2] > 90)
        sim_w = (sim_hsv[:, :, 1] < 45) & (sim_hsv[:, :, 2] > 90)
        white_v_diff = 0.0
        if orig_w.sum() > 10 and sim_w.sum() > 10:
            white_v_diff = abs(float(np.median(sim_hsv[:, :, 2][sim_w]) - np.median(orig_hsv[:, :, 2][orig_w])))

        # 2. 仅彩色区域 P75 饱和度差
        orig_c = orig_hsv[:, :, 1] >= 50
        sim_c = sim_hsv[:, :, 1] >= 50
        chroma_s_diff = 0.0
        if orig_c.sum() > 10 and sim_c.sum() > 10:
            chroma_s_diff = abs(
                float(np.percentile(sim_hsv[:, :, 1][sim_c], 75) - np.percentile(orig_hsv[:, :, 1][orig_c], 75))
            )

        passed = (white_v_diff <= self.max_white_l_diff) and (chroma_s_diff <= self.max_chroma_s_diff)
        return passed, {
            "white_v_diff": round(white_v_diff, 2),
            "chroma_s_diff": round(chroma_s_diff, 2),
            "phys_passed": passed
        }

    def check_gate4_paired_delta(
        self,
        orig_roi: np.ndarray,
        sim_roi: np.ndarray,
        target_cls: str,
        phys_passed: bool,
        teacher_infer_fn: Callable[[np.ndarray], Tuple[Optional[str], float]]
    ) -> Tuple[bool, str]:
        _, orig_conf = teacher_infer_fn(orig_roi)
        sim_pred, sim_conf = teacher_infer_fn(sim_roi)
        sim_ok = (sim_pred == target_cls) and (sim_conf >= self.orig_conf_thresh)

        if orig_conf >= self.orig_conf_thresh:
            if sim_ok:
                return True, "PASS_Normal"
            return False, f"REJECT_SimDegraded(orig={orig_conf:.2f}, sim={sim_conf:.2f}, pred={sim_pred})"

        if sim_ok:
            return True, "PASS_HardCase_Detected"
        if phys_passed and sim_roi.shape[0] >= self.min_hard_case_h:
            return True, "PASS_HardCase_PhysOK"

        return False, "REJECT_HardCase_PhysFail"