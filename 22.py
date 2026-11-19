import cv2
import numpy as np


def align_photometry_v3_patch(
    bg_roi: np.ndarray,
    asset_roi: np.ndarray,
    mask_roi: np.ndarray,
    ab_weight: float = 0.35,
    sat_cap_ratio: float = 1.05
) -> np.ndarray:
    """
    补丁1：LAB P10~P90 动态范围压扁 + HSV 饱和度强制压制
    解决阴天/灰雾场景下“白底过亮、黑字死黑、红圈太鲜艳”的问题
    """
    valid = mask_roi > 127
    if valid.sum() < 10:
        return asset_roi

    # --- 1. LAB 空间：P10~P90 亮度映射（压扁对比度，融入空气灰雾）---
    bg_lab = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    as_lab = cv2.cvtColor(asset_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_l = bg_lab[:, :, 0][valid]
    as_l = as_lab[:, :, 0][valid]
    p10_bg, p90_bg = np.percentile(bg_l, [10, 90])
    p10_as, p90_as = np.percentile(as_l, [10, 90])

    scale_l = (p90_bg - p10_bg) / (p90_as - p10_as + 1e-5)
    as_lab[:, :, 0] = np.clip((as_lab[:, :, 0] - p10_as) * scale_l + p10_bg, 0, 255)

    # A/B 通道色温软迁移
    for ch in (1, 2):
        diff = np.mean(bg_lab[:, :, ch][valid]) - np.mean(as_lab[:, :, ch][valid])
        as_lab[:, :, ch] = np.clip(as_lab[:, :, ch] + diff * ab_weight, 0, 255)

    out_bgr = cv2.cvtColor(as_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # --- 2. HSV 空间：S 通道（饱和度）强制对齐（把鲜红圈压成实拍脏红）---
    bg_hsv = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    as_hsv = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 用 P90 饱和度代表红圈等高饱和区域的上限，避免被白底拉偏
    bg_s90 = np.percentile(bg_hsv[:, :, 1][valid], 90)
    as_s90 = np.percentile(as_hsv[:, :, 1][valid], 90)

    if as_s90 > 1e-3:
        # 只压降不强行提饱和，防止原图发灰时把素材拉爆
        sat_scale = min(1.0, (bg_s90 * sat_cap_ratio) / as_s90)
        as_hsv[:, :, 1] = np.clip(as_hsv[:, :, 1] * sat_scale, 0, 255)

    return cv2.cvtColor(as_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_dashcam_jpeg_artifact(
    blended_roi: np.ndarray,
    mask_roi: np.ndarray,
    quality: int = 58
) -> np.ndarray:
    """
    补丁2：模拟车载低码率视频压缩块与边缘涂抹感
    在 Alpha 融合之后、贴回整图之前调用
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc_buf = cv2.imencode(".jpg", blended_roi, encode_param)
    if not ok:
        return blended_roi

    compressed = cv2.imdecode(enc_buf, cv2.IMREAD_COLOR)

    # 仅在牌面及外扩 2px 区域内应用压缩结果，避免二次压缩污染周围真实背景
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask_roi, k, iterations=1)
    soft_m = (cv2.GaussianBlur(dilated, (3, 3), 0).astype(np.float32) / 255.0)[:, :, None]

    out = compressed.astype(np.float32) * soft_m + blended_roi.astype(np.float32) * (1.0 - soft_m)
    return np.clip(out, 0, 255).astype(np.uint8)