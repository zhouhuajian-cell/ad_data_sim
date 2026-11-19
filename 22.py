import cv2
import numpy as np


def refine_red_circle_bbox(bg_img: np.ndarray, yolo_bbox: tuple, pad_ratio: float = 1.04) -> tuple:
    """
    红圈精定位：在 YOLO 框内找真实红圈的外接框，消除 YOLO 偏移和原红边残留
    yolo_bbox: (x1, y1, x2, y2)
    """
    h_img, w_img = bg_img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in yolo_bbox]
    bw, bh = x2 - x1, y2 - y1

    # 稍微外扩搜索区，防止 YOLO 框把红圈切掉了一点
    sx1, sy1 = max(0, x1 - int(bw * 0.15)), max(0, y1 - int(bh * 0.15))
    sx2, sy2 = min(w_img, x2 + int(bw * 0.15)), min(h_img, y2 + int(bh * 0.15))
    roi = bg_img[sy1:sy2, sx1:sx2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 宽容度较高的双区间红色掩码（覆盖阴天暗红和晴天亮红）
    m1 = cv2.inRange(hsv, np.array([0, 35, 30]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 35, 30]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(m1, m2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (x1, y1, x2, y2)

    main_cnt = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(main_cnt)

    # 合理性校验：如果找到的红圈太小（噪点），回退原 YOLO 框
    if rw < bw * 0.5 or rh < bh * 0.5:
        return (x1, y1, x2, y2)

    # 以真实红圈中心微扩 1.04x，刚好盖死原红边
    cx, cy = sx1 + rx + rw / 2.0, sy1 + ry + rh / 2.0
    nw, nh = rw * pad_ratio, rh * pad_ratio
    nx1, ny1 = max(0, int(cx - nw / 2.0)), max(0, int(cy - nh / 2.0))
    nx2, ny2 = min(w_img, int(cx + nw / 2.0)), min(h_img, int(cy + nh / 2.0))
    return (nx1, ny1, nx2, ny2)


def align_color_three_anchors(
    bg_roi: np.ndarray,
    asset_bgr: np.ndarray,
    asset_alpha: np.ndarray
) -> np.ndarray:
    """
    白/红/黑三锚点语义颜色对齐：
    彻底消除因红白面积比不同导致的“白底发青、亮度过高、红圈发光”
    """
    valid = asset_alpha > 127
    if valid.sum() < 20:
        return asset_bgr

    bg_hsv = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2HSV)
    as_hsv = cv2.cvtColor(asset_bgr, cv2.COLOR_BGR2HSV)

    # --- 1. 提取原图与素材的【红圈】掩码 ---
    def get_red_mask(hsv_img, s_min=45):
        m1 = (hsv_img[:, :, 0] <= 12) | (hsv_img[:, :, 0] >= 160)
        m2 = hsv_img[:, :, 1] >= s_min
        return m1 & m2 & valid

    bg_red = get_red_mask(bg_hsv, s_min=35)
    as_red = get_red_mask(as_hsv, s_min=60)

    # --- 2. 提取原图与素材的【白底】和【黑字】锚点 (BGR 空间直接线性映射，绝不偏色) ---
    bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
    as_gray = cv2.cvtColor(asset_bgr, cv2.COLOR_BGR2GRAY)

    # 白底：非红、低饱和、高亮度区域
    bg_non_red = valid & (~bg_red)
    as_non_red = valid & (~as_red)

    if bg_non_red.sum() > 10 and as_non_red.sum() > 10:
        bg_w_thresh = np.percentile(bg_gray[bg_non_red], 75)
        bg_k_thresh = np.percentile(bg_gray[bg_non_red], 15)
        as_w_thresh = np.percentile(as_gray[as_non_red], 75)
        as_k_thresh = np.percentile(as_gray[as_non_red], 15)

        W_bg = np.mean(bg_roi[bg_non_red & (bg_gray >= bg_w_thresh)], axis=0)
        K_bg = np.mean(bg_roi[bg_non_red & (bg_gray <= bg_k_thresh)], axis=0)
        W_as = np.mean(asset_bgr[as_non_red & (as_gray >= as_w_thresh)], axis=0)
        K_as = np.mean(asset_bgr[as_non_red & (as_gray <= as_k_thresh)], axis=0)

        # 对整张素材先做 BGR 三通道白/黑锚点映射（白底直接对齐原图暖灰白，黑字对齐原图深灰）
        scale = (W_bg - K_bg) / (W_as - K_as + 1e-5)
        out_f = (asset_bgr.astype(np.float32) - K_as) * scale + K_bg
        out_bgr = np.clip(out_f, 0, 255).astype(np.uint8)
    else:
        out_bgr = asset_bgr.copy()

    # --- 3. 单独对齐【红圈】的饱和度(S)与亮度(V) ---
    if bg_red.sum() > 10 and as_red.sum() > 10:
        out_hsv = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_r_s = np.mean(bg_hsv[:, :, 1][bg_red])
        bg_r_v = np.mean(bg_hsv[:, :, 2][bg_red])
        as_r_s = np.mean(out_hsv[:, :, 1][as_red])
        as_r_v = np.mean(out_hsv[:, :, 2][as_red])

        out_hsv[:, :, 1][as_red] = np.clip(out_hsv[:, :, 1][as_red] * (bg_r_s / (as_r_s + 1e-5)), 0, 255)
        out_hsv[:, :, 2][as_red] = np.clip(out_hsv[:, :, 2][as_red] * (bg_r_v / (as_r_v + 1e-5)), 0, 255)
        out_bgr = cv2.cvtColor(out_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out_bgr