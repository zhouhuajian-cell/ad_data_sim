import cv2
import numpy as np


def match_tsr_to_orig_roi(orig_roi_bgr, syn_sign_bgr, syn_alpha,
                          haze_blend=0.30, blur_sigma=1.1):
    """
    将新渲染的 TSR 标牌，按原图老标牌 ROI 的动态范围
    （黑位/白位/色温/饱和度）做物理对齐。

    orig_roi_bgr: 原图老标牌区域的裁剪图 (h, w, 3) uint8
    syn_sign_bgr: 新渲染的标牌图 (h, w, 3) uint8
    syn_alpha:    新标牌的透明通道 (h, w) float32, 0.0~1.0
    haze_blend:   雨雾/空气散射融合度 (晴天 0.05，阴雨雾天 0.15~0.25)
    blur_sigma:   镜头与距离模糊 (0.6~1.2)
    """
    # 尺寸对齐
    h, w = orig_roi_bgr.shape[:2]
    if syn_sign_bgr.shape[:2] != (h, w):
        syn_sign_bgr = cv2.resize(syn_sign_bgr, (w, h),
                                  interpolation=cv2.INTER_LINEAR)
    if syn_alpha.shape[:2] != (h, w):
        syn_alpha = cv2.resize(syn_alpha.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_LINEAR)

    mask = syn_alpha > 0.3
    if not np.any(mask):
        return orig_roi_bgr

    orig_lab = cv2.cvtColor(orig_roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    syn_lab = cv2.cvtColor(syn_sign_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 1. 核心：用原牌 ROI 的 [P5, P95] 锁死新牌的亮度范围
    #    （自动实现雨雾黑位抬升 + 白位压暗）
    orig_L = orig_lab[:, :, 0][mask]
    syn_L = syn_lab[:, :, 0][mask]

    orig_p5, orig_p95 = np.percentile(orig_L, [5, 95])
    syn_p5, syn_p95 = np.percentile(syn_L, [5, 95])

    # 线性重映射到原牌的真实对比度区间
    scale = (orig_p95 - orig_p5) / (syn_p95 - syn_p5 + 1e-5)
    # 限制拉伸倍数，防止原牌过暗时反向拉爆
    # HAZE-STRONG: 允许更强压缩（雨雾天对比度被严重压缩）
    scale = np.clip(scale, 0.15, 0.95)
    mapped_L = (syn_lab[:, :, 0] - syn_p5) * scale + orig_p5
    # HAZE-STRONG: 再向原牌均值收敛
    orig_mean_L = float(np.mean(orig_L))
    mapped_L = mapped_L * 0.7 + orig_mean_L * 0.3
    syn_lab[:, :, 0] = np.clip(mapped_L, orig_p5, orig_p95)

    # 2. 色温对齐 + 饱和度压降（解决红圈太鲜艳、白底偏纯白）
    for c in [1, 2]:  # A, B 通道
        orig_mean = np.mean(orig_lab[:, :, c][mask])
        # 往原牌环境色温偏移 35%，同时压低自身高饱和度
        # HAZE-STRONG: 饱和度压到 50%
        syn_lab[:, :, c] = (syn_lab[:, :, c] - 128.0) * 0.50 + 128.0
        syn_lab[:, :, c] = syn_lab[:, :, c] * 0.6 + orig_mean * 0.4

    matched_bgr = cv2.cvtColor(np.clip(syn_lab, 0, 255).astype(np.uint8),
                               cv2.COLOR_LAB2BGR).astype(np.float32)

    # 3. 叠加雨雾大气散射（用原 ROI 周围的灰白雾色轻罩一层）
    if haze_blend > 0:
        veil_color = np.percentile(
            orig_roi_bgr.reshape(-1, 3), 85, axis=0).astype(np.float32)
        matched_bgr = matched_bgr * (1.0 - haze_blend) + \
            veil_color * haze_blend

    # 4. 镜头光学与雨雾柔和度对齐
    if blur_sigma > 0:
        matched_bgr = cv2.GaussianBlur(matched_bgr, (0, 0),
                                       sigmaX=blur_sigma)
        syn_alpha = cv2.GaussianBlur(syn_alpha, (0, 0),
                                     sigmaX=blur_sigma * 0.7)

    # 5. 融合回原图 ROI
    alpha_3ch = np.clip(syn_alpha, 0.0, 1.0)[..., None]
    out = matched_bgr * alpha_3ch + \
        orig_roi_bgr.astype(np.float32) * (1.0 - alpha_3ch)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_to_tsr_result(orig_bgr, composed_bgr, bbox,
                        haze_blend=0.30, blur_sigma=1.1):
    """对整图结果做 ROI 局部对齐：只改 bbox 区域，其他不动。

    orig_bgr:     原图
    composed_bgr: 已合成（贴好新牌的图）
    bbox:         [x1, y1, x2, y2] 新牌位置
    返回：对齐后的整图
    """
    h, w = orig_bgr.shape[:2]
    x1 = max(0, int(round(bbox[0])) - 8)
    y1 = max(0, int(round(bbox[1])) - 8)
    x2 = min(w, int(round(bbox[2])) + 8)
    y2 = min(h, int(round(bbox[3])) + 8)
    if x2 <= x1 or y2 <= y1:
        return composed_bgr

    # ALPHA-BBOX-FULL: 整个 bbox 区域都要对齐（不要用差异阈值筛，会漏）
    roi_o = orig_bgr[y1:y2, x1:x2].copy()
    roi_c = composed_bgr[y1:y2, x1:x2].copy()

    # 用差异做软权重，但下限 0.5，保证整块都被处理
    diff = np.abs(roi_c.astype(np.int16) - roi_o.astype(np.int16)).max(axis=2)
    alpha = np.clip(diff.astype(np.float32) / 30.0, 0.5, 1.0)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    fixed = match_tsr_to_orig_roi(roi_o, roi_c, alpha,
                                  haze_blend=haze_blend,
                                  blur_sigma=blur_sigma)
    out = composed_bgr.copy()
    out[y1:y2, x1:x2] = fixed
    return out
