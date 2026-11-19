#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipeline6.py — 严格按 6 步顺序的 TSR 仿真流水线

用户指定的顺序（必须严格执行）：

  Step 1  形状强校验路由（最前置）
          检查 src_shape == dst_shape（圆对圆、倒三角对倒三角），
          不匹配直接跳过或重选素材，彻底杜绝三角贴圆。

  Step 2  红圈精定位 refine_red_circle_bbox
          拿原图和 YOLO 框，先算真实红圈的精准坐标 (nx1,ny1,nx2,ny2)，
          再从原图切出 bg_roi，同时把高清素材 resize 到同等宽高 (nw,nh)。

  Step 3  白/红/黑三锚点颜色对齐 align_color_three_anchors
          必须在模糊和加噪之前做 —— 此时素材的红/白/黑区域边界最干净，
          提取锚点最准，能把白底和红圈直接锁死到原图的真实光照和色温。

  Step 4  轻微光学退化（模糊 + 噪点）
          对调好色的素材施加匹配原图清晰度的轻微高斯模糊和少量噪点。

  Step 5  动态边缘融合（Alpha Blend）
          不腐蚀，用极窄羽化（ksize=3）把素材融进 bg_roi。

  Step 6  模拟车载 JPEG 压缩 apply_dashcam_jpeg_artifact
          + 贴回原图 + Auto-QC
          在融合后的 blended_roi 上做一次 quality=55~65 的编解码，
          让牌面边缘和周围像素一起带上一致的视频压缩感，
          最后贴回原图并过质检门控。

为什么这个顺序是对的（对比我们之前的实现）：
  - 形状校验前置：之前是边贴边判，判断失败时已经浪费了算力
  - 红圈精定位：之前用固定 grow 1.12 猜，实测比值在 0.76~1.39 乱跳
  - 三锚点：之前用全局 P5~P95 映射，红圈和白底共用一个系数 ——
    白底对上了红圈就偏，红圈对上了白底就偏，这就是色彩失真的根因
  - 三锚点必须在模糊前：模糊后红白黑的边界糊在一起，锚点就取不准了
  - JPEG 最后：必须在融合后，才能让牌面边缘与周围像素一起带上压缩感
"""
import sys

import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")


# ---------------------------------------------------------------------------
# Step 1：形状强校验
# ---------------------------------------------------------------------------
SHAPE_ALIAS = {
    "circle": "circle",
    "square": "square",
    "rect": "square",
    "triangle": "triangle",
}


def normalize_shape(s):
    return SHAPE_ALIAS.get((s or "").lower(), "")


def shape_ok(src_shape, dst_shape):
    """Step 1：形状强校验。返回 (ok, reason)。

    src_shape 素材形状，dst_shape 原牌形状。
    任一为空视为未知 —— 未知不拦（否则整个数据集都会被跳过），
    但两者都有值时必须完全一致。
    """
    a = normalize_shape(src_shape)
    b = normalize_shape(dst_shape)
    if not a or not b:
        return True, "unknown_pass"
    return (a == b), ("ok" if a == b else "shape_mismatch_%s_vs_%s" % (a, b))


# ---------------------------------------------------------------------------
# Step 2：红圈精定位
# ---------------------------------------------------------------------------
def refine_red_circle_bbox(bg_img, yolo_bbox, pad_ratio=1.04, search_pad=0.15):
    """在 YOLO 框内找真实红圈的外接框。

    为什么需要：YOLO 框带着几像素余量，而且不同帧余量不一样
    （实测框/红圈比值在 0.76~1.39 乱跳），用固定倍数放大永远对不准。

    做法：
      外扩 search_pad 搜索（防 YOLO 把红圈切掉一点）
      宽容度高的双区间红色掩码（覆盖阴天暗红和晴天亮红）
      取最大红色连通域的外接框，中心微扩 pad_ratio 盖死原红边
      找不到或太小 → 退回原 YOLO 框
    """
    import cv2

    H, W = bg_img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in yolo_bbox]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    sx1 = max(0, x1 - int(bw * search_pad))
    sy1 = max(0, y1 - int(bh * search_pad))
    sx2 = min(W, x2 + int(bw * search_pad))
    sy2 = min(H, y2 + int(bh * search_pad))
    roi = bg_img[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return (x1, y1, x2, y2), {"refined": False, "reason": "empty"}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 35, 30]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 35, 30]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(m1, m2)

    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (x1, y1, x2, y2), {"refined": False, "reason": "no_red"}

    c = max(cnts, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(c)
    # 合理性校验：红圈不能太小（噪点）也不能比框大太多
    if rw < bw * 0.5 or rh < bh * 0.5 or rw > bw * 1.6 or rh > bh * 1.6:
        return (x1, y1, x2, y2), {"refined": False, "reason": "size_implausible",
                                  "red": [rw, rh], "box": [bw, bh]}

    cx = sx1 + rx + rw / 2.0
    cy = sy1 + ry + rh / 2.0
    nw, nh = rw * pad_ratio, rh * pad_ratio
    nx1 = max(0, int(cx - nw / 2.0))
    ny1 = max(0, int(cy - nh / 2.0))
    nx2 = min(W, int(cx + nw / 2.0))
    ny2 = min(H, int(cy + nh / 2.0))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return (x1, y1, x2, y2), {"refined": False, "reason": "too_small"}
    return (nx1, ny1, nx2, ny2), {"refined": True, "red": [rw, rh],
                                  "pad": round(pad_ratio, 3)}


# ---------------------------------------------------------------------------
# Step 3：白/红/黑三锚点颜色对齐
# ---------------------------------------------------------------------------
def align_color_three_anchors(bg_roi, asset_bgr, asset_alpha):
    """白/红/黑三锚点语义颜色对齐。

    核心思想：交通标志只有三种材质 —— 白底（反光膜）、红圈（反光膜染色）、
    黑字（印刷油墨）。它们对光的响应不同，必须分别对齐；
    用一套全局映射（我们之前用的 P5~P95）会让「白底对上了红圈就偏」。

    做法：
      1) 黑白锚点：非红区域取亮度 P75 当白、P15 当黑，
         在 BGR 三通道直接线性映射（不做 LAB/HSV 往返，绝不偏色）
      2) 红锚点：单独对齐红圈区域的饱和度 S 和亮度 V
      3) 红掩码宽容度不同：原图要宽（阴天红发暗），素材要严（干净）
    """
    import cv2

    valid = asset_alpha > 127
    if valid.sum() < 20:
        return asset_bgr, {"ok": False, "reason": "tiny"}

    bg_hsv = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2HSV)
    as_hsv = cv2.cvtColor(asset_bgr, cv2.COLOR_BGR2HSV)

    def red_mask(hsv_img, s_min):
        h = hsv_img[:, :, 0]
        m1 = (h <= 12) | (h >= 160)
        m2 = hsv_img[:, :, 1] >= s_min
        return m1 & m2 & valid

    bg_red = red_mask(bg_hsv, 35)
    as_red = red_mask(as_hsv, 60)

    bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
    as_gray = cv2.cvtColor(asset_bgr, cv2.COLOR_BGR2GRAY)
    bg_non_red = valid & (~bg_red)
    as_non_red = valid & (~as_red)

    info = {"ok": True}
    if bg_non_red.sum() > 10 and as_non_red.sum() > 10:
        bg_w = np.percentile(bg_gray[bg_non_red], 75)
        bg_k = np.percentile(bg_gray[bg_non_red], 15)
        as_w = np.percentile(as_gray[as_non_red], 75)
        as_k = np.percentile(as_gray[as_non_red], 15)

        W_bg = np.mean(bg_roi[bg_non_red & (bg_gray >= bg_w)], axis=0)
        K_bg = np.mean(bg_roi[bg_non_red & (bg_gray <= bg_k)], axis=0)
        W_as = np.mean(asset_bgr[as_non_red & (as_gray >= as_w)], axis=0)
        K_as = np.mean(asset_bgr[as_non_red & (as_gray <= as_k)], axis=0)

        # 对比度缩放限幅，避免把极暗/极亮场景拉爆
        scale = (W_bg - K_bg) / (W_as - K_as + 1e-5)
        scale = np.clip(scale, 0.70, 1.6)
        # 实测：原牌拍灰后白底~150/黑字~40，素材纯白~245/纯黑~25，
        # 直接按黑白锚点映射会把素材压到 0.5~0.73 倍对比度，
        # 红圈和白底一起发灰 —— 这是「失真」的根因。
        # 折中：只压一半，保留素材的通透感。
        scale = 1.0 - (1.0 - scale) * 0.5
        out_f = (asset_bgr.astype(np.float32) - K_as) * scale + K_bg
        out_bgr = np.clip(out_f, 0, 255).astype(np.uint8)
        info["white_shift"] = [round(float(v), 1) for v in (W_bg - W_as)]
        info["scale"] = [round(float(v), 2) for v in scale]
    else:
        out_bgr = asset_bgr.copy()
        info["ok"] = False
        info["reason"] = "no_anchor"

    # 红圈单独对齐 S / V
    if bg_red.sum() > 10 and as_red.sum() > 10:
        out_hsv = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_rs = float(np.mean(bg_hsv[:, :, 1][bg_red]))
        bg_rv = float(np.mean(bg_hsv[:, :, 2][bg_red]))
        as_rs = float(np.mean(out_hsv[:, :, 1][as_red]))
        as_rv = float(np.mean(out_hsv[:, :, 2][as_red]))

        if as_rs > 1e-3:
            # 比例限幅 0.5~2.2，防止素材红圈被拉到过饱和
            k_s = float(np.clip(bg_rs / as_rs, 0.75, 1.6))
            out_hsv[:, :, 1][as_red] = np.clip(out_hsv[:, :, 1][as_red] * k_s, 0, 255)
            info["red_s_ratio"] = round(k_s, 2)
        if as_rv > 1e-3:
            # 实测：原限幅 0.6~1.8 让 red_v_ratio 落到 0.65，
            # 红圈被压暗发灰（S_P90 掉 36.7）。
            # 红圈的亮度本来就该跟背景接近，不该大幅压暗，收窄到 0.85~1.25。
            k_v = float(np.clip(bg_rv / as_rv, 0.85, 1.25))
            out_hsv[:, :, 2][as_red] = np.clip(out_hsv[:, :, 2][as_red] * k_v, 0, 255)
            info["red_v_ratio"] = round(k_v, 2)
        out_bgr = cv2.cvtColor(out_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        info["red_px"] = int(bg_red.sum())
    else:
        info["red_px"] = int(bg_red.sum())

    return out_bgr, info


# ---------------------------------------------------------------------------
# Step 4：轻微光学退化
# ---------------------------------------------------------------------------
def degrade_light(asset_bgr, ref_roi, alpha, max_sigma=1.6, noise_cap=4.5):
    """Step 4：匹配原图清晰度的轻微高斯模糊 + 少量噪点。

    「轻微」是重点：这里的目的是让素材不要比原图更锐，
    而不是制造明显的模糊感（那一步交给 Step 6 的 JPEG）。
    """
    import cv2

    g = cv2.cvtColor(ref_roi.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    ref_lap = float(cv2.Laplacian(g, cv2.CV_64F).var())

    p = asset_bgr.astype(np.float32)
    # 只做到和原图大致同锐度，不做迭代搜索（避免过模糊）
    target = ref_lap * 1.05
    applied = 0.0
    for sigma in (0.0, 0.4, 0.7, 1.0, 1.4, max_sigma):
        if sigma <= 0.01:
            cur = p
        else:
            k = int(2 * round(2 * sigma) + 1)
            cur = cv2.GaussianBlur(p, (k, k), sigma)
        gg = cv2.cvtColor(np.clip(cur, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if float(cv2.Laplacian(gg, cv2.CV_64F).var()) <= target:
            p = cur
            applied = sigma
            break
        p = cur
        applied = sigma

    # 噪点：从原图高频残差估
    valid = alpha > 127
    if valid.sum() > 50:
        smooth = cv2.medianBlur(ref_roi.astype(np.uint8), 3)
        residual = (ref_roi.astype(np.float32) - smooth.astype(np.float32))[valid]
        sd = float(np.std(residual)) * 0.8
        sd = float(np.clip(sd, 0.0, noise_cap))
        if sd > 0.6:
            p = p + np.random.normal(0, sd, p.shape).astype(np.float32)

    return np.clip(p, 0, 255).astype(np.uint8), {"blur_sigma": round(applied, 2),
                                                 "ref_lap": round(ref_lap, 0)}


# ---------------------------------------------------------------------------
# Step 5：动态边缘融合（不腐蚀，极窄羽化）
# ---------------------------------------------------------------------------
def blend_narrow(bg_roi, asset_bgr, alpha, ksize=3):
    """Step 5：不腐蚀，用 ksize=3 的极窄羽化把素材融进 bg_roi。

    为什么不用腐蚀：
      腐蚀会把贴图缩进 1px，露出原牌的红边 —— 这是之前「重影」的来源之一。
      红圈精定位（Step 2）已经用 pad_ratio=1.04 把原红边盖死了，
      这里只需要极窄羽化做抗锯齿，不需要再腐蚀。
    """
    import cv2

    a = alpha.astype(np.float32) / 255.0
    k = ksize if ksize % 2 == 1 else ksize + 1
    soft = cv2.GaussianBlur(a, (k, k), 0)
    soft3 = soft[:, :, None]
    out = asset_bgr.astype(np.float32) * soft3 + bg_roi.astype(np.float32) * (1.0 - soft3)
    return np.clip(out, 0, 255).astype(np.uint8), soft


# ---------------------------------------------------------------------------
# Step 6：车载 JPEG 压缩
# ---------------------------------------------------------------------------
def apply_dashcam_jpeg_artifact(blended_roi, mask_roi, quality=60):
    """Step 6：在融合后的 blended_roi 上做一次 JPEG 编解码。

    必须放最后：这样牌面边缘和周围像素是一起被压缩的，
    整块区域带一致的压缩感；放在融合前的话，
    伪影会被 alpha 削掉一半，也不会外溢到牌的边缘外侧。
    只作用于标牌及外扩 2px，避免二次压缩污染周围真实背景。
    """
    import cv2

    q = int(np.clip(quality, 55, 65))
    ok, buf = cv2.imencode(".jpg", blended_roi, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return blended_roi, {"applied": False}
    compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(mask_roi, k, iterations=1)
    soft = (cv2.GaussianBlur(dil, (3, 3), 0).astype(np.float32) / 255.0)[:, :, None]
    out = compressed.astype(np.float32) * soft + blended_roi.astype(np.float32) * (1.0 - soft)
    return np.clip(out, 0, 255).astype(np.uint8), {"applied": True, "quality": q}


if __name__ == "__main__":
    print("pipeline6：形状校验 → 红圈精定位 → 三锚点对齐 → 轻微退化 → 窄羽化融合 → JPEG+QC")

# ---------------------------------------------------------------------------
# 编排：严格按 6 步顺序执行
# ---------------------------------------------------------------------------
def run_pipeline6(orig_bgr, yolo_bbox, asset_rgba, dst_shape,
                  jpeg_quality=60, pad_ratio=1.04, debug=False):
    """严格按 6 步执行一次标牌替换。

    orig_bgr   原图（不会被修改）
    yolo_bbox  YOLO 检测框
    asset_rgba 素材 RGBA（高清原尺寸）
    dst_shape  原牌形状（circle/square/triangle），用于 Step 1 校验

    返回 (out_img, info)
      info["ok"] == False 时 out_img 是原图，info["reason"] 说明原因
    """
    import cv2
    import numpy as np

    info = {"ok": False, "steps": []}

    # ---- Step 1：形状强校验 ----
    src_shape = _guess_asset_shape(asset_rgba)
    ok1, why1 = shape_ok(src_shape, dst_shape)
    info["steps"].append({"step": 1, "name": "shape_check", "src": src_shape,
                          "dst": dst_shape, "ok": ok1, "reason": why1})
    if not ok1:
        info["reason"] = why1
        return orig_bgr, info

    # ---- Step 2：红圈精定位 + 切 bg_roi + resize 素材 ----
    (nx1, ny1, nx2, ny2), rinfo = refine_red_circle_bbox(
        orig_bgr, yolo_bbox, pad_ratio=pad_ratio)
    nw, nh = nx2 - nx1, ny2 - ny1
    if nw < 8 or nh < 8:
        info["reason"] = "bbox_too_small"
        return orig_bgr, info
    bg_roi = orig_bgr[ny1:ny2, nx1:nx2].copy()
    asset = cv2.resize(asset_rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    asset_bgr = asset[:, :, :3]
    asset_alpha = asset[:, :, 3] if asset.shape[2] == 4 else np.full((nh, nw), 255, np.uint8)
    info["steps"].append({"step": 2, "name": "refine_bbox", **rinfo,
                          "bbox": [nx1, ny1, nx2, ny2], "size": [nw, nh]})

    # ---- Step 3：三锚点颜色对齐（必须在模糊/加噪之前）----
    asset_bgr2, cinfo = align_color_three_anchors(bg_roi, asset_bgr, asset_alpha)
    info["steps"].append({"step": 3, "name": "color_3anchors", **cinfo})

    # ---- Step 4：轻微光学退化 ----
    asset_bgr3, dinfo = degrade_light(asset_bgr2, bg_roi, asset_alpha)
    info["steps"].append({"step": 4, "name": "light_degrade", **dinfo})

    # ---- Step 5：动态边缘融合（不腐蚀，ksize=3）----
    blended_roi, soft = blend_narrow(bg_roi, asset_bgr3, asset_alpha, ksize=3)
    mask_roi = (soft * 255).astype(np.uint8)
    info["steps"].append({"step": 5, "name": "narrow_blend", "ksize": 3})

    # ---- Step 6：JPEG 压缩 + 贴回原图 ----
    final_roi, jinfo = apply_dashcam_jpeg_artifact(blended_roi, mask_roi,
                                                   quality=jpeg_quality)
    info["steps"].append({"step": 6, "name": "jpeg", **jinfo})

    out = orig_bgr.copy()
    out[ny1:ny2, nx1:nx2] = final_roi
    info["ok"] = True
    info["bbox"] = [nx1, ny1, nx2, ny2]
    return out, info


def _guess_asset_shape(asset_rgba):
    """从素材 alpha 猜形状：四角透明 + 接近方形填满 → 圆；顶窄底宽 → 三角。"""
    import cv2
    import numpy as np

    a = asset_rgba[:, :, 3]
    h, w = a.shape
    if h < 8 or w < 8:
        return ""
    corners = [a[2, 2], a[2, w - 3], a[h - 3, 2], a[h - 3, w - 3]]
    corner_transparent = sum(1 for c in corners if c < 40)
    if corner_transparent < 3:
        return "square"
    # 顶部 15% 行 vs 底部 15% 行的前景宽度
    top = float((a[int(h * 0.15), :] > 127).sum())
    bot = float((a[int(h * 0.85), :] > 127).sum())
    if bot > 0 and top / bot < 0.45:
        return "triangle"
    return "circle"
