#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_order.py — 按用户给的顺序调整降质流水线

用户指出的正确顺序：
  ① align_photometry_v3_patch(bg_roi, asset_roi, mask_roi)
     亮度 + 红圈饱和度压进原图环境
  ② 模糊、加噪、边缘 Alpha 融合 → blended_roi
  ③ apply_dashcam_jpeg_artifact(blended_roi, mask_roi, quality)
     压缩伪影放在融合之后，这样伪影才能延伸到牌的边缘与周围背景

当前实现的问题：
  JPEG 压缩被放在了「计算 alpha 之前」，
  等于先压缩图案再和背景混合 —— 压缩伪影会被 alpha 削弱一半，
  而且不会外溢到牌的边缘之外，缺少车载视频那种「整块区域发糊」的观感。

另外按用户建议：JPEG 质量要跟着原图画质走
  原图很糊 → quality 45~50
  原图偏高清 → quality 65~75
  用原图局部（ref_patch）的拉普拉斯方差判断。
"""
import io
import re

p = "/opt/ue5_synth/tools/sign_realism2.py"
s = io.open(p, encoding="utf-8").read()

if "def _jpeg_quality_for" in s:
    print("已打过顺序补丁")
    raise SystemExit(0)

# 1) 从 degrade_v3 里移除 JPEG 步骤
old_jpeg = '''    ok, buf = cv2.imencode(".jpg", np.clip(p, 0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        p = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)

'''
if old_jpeg not in s:
    print("WARN 未找到 JPEG 步骤")
else:
    s = s.replace(old_jpeg, "", 1)
    print("已从 degrade_v3 移除 JPEG（改到融合之后）")

# 2) 新增 JPEG 质量自适应函数 + 融合后压缩函数
anchor = "def degrade_v3(sign_bgr, mask, ref_patch, scene, strength=1.0):"
idx = s.find(anchor)
helpers = '''def _jpeg_quality_for(ref_patch):
    """按原图局部清晰度决定 JPEG 质量。

    用户建议：原图很糊 → 45~50；偏高清 → 65~75。
    用拉普拉斯方差衡量，经验映射：
      < 120   很糊   → 46
      120~400 一般   → 56
      400~900 较清晰 → 66
      > 900   高清   → 74
    """
    import cv2
    if ref_patch is None or ref_patch.size == 0:
        return 56
    g = cv2.cvtColor(ref_patch.astype("uint8"), cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    if lap < 120:
        return 46
    if lap < 400:
        return 56
    if lap < 900:
        return 66
    return 74


def apply_jpeg_artifact(blended, mask, quality):
    """在融合之后的成品上打 JPEG 压缩伪影（只作用于标牌及外扩 2px）。

    为什么放在融合之后：
      放在前面的话，压缩块会被边缘 alpha 再削一遍，
      而且伪影不会外溢到牌的边缘外侧 —— 缺少车载视频那种
      「标牌周围一圈也糊」的观感。放在后面才对得上真实编码顺序。
    """
    import cv2
    import numpy as np

    if blended is None or blended.size == 0:
        return blended
    ok, buf = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return blended
    compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # 只在标牌及外扩 2px 范围内应用，避免污染周围真实背景
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(mask, k, iterations=1)
    soft = (cv2.GaussianBlur(dil, (3, 3), 0).astype(np.float32) / 255.0)[:, :, None]
    out = compressed.astype(np.float32) * soft + blended.astype(np.float32) * (1.0 - soft)
    return np.clip(out, 0, 255).astype(np.uint8)


'''
s = s[:idx] + helpers + s[idx:]

# 3) paste_v2 里在 blend 之后调 JPEG
old_blend = '''    out[y1:y2, x1:x2] = np.clip(roi, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8), {
        "ok": True, "scene": {k: round(v, 1) for k, v in scene.items()},
        "target_brightness": round(scene["bright"] * (0.95 - 0.15 * scene["haze"]), 1),
    }'''
new_blend = '''    out[y1:y2, x1:x2] = np.clip(roi, 0, 255)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # ③ 融合之后再打 JPEG 压缩伪影（用户指定的顺序）
    jq = _jpeg_quality_for(ref)
    full_mask = np.zeros(out.shape[:2], np.uint8)
    full_mask[y1:y2, x1:x2] = soft.astype(np.uint8)
    out = apply_jpeg_artifact(out, full_mask, jq)

    return out, {
        "ok": True, "scene": {k: round(v, 1) for k, v in scene.items()},
        "jpeg_quality": jq,
        "target_brightness": round(scene["bright"] * (0.95 - 0.15 * scene["haze"]), 1),
    }'''
if old_blend not in s:
    print("WARN paste_v2 返回点未找到")
else:
    s = s.replace(old_blend, new_blend, 1)
    print("paste_v2 已在融合后应用 JPEG")

io.open(p, "w", encoding="utf-8").write(s)
print("顺序补丁完成")
