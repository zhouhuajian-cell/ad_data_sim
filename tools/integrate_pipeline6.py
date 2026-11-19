#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""integrate_pipeline6.py — 把 6 步流水线接入 batch_walk"""
import io

p = "/opt/ue5_synth/tools/pipeline6.py"
s = io.open(p, encoding="utf-8").read()

if "def run_pipeline6" in s:
    print("已存在")
    raise SystemExit(0)

runner = '''

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
'''

s = s.rstrip() + runner
io.open(p, "w", encoding="utf-8").write(s)
print("run_pipeline6 已加入 pipeline6.py")

# ---- 接入 batch_walk ----
p2 = "/opt/ue5_synth/tools/batch_walk.py"
s2 = io.open(p2, encoding="utf-8").read()
if "run_pipeline6" in s2:
    print("batch_walk 已接入")
    raise SystemExit(0)

s2 = s2.replace(
    "    from fix_grow import grow_for_frame",
    "    from fix_grow import grow_for_frame\n"
    "    from pipeline6 import run_pipeline6",
)
s2 = s2.replace(
    '    ap.add_argument("--no-rotate", action="store_true",',
    '    ap.add_argument("--pipe6", action="store_true", default=True,\n'
    '                    help="使用 6 步严格流水线（默认开）：形状校验→红圈精定位→'
    '三锚点对齐→轻微退化→窄羽化融合→JPEG+QC")\n'
    '    ap.add_argument("--legacy-pipe", action="store_true",\n'
    '                    help="回退到旧的 sign_realism2 流水线")\n'
    '    ap.add_argument("--no-rotate", action="store_true",',
)
io.open(p2, "w", encoding="utf-8").write(s2)
print("batch_walk 参数已加")
