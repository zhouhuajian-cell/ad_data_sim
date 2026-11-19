#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_rotate.py — 估标牌旋转角度并做仿射对齐，解决「贴斜」

方案 C：用 cv2.minAreaRect 拟合标牌本体的最小外接旋转矩形，
拿到角度后把素材旋转同样角度再贴 —— 标牌倾斜时也能贴正。

为什么要做：
  现在贴图永远是正的（轴对齐）。实拍里标牌常常因为：
    立柱倾斜、相机俯仰、路面坡度
  在画面里是有旋转角的。贴正的牌放在斜的牌位置，一眼就假。

怎么估角度：
  1) 框内 Otsu 分割出标牌本体
  2) cv2.minAreaRect 拿最小外接旋转矩形 (center, (w,h), angle)
  3) 用「宽/高比」判断哪个边是主轴：
       圆形牌 → w≈h，角度不可靠，直接返回 0
       方形牌 → 同理
       非正方（椭圆、竖长牌） → 角度有意义
  4) 角度做安全限幅：只接受 -35°~35°，超出视为误估

局限：
  圆牌和方牌本身就是旋转对称的，估不出角度 —— 但这两类也不需要，
  因为正着贴和斜着贴在视觉上差别极小。真正需要的是椭圆/长条牌。
"""
import sys

import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")


def estimate_angle(image, bbox, max_angle=35.0, min_aspect_dev=0.12):
    """估标牌旋转角（度）。返回 (angle, info)。

    angle 正值表示逆时针（OpenCV 约定）。
    估不出来返回 0。
    """
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = image[y1:y2, x1:x2]
    if roi.shape[0] < 24 or roi.shape[1] < 24:
        return 0.0, {"reason": "too_small"}

    h, w = roi.shape[:2]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 10) | 1,
                                                      max(3, h // 10) | 1))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, {"reason": "no_contour"}
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 120:
        return 0.0, {"reason": "tiny"}

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
    if rw < 6 or rh < 6:
        return 0.0, {"reason": "degenerate"}

    # minAreaRect 的角度约定：返回 0~90 或 -90~0，需要归一化到 -45~45
    a = float(angle)
    if a > 45:
        a -= 90
    elif a < -45:
        a += 90
    # 宽高比接近 1 的形状（圆/方）旋转对称，角度无意义
    asp = max(rw, rh) / float(max(1e-3, min(rw, rh)))
    if asp < (1.0 + min_aspect_dev):
        return 0.0, {"reason": "symmetric", "aspect": round(asp, 3)}
    # 限幅
    if abs(a) > max_angle:
        return 0.0, {"reason": "angle_too_big", "raw": round(a, 1)}
    return a, {"reason": "ok", "aspect": round(asp, 3), "raw": round(a, 1)}


def rotate_asset(asset_rgba, angle):
    """把素材旋转 -angle 度（抵消标牌的旋转）。返回旋转后的 RGBA。"""
    import cv2

    if abs(angle) < 0.8:
        return asset_rgba
    h, w = asset_rgba.shape[:2]
    # 扩画布防止旋出边界
    diag = int(np.ceil(np.hypot(h, w))) + 4
    canvas = np.zeros((diag, diag, 4), np.uint8)
    oy = (diag - h) // 2
    ox = (diag - w) // 2
    canvas[oy:oy + h, ox:ox + w] = asset_rgba

    M = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (diag, diag),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    # 裁掉透明边，回到紧凑画布
    a = rotated[:, :, 3]
    ys, xs = np.where(a > 8)
    if len(ys) == 0:
        return asset_rgba
    return rotated[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


if __name__ == "__main__":
    print("estimate_angle：用 minAreaRect 估标牌旋转角")
    print("rotate_asset：按角度旋转素材（含扩画布防裁剪）")
