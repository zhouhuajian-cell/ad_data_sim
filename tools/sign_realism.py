#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_realism 兼容层 —— 原模块缺失，映射到 sign_realism2。

原文件在仓库整理时被误删（git 里也没有历史版本），
这里提供向后兼容的 paste_realistic / make_realistic，
内部走 sign_realism2 的 degrade_v2 + paste_v2。
"""
import cv2
import numpy as np

from sign_realism2 import degrade_v2, paste_v2, scene_stats


def make_realistic(sign_bgr, sign_alpha, ref_patch, strength=1.0):
    """兼容旧接口：把素材（含 alpha）劣化成匹配 ref_patch 的样子。

    返回 (bgr, alpha)。
    """
    try:
        if sign_bgr.ndim == 3 and sign_bgr.shape[2] == 4:
            bgr = sign_bgr[:, :, :3]
            a = sign_bgr[:, :, 3]
        else:
            bgr = sign_bgr
            a = sign_alpha
        if a.dtype != np.uint8:
            a = np.clip(a * 255 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
        if ref_patch is None or not hasattr(ref_patch, "shape"):
            return bgr.copy(), a.copy()
        scene = scene_stats(ref_patch, [0, 0, ref_patch.shape[1] - 1,
                                        ref_patch.shape[0] - 1])
        out = degrade_v2(bgr, a, ref_patch, scene, strength=strength)
        return out, a.copy()
    except Exception:
        if sign_bgr.ndim == 3 and sign_bgr.shape[2] == 4:
            return sign_bgr[:, :, :3].copy(), sign_bgr[:, :, 3].copy()
        return sign_bgr.copy(), sign_alpha.copy()


def paste_realistic(image, sign_bgr, sign_alpha, bbox, strength=1.0, **kw):
    """兼容旧接口：转发到 paste_v2。"""
    return paste_v2(image, sign_bgr, sign_alpha, bbox,
                    strength=strength, edge_glow=kw.get("edge_glow", True))
