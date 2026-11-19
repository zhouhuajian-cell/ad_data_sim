#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auto_tune.py — 逐帧自动寻优 grow / brightness，选 QC 最优参数

问题（qc_stats 实测）：
  通过率 71.8% 时，瓶颈是 Gate2（边缘 38.1%）和 Gate3（亮度 35.4%）。
  根因是这两个参数全局固定：
    grow 固定 1.12，但检测框/实际标牌的尺寸比在 0.76~1.39 之间浮动
    brightness_gain 固定 0.50，但各帧环境亮度差异很大

  固定参数永远只能在「某些帧偏大、某些帧偏小」之间折中。

做法：
  对每一帧，在小范围内试几组 (grow, brightness) 组合，
  每组跑一次合成 + QC，选最好的一组。
  评分规则：QC 通过优先，其次边缘跳变 + 亮度偏差最小。

  为控制耗时，二维网格只取 3×3 = 9 组，且只在第一次不通过时才继续试 ——
  第一组就过了就直接用，多数帧只跑一次。

用法：
  from auto_tune import best_params_for_frame
  grow, bright, qc_info = best_params_for_frame(im, sign, mask, bbox, shape)
"""
import numpy as np


# 候选值范围由实测确定：
#   400 个样本里 Gate3 失败帧的亮度偏差在 -102 ~ +51 之间，
#   brightness 0.42~0.60 只能覆盖很小一段，必须放宽到 0.35~1.00。
#   grow 同理，检测框/实际尺寸比在 0.76~1.39 浮动，候选要覆盖到 1.0~1.35。
GROW_CANDIDATES = (0.95, 1.08, 1.20, 1.32)
BRIGHT_CANDIDATES = (0.35, 0.50, 0.68, 0.85, 1.00, 1.15, 1.30)


def _score(info):
    """QC 结果打分，越小越好。"""
    if not info.get("passed"):
        # 不通过：按偏离量给一个较大的分数
        base = 1000.0
        base += abs(info.get("edge_jump", 0) - 38) * 2
        base += abs(info.get("bright_diff", 0)) * 2
        return base
    # 通过：跳变和亮度偏差越小越好
    return info.get("edge_jump", 0) * 1.0 + abs(info.get("bright_diff", 0)) * 1.2


def best_params_for_frame(im, sign_bgr, sign_mask, bbox0, shape="circle",
                          grow_candidates=GROW_CANDIDATES,
                          bright_candidates=BRIGHT_CANDIDATES,
                          qc_edge=38.0, qc_bright=38.0,
                          stop_on_first_pass=True):
    """在一帧上搜最优 (grow, brightness)。

    参数：
      im            原图
      sign_bgr      素材 BGR（原始尺寸 512×512 之类）
      sign_mask     素材 alpha
      bbox0         YOLO 检测框
      shape         原牌形状（circle/square/triangle）
      stop_on_first_pass  第一组通过就停（省时间）

    返回 (grow, brightness, info, composed)
      composed 是选定参数下合成好的图，可直接用，不用再算一遍
    """
    import cv2

    from auto_qc import qc_check
    from batch_replace import fit_bbox
    from sign_realism2 import PARAMS, paste_v2

    orig_bright = PARAMS["bright_gain"]
    best = None
    try:
        for g in grow_candidates:
            # 按 grow 算贴图区域
            cx = (bbox0[0] + bbox0[2]) * 0.5
            cy = (bbox0[1] + bbox0[3]) * 0.5
            hw = (bbox0[2] - bbox0[0]) * 0.5 * g
            hh = (bbox0[3] - bbox0[1]) * 0.5 * g
            grown = [cx - hw, cy - hh, cx + hw, cy + hh]
            bbox = fit_bbox(grown, sign_mask.shape)

            for bg in bright_candidates:
                PARAMS["bright_gain"] = bg
                composed, cinfo = paste_v2(im, sign_bgr, sign_mask, bbox, use_v3=True)
                if not cinfo.get("ok"):
                    continue
                ok, qinfo = qc_check(im, composed, bbox, shape,
                                     edge_jump_max=qc_edge,
                                     bright_diff_max=qc_bright)
                sc = _score(qinfo)
                if best is None or sc < best[0]:
                    best = (sc, g, bg, qinfo, composed, bbox)
                if stop_on_first_pass and ok:
                    # 第一组就通过，直接返回
                    PARAMS["bright_gain"] = orig_bright
                    return g, bg, qinfo, composed
    finally:
        PARAMS["bright_gain"] = orig_bright

    if best is None:
        return None, None, {"passed": False, "reason": "all_failed"}, im
    return best[1], best[2], best[3], best[4]


if __name__ == "__main__":
    print("auto_tune：逐帧搜索 grow / brightness，选 QC 最优参数")
    print("  GROW_CANDIDATES   =", GROW_CANDIDATES)
    print("  BRIGHT_CANDIDATES =", BRIGHT_CANDIDATES)
