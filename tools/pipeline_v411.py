#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipeline_v411.py — V4.6 + 双牌门禁（最小改动，用户方案里唯一有效的部分）

只做一件事：贴图前判断检测框上下邻域是否还有别的牌。
  有 → 抛 DoubleSignSkip，调用方跳过该样本
  无 → 完全走 V4.6 原有流程（圆环模板匹配 + 三锚点对齐 + 窄羽化 + 局部JPEG）

为什么只保留这一条：
  用户方案的另外两个动作在真实数据上验证无效：
  · 椭圆透视拟合 → 圆牌 fitEllipse 会得到随机 angle，导致素材被拉歪
  · 白底板擦除   → 擦掉原红圈后留下白色圆盘，比不擦更违和
  实测 8 张对比（ellipse_cmp.jpg）两个动作都让结果更差。
  只有「双牌门禁」是真需求：方底板上两块牌只换一块，必然"一真一假"。
"""
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")

from pipeline_v46 import SignRealismV46Renderer
from sign_gate import refine_circle_and_check_double


class DoubleSignSkip(Exception):
    """双牌场景：调用方捕获后跳过该样本。"""
    pass


class SignRealismV411Renderer(SignRealismV46Renderer):
    """V4.6 + 双牌门禁。"""

    def __init__(self, jpeg_quality=82, skip_double=True,
                 double_abs_thresh=300, double_rel_thresh=0.25):
        super().__init__((jpeg_quality, jpeg_quality))
        self.skip_double = bool(skip_double)
        self.double_abs_thresh = float(double_abs_thresh)
        self.double_rel_thresh = float(double_rel_thresh)
        self.stats = {"ok": 0, "double": 0}

    def render(self, bg_img, yolo_bbox, asset_rgba):
        # ---- 门禁：用 sign_gate 的双牌判据（只取 double 判定）----
        if self.skip_double:
            status, _, _, _ = refine_circle_and_check_double(
                bg_img, yolo_bbox,
                double_abs_thresh=self.double_abs_thresh,
                double_rel_thresh=self.double_rel_thresh)
            if status == "double":
                self.stats["double"] += 1
                raise DoubleSignSkip("双牌/组合牌，跳过")
        self.stats["ok"] += 1

        # ---- 其余完全走 V4.6 ----
        return super().render(bg_img, yolo_bbox, asset_rgba)


if __name__ == "__main__":
    print("V4.11 = V4.6 + 双牌门禁（唯一保留改动）")
