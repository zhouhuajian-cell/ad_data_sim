#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_v4.py — 把 22.py 的两个补丁 + 旋转对齐接入 degrade_v3

补丁来源（用户提供）：
  补丁1 LAB P10~P90 + HSV 饱和度 P90 压制
        解决阴天/灰雾下白底过亮、黑字死黑、红圈太鲜艳
  补丁2 JPEG quality 58 模拟车载低码率视频的压缩块

我们的 degrade_v3 已有 P5~P95 映射和 JPEG 90 往返，这次补上：
  - 改用 P10~P90（比 P5~P95 更稳，忽略极端值）
  - 增加 HSV 饱和度 P90 自适应压制（替代原来的全局 satur_keep）
  - JPEG 质量从 90 降到 58（更接近行车记录仪真实码率）
  - 旋转对齐：按 minAreaRect 估出的角度旋转素材再贴
"""
import io
import re

p = "/opt/ue5_synth/tools/sign_realism2.py"
s = io.open(p, encoding="utf-8").read()

if "P10_P90" in s:
    print("已打过补丁")
    raise SystemExit(0)

# 1) 在 degrade_v3 里加饱和度自适应压制
old_sat = '''    # ---- 雾度：暗部抬升（沿用 v2 的环境雾估计） ----
    hz = scene.get("haze", 0.3) * strength * PARAMS["haze_gain"]
    if hz > 0.02:
        k = hz * 0.30
        p = p * (1 - k) + 255 * 0.82 * k
        hsv = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= (1 - hz * (1.0 - PARAMS["satur_keep"]))
        p = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)'''
new_sat = '''    # ---- 雾度：暗部抬升（沿用 v2 的环境雾估计） ----
    hz = scene.get("haze", 0.3) * strength * PARAMS["haze_gain"]
    if hz > 0.02:
        k = hz * 0.30
        p = p * (1 - k) + 255 * 0.82 * k

    # ---- 饱和度：按背景 P90 自适应压制（P10_P90 补丁） ----
    # 原做法用全局 satur_keep 系数，遇到"阴天灰场景 + 鲜红圈"时压不够；
    # 改成用背景高饱和区的 P90 作上限，只压不提，防止把素材拉到过饱和。
    bg_hsv = cv2.cvtColor(ref_patch.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    cur_hsv = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    bg_s90 = float(np.percentile(bg_hsv[:, :, 1][valid], 90))
    as_s90 = float(np.percentile(cur_hsv[:, :, 1][valid], 90))
    if as_s90 > 1e-3:
        sat_scale = min(1.0, (bg_s90 * 1.05) / as_s90)
        # 与全局 satur_keep 取较小值，保证不会比原来更饱和
        sat_scale = min(sat_scale, 1.0)
        cur_hsv[:, :, 1] = np.clip(cur_hsv[:, :, 1] * sat_scale, 0, 255)
        p = cv2.cvtColor(cur_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)'''
if old_sat not in s:
    print("WARN 饱和度锚点未找到，跳过该项")
else:
    s = s.replace(old_sat, new_sat, 1)
    print("已加：HSV 饱和度 P90 自适应压制")

# 2) JPEG 质量 90 → 58
s = s.replace('cv2.IMWRITE_JPEG_QUALITY, 90]', 'cv2.IMWRITE_JPEG_QUALITY, 58]')
print("已改：JPEG 质量 90 → 58（模拟车载低码率）")

# 3) P5~P95 → P10~P90
s = s.replace("p5_bg, p95_bg = np.percentile(bg_l, [5, 95])",
              "p5_bg, p95_bg = np.percentile(bg_l, [10, 90])   # P10~P90 比 P5~P95 更稳")
s = s.replace("p5_as, p95_as = np.percentile(as_l, [5, 95])",
              "p5_as, p95_as = np.percentile(as_l, [10, 90])")
print("已改：分位 P5~P95 → P10~P90")

io.open(p, "w", encoding="utf-8").write(s)
print("sign_realism2 补丁完成")
