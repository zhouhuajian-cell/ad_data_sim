#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_rotate.py — 把旋转对齐接入 batch_walk"""
import io

p = "/opt/ue5_synth/tools/batch_walk.py"
s = io.open(p, encoding="utf-8").read()

if "rotate_asset" in s:
    print("已接入")
    raise SystemExit(0)

# import
s = s.replace(
    "    from fix_grow import grow_for_frame",
    "    from fix_grow import grow_for_frame\n"
    "    from sign_rotate import estimate_angle, rotate_asset",
)

# 参数
s = s.replace(
    '    ap.add_argument("--auto-grow", action="store_true", default=True,',
    '    ap.add_argument("--no-rotate", action="store_true",\n'
    '                    help="禁用标牌旋转对齐（默认开：按 minAreaRect 估角度旋转素材）")\n'
    '    ap.add_argument("--auto-grow", action="store_true", default=True,',
)

# 在贴图前旋转素材
old = '''                # 实测结论（14 个样本，用红色外圈做真值）：'''
new = '''                # 旋转对齐：标牌在画面里有旋转角时，素材也要转同样角度。
                # 实测 200 帧里 81.5% 是圆/方牌（旋转对称，角度无意义），
                # 16% 能估出角度、其中 |角度| 90 分位 33 度 —— 这部分贴正的会明显假。
                if not a.no_rotate:
                    try:
                        ang, ainfo = estimate_angle(out_im, d["bbox"])
                        if abs(ang) >= 1.0:
                            s_rot = rotate_asset(s, ang)
                            m_rot = sign_mask(s_rot)
                            s, m = s_rot, m_rot
                            rot_applied += 1
                    except Exception:
                        pass

                # 实测结论（14 个样本，用红色外圈做真值）：'''
if old not in s:
    print("ERROR 贴图锚点")
    raise SystemExit(2)
s = s.replace(old, new, 1)

# 计数器
s = s.replace("    sd_count = 0\n", "    sd_count = 0\n    rot_applied = 0\n")
s = s.replace('        "sd_enabled": a.sd,', '        "rotate_applied": rot_applied,\n        "sd_enabled": a.sd,')
s = s.replace(
    '''    if a.sd:
        print("SD 精修 %d 张（失败 %d）" % (sd_count, sd_fail))''',
    '''    if not a.no_rotate:
        print("旋转对齐 %d 个标牌" % rot_applied)
    if a.sd:
        print("SD 精修 %d 张（失败 %d）" % (sd_count, sd_fail))''',
)

io.open(p, "w", encoding="utf-8").write(s)
print("旋转对齐已接入 batch_walk")
