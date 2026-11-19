#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接入动态 grow 到 batch_walk"""
import io

p = "/opt/ue5_synth/tools/batch_walk.py"
s = io.open(p, encoding="utf-8").read()

if "grow_for_frame" in s:
    print("已接入")
    raise SystemExit(0)

# import
s = s.replace(
    "    from auto_qc import qc_check, gate4_model_check",
    "    from auto_qc import qc_check, gate4_model_check\n"
    "    from fix_grow import grow_for_frame",
)

# 新参数
s = s.replace(
    '    ap.add_argument("--grow", type=float, default=1.12,',
    '    ap.add_argument("--auto-grow", action="store_true", default=True,\n'
    '                    help="按框内红圈占比动态决定 grow（默认开）：'
    '框刚好圈住牌时不放大，避免压到相邻标牌")\n'
    '    ap.add_argument("--fixed-grow", action="store_true",\n'
    '                    help="禁用动态 grow，一律用 --grow 的值")\n'
    '    ap.add_argument("--grow", type=float, default=1.12,',
)

# 使用动态 grow
old = '''                cx = (d["bbox"][0] + d["bbox"][2]) * 0.5
                cy = (d["bbox"][1] + d["bbox"][3]) * 0.5
                hw = (d["bbox"][2] - d["bbox"][0]) * 0.5 * a.grow
                hh = (d["bbox"][3] - d["bbox"][1]) * 0.5 * a.grow'''
new = '''                # 动态 grow：框刚好圈住红圈时（占比 >= 0.97）不放大，
                # 否则会越过框边界压到方底板上相邻的标牌。
                g_use = a.grow
                if not a.fixed_grow:
                    try:
                        g_use = grow_for_frame(out_im, d["bbox"], fallback=a.grow)
                    except Exception:
                        g_use = a.grow
                cx = (d["bbox"][0] + d["bbox"][2]) * 0.5
                cy = (d["bbox"][1] + d["bbox"][3]) * 0.5
                hw = (d["bbox"][2] - d["bbox"][0]) * 0.5 * g_use
                hh = (d["bbox"][3] - d["bbox"][1]) * 0.5 * g_use'''
if old not in s:
    print("ERROR grow 锚点")
    raise SystemExit(2)
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("动态 grow 已接入")
