#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接入 SD 精修到 batch_walk.py"""
import io

p = "/opt/ue5_synth/tools/batch_walk.py"
s = io.open(p, encoding="utf-8").read()

if "--sd" in s:
    print("已接入")
    raise SystemExit(0)

# 1) import
s = s.replace(
    "    from auto_qc import qc_check, gate4_model_check",
    "    from auto_qc import qc_check, gate4_model_check\n"
    "    from sd_refine import refine as sd_refine, make_sign_mask, strength_for_size as sd_strength",
)

# 2) 参数
s = s.replace(
    '    ap.add_argument("--gate4", action="store_true",',
    '    ap.add_argument("--sd", action="store_true",\n'
    '                    help="用 Stable Diffusion 局部重绘精修（光影融合），需要 /data/models/sd15-inpaint")\n'
    '    ap.add_argument("--sd-strength", type=float, default=None,\n'
    '                    help="SD 重绘幅度；不给就按标牌尺寸自动选 0.20~0.28")\n'
    '    ap.add_argument("--sd-model", default="/data/models/sd15-inpaint")\n'
    '    ap.add_argument("--gate4", action="store_true",',
)

# 3) 在 Gate4 之前做 SD 精修
old = '''            # Gate4：模型自检 —— 合成图喂回 YOLO，认不出来就说明这帧对训练没价值
            if a.gate4:'''
new = '''            # SD 局部重绘：让模型按周围环境重新打光，去掉"贴纸感"
            # 放在 Gate4 之前，因为 Gate4 要检验的正是精修后的成品
            if a.sd:
                sd_applied = 0
                for ob in objects:
                    try:
                        bb = ob["bbox"]
                        sz = min(bb[2] - bb[0], bb[3] - bb[1])
                        mk = make_sign_mask(out_im.shape, bb)
                        st = a.sd_strength if a.sd_strength is not None else sd_strength(sz)
                        out_im, sinfo = sd_refine(out_im, mk, strength=st,
                                                  sign_px=sz, model_dir=a.sd_model,
                                                  seed=int(sz))
                        if sinfo.get("ok"):
                            sd_applied += 1
                            ob["sd_strength"] = sinfo["strength"]
                    except Exception as e:
                        if sd_fail < 3:
                            print("    [SD-ERR] %s" % str(e)[:80], flush=True)
                        sd_fail += 1
                if sd_applied:
                    sd_count += 1

            # Gate4：模型自检 —— 合成图喂回 YOLO，认不出来就说明这帧对训练没价值
            if a.gate4:'''
if old not in s:
    print("ERROR Gate4 锚点")
    raise SystemExit(2)
s = s.replace(old, new, 1)

# 4) 计数器
s = s.replace(
    "    qc_reject = 0\n    tune_saved = 0\n    g4_reject = 0\n",
    "    qc_reject = 0\n    tune_saved = 0\n    g4_reject = 0\n    sd_count = 0\n    sd_fail = 0\n",
)

# 5) summary
s = s.replace(
    '        "gate4_reject": g4_reject,',
    '        "gate4_reject": g4_reject,\n        "sd_enabled": a.sd,\n        "sd_refined": sd_count,\n        "sd_fail": sd_fail,',
)

# 6) 输出
s = s.replace(
    '''    if a.gate4:
        print("Gate4 模型自检剔除 %d 张" % g4_reject)''',
    '''    if a.gate4:
        print("Gate4 模型自检剔除 %d 张" % g4_reject)
    if a.sd:
        print("SD 精修 %d 张（失败 %d）" % (sd_count, sd_fail))''',
)

io.open(p, "w", encoding="utf-8").write(s)
print("SD 精修已接入 batch_walk")
