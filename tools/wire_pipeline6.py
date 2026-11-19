#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wire_pipeline6.py — 把 run_pipeline6 挂进 batch_walk 的贴图环节"""
import io

p = "/opt/ue5_synth/tools/batch_walk.py"
s = io.open(p, encoding="utf-8").read()

if "run_pipeline6(out_im" in s:
    print("已接线")
    raise SystemExit(0)

old = '''                out_im, cinfo = paste_v2(out_im, s[:, :, :3], m, bbox,
                                         use_v3=a.realism == "v3")
                if not cinfo.get("ok"):
                    continue
                objects.append({
                    "class": cls_name,
                    "category": cls_name.split("_")[0],
                    "bbox": [round(float(v), 1) for v in bbox],
                    "score": round(float(d["conf"]), 3),
                    "material": os.path.basename(path),
                    "occluder_ratio": cinfo.get("occluder_ratio", 0),
                })'''

new = '''                # ===== 6 步严格流水线（默认）=====
                # 形状校验 → 红圈精定位 → 三锚点对齐 → 轻微退化
                # → 窄羽化融合 → JPEG 压缩 + 贴回
                if a.pipe6 and not a.legacy_pipe:
                    # Step 1 已经在上面用 std_shape 判过 dst_shape，
                    # 这里把素材包成 RGBA 交给流水线，由它再做一次形状强校验
                    asset_rgba = cv2.cvtColor(s[:, :, :3], cv2.COLOR_BGR2GRAY) if False else None
                    s_rgba = np.dstack([s[:, :, :3], m]) if s.shape[2] == 3 else s
                    out_im, p6 = run_pipeline6(
                        out_im, d["bbox"], s_rgba, sh0,
                        jpeg_quality=a.pipe6_jpeg, pad_ratio=a.pipe6_pad)
                    if not p6.get("ok"):
                        p6_fail += 1
                        p6_reasons[p6.get("reason", "unknown").split("_")[0]] = \\
                            p6_reasons.get(p6.get("reason", "unknown").split("_")[0], 0) + 1
                        continue
                    bbox_used = p6["bbox"]
                    objects.append({
                        "class": cls_name,
                        "category": cls_name.split("_")[0],
                        "bbox": [round(float(v), 1) for v in bbox_used],
                        "score": round(float(d["conf"]), 3),
                        "material": os.path.basename(path),
                        "pipeline": "v6",
                    })
                    continue

                out_im, cinfo = paste_v2(out_im, s[:, :, :3], m, bbox,
                                         use_v3=a.realism == "v3")
                if not cinfo.get("ok"):
                    continue
                objects.append({
                    "class": cls_name,
                    "category": cls_name.split("_")[0],
                    "bbox": [round(float(v), 1) for v in bbox],
                    "score": round(float(d["conf"]), 3),
                    "material": os.path.basename(path),
                    "occluder_ratio": cinfo.get("occluder_ratio", 0),
                })'''

if old not in s:
    print("ERROR 贴图锚点未找到")
    raise SystemExit(2)
s = s.replace(old, new, 1)

# 参数
if "--pipe6-jpeg" not in s:
    s = s.replace(
        '    ap.add_argument("--legacy-pipe", action="store_true",',
        '    ap.add_argument("--pipe6-jpeg", type=int, default=60,\n'
        '                    help="6 步流水线的 JPEG 质量（55~65，默认 60）")\n'
        '    ap.add_argument("--pipe6-pad", type=float, default=1.04,\n'
        '                    help="红圈精定位后的外扩系数（默认 1.04 盖死原红边）")\n'
        '    ap.add_argument("--legacy-pipe", action="store_true",',
    )

# 计数器
if "p6_fail = 0" not in s:
    s = s.replace("    rot_applied = 0\n", "    rot_applied = 0\n    p6_fail = 0\n    p6_reasons = {}\n")
    s = s.replace('        "rotate_applied": rot_applied,',
                  '        "rotate_applied": rot_applied,\n        "pipe6_fail": p6_fail,\n        "pipe6_fail_reasons": p6_reasons,')
    s = s.replace(
        '''    if not a.no_rotate:
        print("旋转对齐 %d 个标牌" % rot_applied)''',
        '''    if not a.no_rotate:
        print("旋转对齐 %d 个标牌" % rot_applied)
    if a.pipe6 and not a.legacy_pipe:
        print("6 步流水线失败 %d 个%s" % (p6_fail,
              ("（%s）" % ", ".join("%s:%d" % kv for kv in p6_reasons.items())) if p6_reasons else ""))''',
    )

io.open(p, "w", encoding="utf-8").write(s)
print("6 步流水线已接线")
