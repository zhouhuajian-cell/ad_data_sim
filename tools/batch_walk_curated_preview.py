#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_walk.py — 遍历式批量替换：递归扫描目录树，自动挑图跑批量

背景：
  真实数据是两层结构 images/<序列名>/<图片>.jpg，
  而且有大量空目录（151276 个序列里大部分没图）。
  原来的 batch_replace.py 只处理单层目录，需要先手写文件清单。

本脚本做的事：
  1) 递归遍历目录树，收集所有 jpg/png
  2) 可按「最少检出数」「最小标牌尺寸」过滤（只跑有标牌、够大的帧）
  3) 输出到扁平目录：<序列名>__<帧名>，避免不同序列同名覆盖
  4) 产出统一标注 + summary.json

用法：
  python3 batch_walk.py --images <根目录> --country BR --out <输出目录> \
      --limit 1000 --min-sign 40
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 颜色一致性校验：防止 YOLO 类别误判导致贴错素材
# ============================================================
def _red_ratio(img):
    """红色占比（禁止/限速牌的红圈）"""
    import cv2
    if img is None or img.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 70, 60), (180, 255, 255))
    n = cv2.countNonZero(m1) + cv2.countNonZero(m2)
    return n / float(img.shape[0] * img.shape[1])


def _blue_ratio(img):
    """蓝色占比（指示/指路牌）"""
    import cv2
    if img is None or img.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (95, 60, 50), (130, 255, 255))
    return cv2.countNonZero(m) / float(img.shape[0] * img.shape[1])


def _sign_family(cls):
    """类别归族：决定该用红牌还是蓝牌校验"""
    c = (cls or "").lower()
    if c.startswith("speed_limit") or c.startswith("prohibition"):
        return "red"
    if c.startswith("mandatory") or c.startswith("guide") or c.startswith("info"):
        return "blue"
    if c.startswith("warning"):
        return "yellow"
    return "any"


def color_consistent(src_crop, asset_img, det_cls, min_red=0.06, min_blue=0.10):
    """检测框里的主色要和素材族匹配，否则判为误检。

    实测：YOLO 在 40~70px 的小标牌上分类不可靠，
    会把「禁止超车」认成 speed_limit。用颜色兜底能挡掉大部分误贴。
    """
    fam = _sign_family(det_cls)
    if fam == "red":
        r_src = _red_ratio(src_crop)
        r_ast = _red_ratio(asset_img)
        # 原框几乎无红色 -> 不可能是红圈牌
        if r_src < min_red:
            return False, "src_no_red(%.3f)" % r_src
        # 素材也是红圈才贴
        if r_ast < min_red:
            return False, "asset_no_red(%.3f)" % r_ast
    elif fam == "blue":
        b_src = _blue_ratio(src_crop)
        if b_src < min_blue:
            return False, "src_no_blue(%.3f)" % b_src
    return True, "ok"


def walk_images(root, max_scan=2000000, on_progress=None, every=100):
    """递归收集图片路径。

    on_progress(scanned_files, scanned_dirs, cur_dir) 每扫 every 张回调一次，
    用于前端显示「扫描阶段」的进度（14980 个子目录扫一遍要几十秒）。
    """
    exts = (".jpg", ".jpeg", ".png")
    out = []
    n = 0
    ndir = 0
    # 立即回调一次，前端立刻从 "unknown" 切到 "扫描中"
    if on_progress:
        try:
            on_progress(0, 0, root)
        except Exception:
            pass
    for dirpath, dirnames, filenames in os.walk(root):
        ndir += 1
        for fn in filenames:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
                n += 1
                if n % every == 0 and on_progress:
                    try:
                        on_progress(n, ndir, dirpath)
                    except Exception:
                        pass
                if n >= max_scan:
                    return out
        # 目录数变化也回调一次（某些目录没有图）
        if ndir % 100 == 0 and on_progress:
            try:
                on_progress(n, ndir, dirpath)
            except Exception:
                pass
    if on_progress:
        try:
            on_progress(n, ndir, root)
        except Exception:
            pass
    return out


def main():
    import cv2
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="根目录（递归）")
    ap.add_argument("--country", default="BR")
    ap.add_argument("--pin-class", default="",
                    help="指定标牌类别：只仿这一类（如 speed_limit_60）")
    ap.add_argument("--pin-material", default="",
                    help="指定素材文件名（如 speed_limit_60.png）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1000, help="成功生成的目标张数")
    ap.add_argument("--shuffle", action="store_true",
                    help="随机打乱文件顺序（避免只扫到某个标牌密度低的区域）")
    ap.add_argument("--max-scan", type=int, default=200000,
                    help="最多扫描多少个文件就停（防止扫全盘太慢）")
    ap.add_argument("--min-sign", type=int, default=40,
                    help="标牌短边最小像素（太小看不出效果，默认 40）")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--signs-dir", default="/data/tsr_library_v2")
    ap.add_argument("--vary-class", action="store_true")
    for name in ("bright-gain", "haze-gain", "contrast-gain", "satur-keep", "feather-ratio"):
        ap.add_argument("--" + name, type=float, default=None)
    # 实测检测框/原牌红圈的比值在 0.76~1.39 之间乱跳，任何固定倍数都会在
    # 某些帧上偏大、另一些帧上偏小。1.12 在多数帧上能盖住原牌边缘
    # （偏小会让原牌描边露出来，比偏大更明显），所以默认用 1.12。
    ap.add_argument("--v46", action="store_true", default=True,
                    help="V4.6 封板渲染器（默认开）：圆环模板精定位 + V4.4亮度参数 + 条件化白板软钳位")
    ap.add_argument("--legacy-render", action="store_true",
                    help="回退到旧渲染器（sign_realism2 / pipeline6）")
    ap.add_argument("--pipe6", action="store_true", default=True,
                    help="使用 6 步严格流水线（默认开）：形状校验→红圈精定位→三锚点对齐→轻微退化→窄羽化融合→JPEG+QC")
    ap.add_argument("--pipe6-jpeg", type=int, default=60,
                    help="6 步流水线的 JPEG 质量（55~65，默认 60）")
    ap.add_argument("--pipe6-pad", type=float, default=1.04,
                    help="红圈精定位后的外扩系数（默认 1.04 盖死原红边）")
    ap.add_argument("--legacy-pipe", action="store_true",
                    help="回退到旧的 sign_realism2 流水线")
    ap.add_argument("--no-rotate", action="store_true",
                    help="禁用标牌旋转对齐（默认开：按 minAreaRect 估角度旋转素材）")
    ap.add_argument("--auto-grow", action="store_true", default=True,
                    help="按框内红圈占比动态决定 grow（默认开）：框刚好圈住牌时不放大，避免压到相邻标牌")
    ap.add_argument("--fixed-grow", action="store_true",
                    help="禁用动态 grow，一律用 --grow 的值")
    ap.add_argument("--grow", type=float, default=1.12,
                    help="粘贴区域放大倍数（默认 1.12，盖住原牌描边；"
                         "1.0 按检测框原尺寸，实测会露出原牌边缘）")
    ap.add_argument("--tune", action="store_true",
                    help="逐帧自动寻优 grow/brightness（QC 不通过时尝试其他参数组合）")
    # SD 精修实测结论（三轮参数搜索）：
    #   SDXL + ControlNet（strength 0.28~0.58、canny 0.75~1.0）会把红圈重绘成
    #   粗黑边卡通样式、字体改成 AI 风格，连中性 prompt 也救不回来。
    #   SD 1.5 低强度（0.20~0.24）不篡改字形，但融合幅度极小。
    # 数据生产要的是「结构 100% 准确 + 画质自然退化」，
    # 引入不可控生成模型的风险大于收益，所以默认关闭，只留作实验开关。
    ap.add_argument("--sd", action="store_true",
                    help="[实验性，不推荐] SD 局部重绘精修；实测会篡改标牌设计风格")
    ap.add_argument("--sd-strength", type=float, default=None,
                    help="SD 重绘幅度；不给就按标牌尺寸自动选 0.20~0.28")
    ap.add_argument("--sd-model", default="/data/models/sd15-inpaint")
    ap.add_argument("--gate4", action="store_true",
                    help="启用 Gate4 模型自检：合成图喂回 YOLO，认不出来就丢弃")
    ap.add_argument("--gate4-conf", type=float, default=0.35)
    ap.add_argument("--qc", action="store_true",
                    help="开启自动质检：写盘前跑三道门控，不合格的样本直接丢弃")
    ap.add_argument("--qc-edge", type=float, default=38.0,
                    help="边缘跳变阈值（默认 38）")
    ap.add_argument("--qc-bright", type=float, default=38.0,
                    help="亮度偏差阈值（默认 38）")
    ap.add_argument("--realism", default="v3", choices=["v2", "v3"],
                    help="劣化算法版本：v3（P5-P95黑白点映射+迭代模糊匹配）或 v2（旧版）")
    ap.add_argument("--jobs-progress", default="",
                    help="进度写到这个文件，面板可轮询")
    a = ap.parse_args()

    # P1-FIX: 旧寻优分支更换图片和素材后未同步重建标注，
    # 会产出图片与标签错位的数据，先明确拒绝。
    if a.tune:
        ap.error(
            "--tune 暂不可用：旧寻优分支未同步更新类别、素材和标注框。"
            "请移除 --tune，质量不合格样本直接跳过。"
        )

    os.makedirs(a.out, exist_ok=True)
    ann_dir = os.path.join(a.out, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    from replace_sign import detect_signs, load_sign, sign_mask
    try:
        import walk_cache
    except Exception:
        walk_cache = None
    try:
        import walk_index
    except Exception:
        walk_index = None
    from sign_realism2 import paste_v2, set_params
    effective_params = set_params(bright_gain=a.bright_gain, haze_gain=a.haze_gain,
                                  contrast_gain=a.contrast_gain, satur_keep=a.satur_keep,
                                  feather_ratio=a.feather_ratio)
    from sign_occlusion import detect_occluder, add_backing_plate
    from auto_qc import qc_check, gate4_model_check
    from quality_contract import check_frame
    from fix_grow import grow_for_frame
    from pipeline6 import run_pipeline6
    from pipeline_v411 import SignRealismV411Renderer, DoubleSignSkip
    from sign_rotate import estimate_angle, rotate_asset
    from sd_refine import refine as sd_refine, make_sign_mask, strength_for_size as sd_strength
    from auto_tune import best_params_for_frame
    from batch_replace import (fit_bbox, pick_varied_asset, list_signs,
                               fit_to_sign_contour, std_shape, asset_shape,
                               fit_to_inner, measure_face)

    t0 = time.time()
    print("[1/3] 扫描目录…", flush=True)
    def write_progress_scan(nfile, ndir, cur_dir):
        """扫描阶段进度：前端据此显示「已扫 N 个目录 / M 张图」。"""
        if not a.jobs_progress:
            return
        d = {"phase": "scan", "done": nfile, "total": nfile,
             "scanned_files": nfile, "scanned_dirs": ndir,
             "cur_dir": cur_dir[-60:] if cur_dir else "",
             "ok": 0, "elapsed": round(time.time() - t0, 1)}
        try:
            with open(a.jobs_progress, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
        except Exception:
            pass

    # 扫描阶段也要写进度，避免前端一直显示 "unknown"
    def _scan_progress(nfile, ndir, cur):
        try:
            write_progress_scan(nfile, ndir, cur)
        except Exception:
            pass

    files = walk_images(a.images, a.max_scan, on_progress=_scan_progress)
    if a.shuffle:
        import random as _rnd
        _rnd.seed(42)          # 固定种子，结果可复现
        _rnd.shuffle(files)
        print("  已随机打乱顺序（种子 42）")
    print("  找到 %d 个图片文件（%.1fs）" % (len(files), time.time() - t0), flush=True)

    table = list_signs(a.signs_dir, a.country)
    if not table:
        print("ERROR: 国家 %s 在素材库里没有资产" % a.country)
        return 2
    print("  素材库 %s: %d 个类别" % (a.country, len(table)), flush=True)

    ok = skip_no_sign = skip_small = skip_color = fail = 0
    qc_reject = 0
    qc_unchecked = 0
    g4_soft_low = 0
    tune_saved = 0
    g4_reject = 0
    sd_count = 0
    rot_applied = 0
    p6_fail = 0
    v46_count = 0
    v46_fail = 0
    v46_double_skip = 0
    cache_hit = 0
    _v46 = SignRealismV411Renderer()
    p6_reasons = {}
    sd_fail = 0
    qc_reasons = {}
    summary = []
    total = len(files)

    def write_progress(phase, cur, tot, extra=None):
        if not a.jobs_progress:
            return
        d = {"phase": phase, "done": cur, "total": tot,
             "ok": ok, "skip_no_sign": skip_no_sign,
             "skip_small": skip_small, "skip_color": skip_color, "fail": fail,
             "elapsed": round(time.time() - t0, 1)}
        if extra:
            d.update(extra)
        try:
            with open(a.jobs_progress, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
        except Exception:
            pass

    print("[2/3] 逐帧检测与合成…", flush=True)
    skip_done = 0
    for i, p in enumerate(files):
        if ok >= a.limit:
            break
        # 增量：以「国家 + 图片名」为键，已经产出且文件还在就跳过
        img_name = os.path.basename(p)
        if walk_index is not None:
            try:
                if walk_index.is_done(a.country, img_name):
                    skip_done += 1
                    continue
            except Exception:
                pass
        if i % 20 == 0:
            write_progress("running", i, total)
            print("  [%d/%d] 成功 %d，跳过(无标牌) %d，跳过(太小) %d，跳过(已产出) %d，失败 %d"
                  % (i, total, ok, skip_no_sign, skip_small, skip_done, fail), flush=True)
            if skip_color:
                print("      颜色不一致跳过 %d 帧" % skip_color, flush=True)

        try:
            # 先查缓存：这张图扫过就直接用，不再跑 YOLO
            dets = None
            if walk_cache is not None:
                try:
                    dets = walk_cache.get_scan(p)
                    if dets is not None:
                        cache_hit += 1
                except Exception:
                    dets = None
            if dets is None:
                dets = detect_signs(p, a.conf)
                if walk_cache is not None:
                    try:
                        walk_cache.put_scan(p, dets)
                    except Exception:
                        pass
            if not dets:
                skip_no_sign += 1
                continue
            # 只处理足够大的标牌（太小合成看不出效果，也检不准）
            dets = [d for d in dets if min(d["w"], d["h"]) >= a.min_sign]
            if not dets:
                skip_small += 1
                continue

            # A detector may return several classes for the same physical sign.
            chosen = []
            for det in sorted(dets, key=lambda item: item["conf"], reverse=True):
                x1, y1, x2, y2 = det["bbox"]
                area = max(0, x2 - x1) * max(0, y2 - y1)
                duplicate = False
                for kept in chosen:
                    a1, b1, a2, b2 = kept["bbox"]
                    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
                    other_area = max(0, a2 - a1) * max(0, b2 - b1)
                    if min(area, other_area) > 0 and intersection / min(area, other_area) >= 0.65:
                        duplicate = True
                        break
                if not duplicate:
                    chosen.append(det)
            dets = chosen

            im = cv2.imread(p)
            if im is None:
                fail += 1
                continue

            # 严格形状过滤：只处理标准圆/方/三角，且素材形状必须一致。
            # 实测形状不匹配是穿帮的主因（圆贴三角、椭圆贴圆都会露边），
            # 宁可跳过，也不贴错。
            keep_dets = []
            for d in dets:
                # 尺寸门槛：太小的标牌 YOLO 分类不可靠，贴上去必然错
                _x1, _y1, _x2, _y2 = [int(v) for v in d["bbox"]]
                _w, _h = max(0, _x2 - _x1), max(0, _y2 - _y1)
                if min(_w, _h) < a.min_sign:
                    continue
                sh = std_shape(im, d["bbox"])
                if sh == "unknown":
                    continue
                # 颜色一致性：检测框主色与类别族必须匹配
                _crop = im[max(0, _y1):_y2, max(0, _x1):_x2]
                ok_color, _why = color_consistent(_crop, _crop, d.get("class") or d.get("cls"))
                if not ok_color:
                    skip_color += 1
                    continue
                want = "circle" if sh == "circle" else ("triangle" if sh == "triangle" else "rect")
                keep_dets.append((d, want))
            if not keep_dets:
                skip_small += 1     # 统计上并入「不适用形状」
                continue

            seq = os.path.basename(os.path.dirname(p))
            base = os.path.splitext(os.path.basename(p))[0]
            stem = "%s__%s" % (seq, base)
            H, W = im.shape[:2]
            objects = []
            out_im = im

            for d, want in keep_dets:
                # 指定标牌模式：只用指定的那一张，检测不到同族就跳过这帧
                if a.pin_class:
                    pin_nm = a.pin_class
                    pin_pp = table.get(pin_nm)
                    if not pin_pp:
                        # 允许不带扩展名
                        for nm, pp in table.items():
                            if nm.rsplit(".", 1)[0] == pin_nm:
                                pin_pp = pp
                                pin_nm = nm
                                break
                    if not pin_pp:
                        continue          # 没这个素材，跳过
                    # 检测框必须和指定标牌同族（大类匹配）
                    base_cls = pin_nm.rsplit(".", 1)[0]
                    fam = []
                    parts = base_cls.split("_")
                    for k in range(min(3, len(parts)), 0, -1):
                        fam.append("_".join(parts[:k]))
                    if not any(d["class"].startswith(f) for f in fam):
                        continue          # 不是同一族标牌，跳过
                    cls_name, path = base_cls, pin_pp
                else:
                    # 在该大类下找「形状匹配」的素材
                    cands = [(nm, pp) for nm, pp in table.items()
                             if (nm == d["class"] or nm.startswith(d["class"] + "_"))
                             and asset_shape(nm) == want]
                    if not cands:
                        continue
                    if a.vary_class:
                        import hashlib
                        h = int(hashlib.md5((a.images + str(d["class"])).encode()).hexdigest()[:8], 16)
                        cls_name, path = cands[h % len(cands)]
                    else:
                        cls_name, path = cands[0]
                s = load_sign(path)
                m = sign_mask(s)
                # 先把检测框收缩到原标牌实际轮廓，避免新牌比原牌大一圈
                # 旋转对齐：标牌在画面里有旋转角时，素材也要转同样角度。
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

                # 实测结论（14 个样本，用红色外圈做真值）：
                #   YOLO 框画的是「白色内圈」，不含红色外圈；
                #   红圈比框平均大 (+5.3, +11.0) px。
                #   所以贴图要按「框 × 1.12」放大，而不是缩小 ——
                #   缩小会让原牌红圈整圈露在外面（用户反馈的那种「重影」）。
                # --grow 控制放大倍数，默认 1.12。
                # 动态 grow：框刚好圈住红圈时（占比 >= 0.97）不放大，
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
                hh = (d["bbox"][3] - d["bbox"][1]) * 0.5 * g_use
                grown = [cx - hw, cy - hh, cx + hw, cy + hh]
                bbox = fit_bbox(grown, m.shape)
                # 底板方案试过并放弃：
                #   从原牌四角采样底板颜色，但四角往往是背景而不是底板，
                #   贴上去多出一块明显的灰方块，比不带底板更突兀。
                #   现在保持纯圆形贴图，退化到「只盖住牌面」。
                # ===== V4.6 封板渲染器（生产默认）=====
                # 圆环模板精定位（修贴歪）+ V4.4 亮度参数
                # + 条件化复合白板软钳位 + 窄羽化融合 + 车载 JPEG
                if a.v46 and not a.legacy_render:
                    try:
                        s_rgba46 = (np.dstack([s[:, :, :3], m])
                                    if s.shape[2] == 3 else s)
                        out_im, bb46, _, _ = _v46.render(
                            out_im, d["bbox"], s_rgba46)
                        objects.append({
                            "class": cls_name,
                            "orig_class": str(d.get("class") or ""),
                            "orig_conf": round(float(d["conf"]), 3),
                            "category": cls_name.split("_")[0],
                            "bbox": [round(float(v), 1) for v in bb46],
                            "source_bbox": [round(float(v), 1) for v in d["bbox"]],
                            "score": round(float(d["conf"]), 3),
                            "material": os.path.basename(path),
                            "renderer": "v4.11",
                        })
                        v46_count += 1
                        continue
                    except DoubleSignSkip:
                        v46_double_skip += 1
                        continue
                    except Exception as e:
                        v46_fail += 1
                        continue

                # ===== 6 步严格流水线（回退选项）=====
                if a.pipe6 and not a.legacy_pipe:
                    # 这里必须现算 dst_shape —— 之前误用了 QC 分支里的 sh0，
                    # 那个变量在当前作用域根本不存在，导致 NameError。
                    sh_dst = std_shape(im, d["bbox"])
                    if sh_dst == "unknown":
                        sh_dst = "circle"
                    s_rgba = np.dstack([s[:, :, :3], m]) if s.shape[2] == 3 else s
                    out_im, p6 = run_pipeline6(
                        out_im, d["bbox"], s_rgba, sh_dst,
                        jpeg_quality=a.pipe6_jpeg, pad_ratio=a.pipe6_pad)
                    if not p6.get("ok"):
                        p6_fail += 1
                        p6_reasons[p6.get("reason", "unknown").split("_")[0]] = \
                            p6_reasons.get(p6.get("reason", "unknown").split("_")[0], 0) + 1
                        continue
                    bbox_used = p6["bbox"]
                    objects.append({
                        "class": cls_name,
                        "orig_class": str(d.get("class") or ""),
                        "orig_conf": round(float(d["conf"]), 3),
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
                    "orig_class": str(d.get("class") or ""),
                    "orig_conf": round(float(d["conf"]), 3),
                    "category": cls_name.split("_")[0],
                    "bbox": [round(float(v), 1) for v in bbox],
                    "score": round(float(d["conf"]), 3),
                    "material": os.path.basename(path),
                    "occluder_ratio": cinfo.get("occluder_ratio", 0),
                })

            if not objects:
                skip_no_sign += 1
                continue

            # 保留原有真实标牌：只标新目标、忽略原图已有目标，
            # 会把真实目标变成训练负样本，是 P/R 下降的头号来源。
            # 这里把未被替换的检出框作为 existing_objects 保留，
            # 并标记替换过的目标，供训练端合并标签使用。
            #
            # FIX: 去重必须用 source_bbox（原检测框），不能用 bbox（bb46 渲染后）。
            #      bbox 会因渲染微调偏移 1~3px，与原检测框永远对不上，
            #      导致同一目标同时出现在 objects 和 existing_objects 里。
            def _overlap(a, b, thr=0.5):
                """两个 xyxy 框的 IoU。"""
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                if inter <= 0:
                    return 0.0
                aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                return inter / max(aa + ab - inter, 1e-9)

            _replaced_boxes = []
            for ob in objects:
                sb = ob.get("source_bbox") or ob.get("bbox") or []
                if len(sb) == 4:
                    _replaced_boxes.append([float(v) for v in sb])

            existing_objects = []
            _dup_skipped = 0
            for _d in dets:
                _db = [float(v) for v in _d.get("bbox") or []]
                if len(_db) != 4:
                    continue
                # 与原检测框做 IoU 比较（同精度）
                if any(_overlap(_db, rb) >= 0.50 for rb in _replaced_boxes):
                    _dup_skipped += 1
                    continue          # 已被替换，由 objects 承担
                existing_objects.append({
                    "class": str(_d.get("class") or _d.get("cls") or ""),
                    "bbox": [round(v, 1) for v in _db],
                    "score": round(float(_d.get("conf") or 0), 3),
                    "source": "detected_unchanged",
                    # 未经人工确认，训练端可据此决定是否计入负样本
                    "verified": False,
                })
            if _dup_skipped:
                print("    去重: 跳过 %d 个已替换目标" % _dup_skipped, flush=True)

            # 自动质检：任一标牌不过就不写盘，直接计入 qc_reject
            if a.qc:
                frame_ok = True
                why = ""
                # P0-FIX: 不再用 zip 按列表位置配对（跳过目标后会错配）。
                # 逐框独立检查：每个 ob 用自己的 bbox 验证像素质量，
                # 形状取该框自身在原图上的判定。
                for ob in objects:
                    try:
                        sh0 = std_shape(im, ob["bbox"])
                    except Exception:
                        sh0 = "unknown"
                    if sh0 == "unknown":
                        sh0 = "circle"
                    ok0, info0 = qc_check(im, out_im, ob["bbox"], sh0,
                                          edge_jump_max=a.qc_edge,
                                          bright_diff_max=a.qc_bright)
                    if not ok0:
                        frame_ok = False
                        why = info0.get("reason", "unknown")
                        break
                if not frame_ok:
                    # 寻优：换 grow/brightness 再试，能救回来就用救回来的
                    if a.tune:
                        tuned_ok = False
                        for d0 in [x[0] for x in keep_dets]:
                            sh0 = std_shape(im, d0["bbox"])
                            if sh0 == "unknown":
                                sh0 = "circle"
                            cands0 = [(nm, pp) for nm, pp in table.items()
                                      if (nm == d0["class"] or nm.startswith(d0["class"] + "_"))
                                      and asset_shape(nm) == ("triangle" if sh0 == "triangle"
                                                              else "rect" if sh0 == "square" else "circle")]
                            if not cands0:
                                continue
                            _, path0 = cands0[0]
                            s0 = load_sign(path0)
                            m0 = sign_mask(s0)
                            g_, bg_, info_, comp_ = best_params_for_frame(
                                im, s0[:, :, :3], m0, d0["bbox"], sh0,
                                stop_on_first_pass=False)
                            if info_.get("passed"):
                                out_im = comp_
                                tuned_ok = True
                                tune_saved += 1
                                break
                        if not tuned_ok:
                            qc_reject += 1
                            qc_reasons[why.split("_")[0]] = qc_reasons.get(why.split("_")[0], 0) + 1
                            continue
                    else:
                        qc_reject += 1
                        qc_reasons[why.split("_")[0]] = qc_reasons.get(why.split("_")[0], 0) + 1
                        continue

            # SD 局部重绘：让模型按周围环境重新打光，去掉"贴纸感"
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

            # Gate4：模型自检 —— 合成图喂回 YOLO。
            # 取消硬门槛：困难但标签可靠的样本正是提升 Recall 的来源，
            # 检测不到不等于没有训练价值。这里只记录分数供筛选，不丢帧。
            if a.gate4:
                gate4_scores = []
                for ob in objects:
                    try:
                        ok4, info4 = gate4_model_check(
                            out_im, ob["bbox"], ob["class"], conf_min=a.gate4_conf)
                        info4 = info4 or {}
                        # gate4_model_check 返回 pred_conf / model_check 字段
                        _conf = info4.get("pred_conf")
                        _conf = float(_conf) if _conf is not None else None
                        gate4_scores.append({
                            "ok": bool(ok4),
                            "conf": _conf,
                            "model_check": info4.get("model_check"),
                        })
                        ob["gate4_ok"] = bool(ok4)
                        ob["gate4_conf"] = (round(_conf, 3) if _conf is not None else None)
                        ob["gate4_model_check"] = info4.get("model_check")
                    except Exception:
                        gate4_scores.append({"ok": None, "conf": None})
                        ob["gate4_ok"] = None
                # 只统计，不拒绝
                if gate4_scores and not all(g.get("ok") for g in gate4_scores):
                    g4_soft_low += 1

            out_jpg = os.path.join(a.out, stem + "_composed.jpg")

            # 先编码，检查实际将交付的像素，避免检查通过后再次压缩改变结果
            encode_ok, encoded = cv2.imencode(
                ".jpg", out_im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not encode_ok:
                raise RuntimeError("JPEG encoding failed")

            delivered = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if delivered is None:
                raise RuntimeError("JPEG decoding failed")

            qstatus, quality_records = check_frame(im, delivered, objects)

            if qstatus == "failed":
                qc_reject += 1
                print(json.dumps({"source": p,
                                  "quality_reject": quality_records},
                                 ensure_ascii=False), flush=True)
                continue

            if qstatus == "unchecked":
                qc_unchecked += 1

            # 保存的就是刚刚验收过的字节，不再二次编码
            with open(out_jpg, "wb") as output_file:
                output_file.write(encoded.tobytes())
            # 写入增量索引（国家 + 图片名）
            if walk_index is not None:
                try:
                    walk_index.mark(a.country, img_name, "done",
                                      src_path=p, out_path=out_jpg,
                                      cls=(objects[0].get("orig_class") if objects else ""),
                                      conf=(objects[0].get("orig_conf") if objects else None),
                                      bbox=(objects[0].get("bbox") if objects else None),
                                      material=(objects[0].get("material") if objects else ""),
                                      sim_cls=(objects[0].get("class") if objects else ""))
                except Exception:
                    pass
            rec = {
                "image": os.path.basename(out_jpg),
                "source_image": os.path.basename(p),
                "source_dir": seq,
                "source_path": p,
                "width": W, "height": H,
                "simulation_params": dict(effective_params),
                "bbox_grow": a.grow,
                "objects": objects,
                "existing_objects": existing_objects,
                # 标签完整性：只有检测提议，未做真实标签合并
                "label_completeness": "detector_proposals_only",
                "quality": {
                    "policy": "pixel_and_position_v1",
                    "passed": qstatus == "passed",
                    "status": qstatus,
                    "semantic_status": "not_checked",
                    "objects": quality_records,
                },
                # 明确声明训练资格
                "training_eligible": False,
                "training_blockers": [
                    "no_real_label_merge",
                    "no_multi_camera_temporal",
                    "semantic_not_checked",
                ],
            }
            with open(os.path.join(ann_dir, stem + ".json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            summary.append({"image": rec["image"], "n": len(objects),
                            "classes": [o["class"] for o in objects]})
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print("    [ERR] %s: %s" % (os.path.basename(p), str(e)[:70]), flush=True)

    dt = time.time() - t0
    print("[3/3] 汇总…", flush=True)
    cls_count = {}
    for s in summary:
        for c in s["classes"]:
            cls_count[c] = cls_count.get(c, 0) + 1
    rep = {
        "country": a.country,
        "root": a.images,
        "scanned_files": total,
        "limit": a.limit,
        "ok": ok, "skip_no_sign": skip_no_sign,
        "skip_small": skip_small, "fail": fail,
        "elapsed_sec": round(dt, 1),
        "min_sign_px": a.min_sign,
        "vary_class": a.vary_class,
        "qc_enabled": a.qc,
        "qc_reject": qc_reject,
        "qc_unchecked": qc_unchecked,
        "gate4_soft_low": g4_soft_low,
        "tune_saved": tune_saved,
        "gate4_enabled": a.gate4,
        "gate4_reject": g4_reject,
        "rotate_applied": rot_applied,
        "pipe6_fail": p6_fail,
        "v46_rendered": v46_count,
        "v46_fail": v46_fail,
        "v46_double_skip": v46_double_skip,
        "cache_hit": cache_hit,
        "pipe6_fail_reasons": p6_reasons,
        "sd_enabled": a.sd,
        "sd_refined": sd_count,
        "sd_fail": sd_fail,
        "qc_reject_reasons": qc_reasons,
        "simulation_params": dict(effective_params),
        "bbox_grow": a.grow,
        "class_distribution": dict(sorted(cls_count.items(), key=lambda kv: -kv[1])),
        "items": summary,
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    write_progress("done", total, total)

    print("-" * 56)
    print("扫描 %d 个文件，生成 %d 张（%.0f 秒）" % (total, ok, dt))
    print("跳过：无标牌 %d，标牌太小 %d；失败 %d"
          % (skip_no_sign, skip_small, fail))
    if a.qc:
        if a.tune:
            print("寻优救回 %d 张" % tune_saved)
    if a.gate4:
        print("Gate4 模型自检：低于阈值 %d 张（仅记录，不剔除）" % g4_soft_low)
    if not a.no_rotate:
        print("旋转对齐 %d 个标牌" % rot_applied)
    if a.v46 and not a.legacy_render:
        print("V4.11 渲染 %d 个（双牌跳过 %d，失败 %d）" % (v46_count, v46_double_skip, v46_fail))
    if a.pipe6 and not a.legacy_pipe:
        print("6 步流水线失败 %d 个%s" % (p6_fail,
              ("（%s）" % ", ".join("%s:%d" % kv for kv in p6_reasons.items())) if p6_reasons else ""))
    if a.sd:
        print("SD 精修 %d 张（失败 %d）" % (sd_count, sd_fail))
        print("质检剔除 %d 张%s" % (qc_reject,
              ("（%s）" % ", ".join("%s:%d" % kv for kv in qc_reasons.items())) if qc_reasons else ""))
    print("类别分布 Top10:")
    for c, n in list(rep["class_distribution"].items())[:10]:
        print("  %-28s %4d" % (c, n))
    print("输出: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
