#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_replace.py — 批量 TSR 图案替换 + 自动标注（可直接当训练数据）

三种用法：
  1) 一对多：一个 TSR，替换到 N 张场景图里的标牌
        python3 batch_replace.py --signs speed_limit_60.png --images 场景目录/ --out 输出/
  2) 多对多：素材目录按类别自动匹配不同场景
        python3 batch_replace.py --signs-dir 素材目录/ --images 场景目录/ --out 输出/
        （文件名约定 <大类>_<子类>.png，如 speed_limit_60.png；类别对不上就跳过）

产出（每张成功替换的图）：
  <name>_composed.jpg    合成图（训练用图）
  <name>_detect.jpg      检测可视化（人工核对用）
  <name>.json            标注（训练用标签，VOC/COCO 同款字段）
输出目录根部：
  annotations/           所有 JSON 汇总在这里（便于直接喂训练脚本）
  summary.json           成功/跳过/失败明细

标注口径：
  - 替换后的标牌 bbox 用**实际粘贴区域**（不是原检测框），因为缩放后会有偏移
  - class 用规范化类别名（speed_limit_60），与 modules/tsr.py 的类别表一致
  - 同时给出 category（大类），便于把细类归并到大类训练
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2                                                    # noqa: E402
from replace_sign import detect_signs, load_sign, sign_mask  # noqa: E402
from sign_realism import paste_realistic, make_realistic     # noqa: E402
from sign_occlusion import paste_with_occlusion              # noqa: E402
from sign_realism2 import paste_v2                           # noqa: E402

# 复用 modules/tsr.py 的类别工具（保持单一事实来源）
_MODULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules")
if os.path.isdir(_MODULES):
    sys.path.insert(0, _MODULES)
try:
    import tsr as tsr_mod
except Exception:                      # 没有 modules 目录时退化为最小实现
    class _Fallback:
        @staticmethod
        def normalize_class_name(n):
            return n

        @staticmethod
        def category_of(n):
            return str(n).split("_")[0]

        @staticmethod
        def class_label(n):
            return n

        @staticmethod
        def shape_of(n):
            return "circle"

        @staticmethod
        def material_for(cat, table):
            return table.get(cat)
    tsr_mod = _Fallback()

IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_images(path, limit=0):
    if os.path.isfile(path):
        return [path]
    out = []
    for f in sorted(os.listdir(path)):
        p = os.path.join(path, f)
        if os.path.isfile(p) and f.lower().endswith(IMG_EXTS):
            out.append(p)
    return out[:limit] if limit > 0 else out


def list_signs(signs_dir, country=""):
    """素材目录 → {规范化类别名: 路径}。

    支持两种布局：
      1) 扁平：素材目录/*.png
      2) 按国家分层：素材根/<CC>/<类别>.png（下面还有 _common）
    传了 country 时优先取该国的素材，找不到再退回 _common。
    """
    raw = {}

    # 按国家分层布局
    if country:
        for cc in (country.upper(), "_common"):
            d = os.path.join(signs_dir, cc)
            if not os.path.isdir(d):
                # 兼容 MTSD 的 <CC>_<中文名> 命名
                for name in sorted(os.listdir(signs_dir)) if os.path.isdir(signs_dir) else []:
                    if name.upper() == cc or name.upper().startswith(cc + "_"):
                        d = os.path.join(signs_dir, name)
                        break
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if os.path.isfile(p) and f.lower().endswith((".png", ".jpg", ".jpeg")):
                    name = tsr_mod.normalize_class_name(os.path.splitext(f)[0])
                    raw.setdefault(name, p)      # 国家优先，先到先得

    # 扁平布局（或没有国家时的兜底）
    if os.path.isdir(signs_dir):
        for f in sorted(os.listdir(signs_dir)):
            p = os.path.join(signs_dir, f)
            if os.path.isfile(p) and f.lower().endswith((".png", ".jpg", ".jpeg")):
                name = tsr_mod.normalize_class_name(os.path.splitext(f)[0])
                raw.setdefault(name, p)
    return raw


# 每个大类的中性代表子类：只检测到大类时用它，避免把「限高5.5m」贴到「禁止右转」的位置
NEUTRAL_SUBTYPE = {
    "speed_limit": "60",
    "prohibition": "no_entry",
    "warning": "bump",
    "mandatory": "go_straight",
    "guide": "direction",
    "signal": "red",
    "crosswalk": "zebra",
}

# 兜底顺序（越靠前越中性、越常见）
NEUTRAL_FALLBACK = ["no_entry", "stop", "no_parking", "no_stopping",
                    "60", "50", "80", "bump", "pedestrian", "red"]

# 形状是否严格校验（默认 False：形状只作参考，不做硬约束）
STRICT_SHAPE = False


# 类别 → 形状（交通标志的国际通用规则，比图像检测可靠得多）
# 在 40~80px 的小标牌上，任何图像形状检测都不准（实测准确率 13%），
# 但「类别→形状」是设计规范，不会错。
CLASS_SHAPE = {
    "speed_limit": "circle",
    "prohibition": "circle",
    "mandatory": "circle",
    "signal": "circle",
    "crosswalk": "circle",
    "warning": "triangle",
    "guide": "rect",
}

# 素材名 → 形状（用于按形状筛选素材）
def asset_shape(name):
    base = name.split("_x")[0]      # 去掉 _x2 之类的补充分类后缀
    for k, v in CLASS_SHAPE.items():
        if base.startswith(k):
            return v
    return "unknown"


def pick_varied_asset(cls, sign_table, seed=None):
    """在大类下随机挑一个子类，用于让批量产出覆盖多样化类别。

    为什么需要：
      原来的 pick_neutral_asset 对同一个大类永远返回同一个子类
      （比如 speed_limit 永远给 speed_limit_60），
      跑 50 张出来全是「限速60」，数据集严重偏斜。

    做法：
      收集该大类下所有可用子类（speed_limit_30/40/60/80...），
      用 seed（通常是文件名哈希）稳定随机挑一个 ——
      同一张图重跑结果一致，不同图之间分布均匀。

    返回 (类别名, 路径) 或 (None, None)。
    """
    want_shape = CLASS_SHAPE.get(cls, "")
    cands = [(nm, p) for nm, p in sign_table.items()
             if nm == cls or nm.startswith(cls + "_") or nm.startswith(cls)]
    # 只保留同形状的（类别推出的形状），形状不符的素材一律不用
    if want_shape:
        same = [(nm, p) for nm, p in cands if asset_shape(nm) == want_shape]
        if same:
            cands = same
    if not cands:
        return None, None
    if len(cands) == 1:
        return cands[0]
    import hashlib
    if seed is None:
        import random
        return random.choice(cands)
    h = int(hashlib.md5(str(seed).encode("utf-8")).hexdigest()[:8], 16)
    return cands[h % len(cands)]


def pick_neutral_asset(cls, sign_table):
    """为「只知道大类」的情况挑一个中性代表素材。

    优先级：NEUTRAL_SUBTYPE 指定子类 → NEUTRAL_FALLBACK 里能命中的 → 该大类第一个。
    返回 (类别名, 路径) 或 (None, None)。
    """
    want = NEUTRAL_SUBTYPE.get(cls)
    if want:
        exact = "%s_%s" % (cls, want)
        if exact in sign_table:
            return exact, sign_table[exact]
    for sub in NEUTRAL_FALLBACK:
        exact = "%s_%s" % (cls, sub)
        if exact in sign_table:
            return exact, sign_table[exact]
    for nm, path in sign_table.items():
        if nm == cls or nm.startswith(cls + "_"):
            return nm, path
    return None, None


def detect_shape(image, bbox):
    """判断原标牌形状：circle / triangle / rect / unknown

    已试过并放弃的方案：
      1) 填充率（contour area / bbox area）区间判定 ——
         实测同一块圆牌连续帧在 0.51~0.95 之间跳，40~50px 的牌判不准
      2) 两向 Otsu 取最大 fill —— 反相结果会把圆判成矩形，更糟

    现在的做法：圆形度 circularity = 4πA / P²
        理论值：圆 1.00、正方形 0.785、正三角形 0.605
      这是尺度无关的指标，比填充率稳。
      同时保留宽高比作为三角的辅助判据（三角高宽比约 0.87）。
      判不准时返回 unknown —— 调用方对 unknown 一律放行，
      宁可不拦，也不要乱拦（之前误拦导致一半场景被跳过）。
    """
    import cv2
    import math

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 12 or roi.shape[1] < 12:
        return "unknown"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 10), max(3, h // 10)))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "unknown"
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    if area < 40 or per < 10:
        return "unknown"

    bx, by, bw, bh = cv2.boundingRect(c)
    if bw < 10 or bh < 10:
        return "unknown"

    circ = 4.0 * math.pi * area / (per * per + 1e-6)
    aspect = bw / float(bh)
    # 顶点数：近似多边形
    approx = cv2.approxPolyDP(c, 0.04 * per, True)
    nv = len(approx)

    # 圆：圆形度接近 1，宽高比接近 1
    if circ >= 0.82 and 0.80 <= aspect <= 1.25:
        return "circle"
    # 三角：圆形度明显低（0.55~0.72），且顶点数 <= 4，宽高比偏宽
    if 0.45 <= circ <= 0.75 and nv <= 5 and 0.70 <= aspect <= 1.45:
        return "triangle"
    # 矩形：圆形度 0.75~0.85 且是方块
    if circ >= 0.75 and 0.78 <= aspect <= 1.28:
        return "rect"
    # 其余判不准，交给调用方放行
    return "unknown"


def _classify(fill, aspect):
    """按填充率与宽高比分类。"""
    # 三角：填充率明显低（理论 0.5）
    if fill < 0.62:
        return "triangle"
    # 圆：填充率在 π/4 附近，且接近正方
    if fill <= 0.90 and 0.70 <= aspect <= 1.40:
        return "circle"
    # 其余按矩形（fill 接近 1）
    return "rect"


def shape_matches(sign_shape, target_shape, user_shape=""):
    """形状是否允许替换 —— 形状只作参考，不做硬约束。

    为什么不做硬约束：
      实测在 40~50 像素的小标牌上，用填充率判形状并不可靠 ——
      同一块圆牌连续 6 帧会在 circle / rect 之间跳变，
      按「严格同形状」会误跳过一半以上场景，反而更差。

    现在的策略：
      1) 默认一律放行（不拦）。形状作为参考信息写进标注（scene_shape 字段），
         训练时可以按需筛。
      2) 提供 --strict-shape 开关，需要严格同形状时才拦。
         用户明确指定 --shape 时，也只在 strict 模式下才拦。
    """
    if not STRICT_SHAPE:
        return True
    if user_shape:
        return sign_shape == user_shape
    if target_shape in ("", "unknown"):
        return True
    return sign_shape == target_shape


def fit_bbox(bbox, sign_mask_shape, keep_ratio=True):
    """按图案长宽比把检测框收缩成实际粘贴区域，返回新 bbox。"""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    sh, sw = sign_mask_shape[:2]
    if not keep_ratio or sw <= 0 or sh <= 0:
        return [x1, y1, x2, y2]
    r = min(bw / sw, bh / sh)
    nw, nh = sw * r, sh * r
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return [cx - nw * 0.5, cy - nh * 0.5, cx + nw * 0.5, cy + nh * 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="场景图目录或单张图")
    ap.add_argument("--signs", default="", help="单个 TSR 图案（一对多模式）")
    ap.add_argument("--signs-dir", default="", help="素材目录（多对多模式，按类别匹配）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--country", default="",
                    help="国家代码（BR/RU/DE…），按国家分层素材库时指定")
    ap.add_argument("--vary-class", action="store_true",
                    help="大类下随机挑子类（避免全部贴成同一类，批量更均衡）")
    ap.add_argument("--shape", default="", choices=["", "circle", "triangle", "rect"],
                    help="用户指定标牌形状：circle/triangle/rect。给了就按它匹配，不再用自动检测")
    ap.add_argument("--target-class", default="",
                    help="指定要造的类别（如 speed_limit_60）。给了就把所有检测到的标牌"
                         "都换成这一类；不给则按大类挑中性代表素材")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少张图")
    ap.add_argument("--scale", type=float, default=1.0, help="粘贴区域相对检测框的倍数")
    ap.add_argument("--strength", type=float, default=1.0, help="劣化强度 0~1.5")
    ap.add_argument("--sat-boost", type=float, default=1.18, help="饱和度提升（交通标志建议 1.1~1.25）")
    ap.add_argument("--min-px", type=int, default=24, help="标牌小于该像素宽就跳过")
    ap.add_argument("--all-signs", action="store_true", help="一张图里所有标牌都替换")
    ap.add_argument("--no-annotate", action="store_true", help="不输出标注")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="不保留前景遮挡（默认保留：立柱/树枝不会被盖住）")
    ap.add_argument("--bright-gain", type=float, default=None,
                    help="目标亮度 = 环境亮分位 × 该值（0.5~1.2，默认 0.95）")
    ap.add_argument("--haze-gain", type=float, default=None,
                    help="雾度强度（0~1.5，默认 1.0，调小更实）")
    ap.add_argument("--contrast-gain", type=float, default=None,
                    help="对比度倍数（0.6~1.5，默认 1.05，调小更灰）")
    ap.add_argument("--satur-keep", type=float, default=None,
                    help="雾天饱和度保留（0.3~1.0，默认 0.70）")
    ap.add_argument("--feather-ratio", type=float, default=None,
                    help="边缘羽化比例（0.02~0.15，默认 0.06）")
    ap.add_argument("--no-shadow", action="store_true",
                    help="不投阴影（默认按场景光照投）")
    ap.add_argument("--strict-shape", action="store_true",
                    help="严格同形状（默认不拦，形状只作参考写入标注）")
    a = ap.parse_args()

    global STRICT_SHAPE
    STRICT_SHAPE = bool(a.strict_shape)
    try:
        from sign_realism2 import set_params as _set_rp
        _set_rp(bright_gain=a.bright_gain, haze_gain=a.haze_gain,
                contrast_gain=a.contrast_gain, satur_keep=a.satur_keep,
                feather_ratio=a.feather_ratio)
    except Exception:
        pass

    if not a.signs and not a.signs_dir:
        raise SystemExit("必须给 --signs 或 --signs-dir")

    ann_dir = os.path.join(a.out, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    single = None
    single_name = ""
    if a.signs:
        single = load_sign(a.signs)
        single_name = tsr_mod.normalize_class_name(
            os.path.splitext(os.path.basename(a.signs))[0])
    sign_table = list_signs(a.signs_dir, a.country) if a.signs_dir else {}
    if a.signs_dir and not sign_table:
        raise SystemExit("素材目录里没有可用图案: %s" % a.signs_dir)

    images = list_images(a.images, a.limit)
    if not images:
        raise SystemExit("场景目录里没有图片: %s" % a.images)

    mode = "multi" if a.signs_dir else "single"
    summary = {"mode": mode, "country": a.country, "target_class": a.target_class, "shape": a.shape, "total": len(images), "ok": 0, "skipped": 0,
               "failed": 0, "annotated": 0,
               "params": {"conf": a.conf, "scale": a.scale, "strength": a.strength,
                          "sat_boost": a.sat_boost, "min_px": a.min_px},
               "items": []}
    print("待处理 %d 张图，模式=%s" % (len(images), mode))

    for i, img_path in enumerate(images):
        name = os.path.splitext(os.path.basename(img_path))[0]
        rec = {"image": img_path, "replaced": [], "skipped": []}
        try:
            dets = detect_signs(img_path, a.conf)
            usable = [d for d in dets if d["w"] >= a.min_px]
            if not usable:
                rec["skipped"].append("无可用检测")
                summary["skipped"] += 1
                summary["items"].append(rec)
                continue

            targets = usable if a.all_signs else [usable[0]]
            im = cv2.imread(img_path)
            composed = None
            annotations = []

            for d in targets:
                if single is not None:
                    sign_img, cls_name = single, single_name
                elif a.target_class:
                    # B) 人工指定：所有检测到的标牌都换成这一类
                    path = sign_table.get(a.target_class)
                    if not path:
                        rec["skipped"].append("指定类别 %s 不在素材库" % a.target_class)
                        continue
                    sign_img = load_sign(path)
                    cls_name = a.target_class
                else:
                    # A) 中性代表：按大类挑最中性的子类，不做无依据的猜测
                    if a.vary_class:
                        cls_name, path = pick_varied_asset(d["class"], sign_table,
                                                            seed=os.path.basename(img_path))
                    else:
                        cls_name, path = pick_neutral_asset(d["class"], sign_table)
                    if not path:
                        rec["skipped"].append("%s 无匹配素材" % d["class"])
                        continue
                    sign_img = load_sign(path)

                sm = sign_mask(sign_img)
                # 形状约束：原标牌是三角形就不贴圆形图案（否则角会露出来）
                tgt_shape = detect_shape(im, d["bbox"])
                sign_shape = tsr_mod.shape_of(cls_name) if hasattr(tsr_mod, "shape_of") else "circle"
                if not shape_matches(sign_shape, tgt_shape, a.shape):
                    rec["skipped"].append("%s(图案%s) vs 场景%s 形状不符"
                                          % (cls_name, sign_shape, tgt_shape))
                    continue

                bbox = fit_bbox(d["bbox"], sm.shape, keep_ratio=True)
                if a.scale != 1.0:
                    cx, cy = (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5
                    hw = (bbox[2] - bbox[0]) * 0.5 * a.scale
                    hh = (bbox[3] - bbox[1]) * 0.5 * a.scale
                    bbox = [cx - hw, cy - hh, cx + hw, cy + hh]

                base = composed if composed is not None else im
                # 用带遮挡保留 + 投影阴影的合成：
                #   原图里挡在标牌前面的立柱/树枝不会被新标牌盖掉，
                #   并按场景光照方向投出渐隐阴影，避免「硬贴」感
                # 优先用 v2（环境氛围匹配 + 边缘过渡）：
                #   亮度向「环境亮分位」对齐而不是标牌暗部，
                #   并在雾天自动抬暗部、降对比，解决「像贴纸」的问题
                composed, cinfo = paste_v2(
                    base, sign_img[:, :, :3], sm, bbox,
                    strength=a.strength, edge_glow=True)
                if not cinfo.get("ok"):
                    def _deg(sig, alp, ref):
                        return make_realistic(sig, alp, ref, strength=a.strength)
                    composed, cinfo = paste_with_occlusion(
                        base, sign_img[:, :, :3], sm, bbox, degrade_fn=_deg,
                        shadow=not a.no_shadow, strength=a.strength)
                soft = cinfo if cinfo.get("ok") else None
                if soft is None:
                    rec["skipped"].append("bbox 过小")
                    continue

                rec["replaced"].append({"class": cls_name, "conf": d["conf"],
                                        "bbox": [round(v, 1) for v in bbox]})
                annotations.append({
                    "class": cls_name,
                    "category": tsr_mod.category_of(cls_name),
                    "label": tsr_mod.class_label(cls_name),
                    "bbox": [round(v, 1) for v in bbox],
                    "detected_as": d["class"],
                    "scene_shape": tgt_shape,
                    "det_conf": d["conf"],
                })

            if rec["replaced"]:
                out_img = os.path.join(a.out, name + "_composed.jpg")
                cv2.imwrite(out_img, composed, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if not a.no_annotate and annotations:
                    h, w = composed.shape[:2]
                    ann = {"image": os.path.basename(out_img),
                           "source_image": os.path.basename(img_path),
                           "width": w, "height": h,
                           "objects": annotations}
                    with open(os.path.join(ann_dir, name + ".json"), "w",
                              encoding="utf-8") as f:
                        json.dump(ann, f, indent=1, ensure_ascii=False)
                    summary["annotated"] += 1

                summary["ok"] += 1
                print("  [%d/%d] %s -> 替换 %d 个，标注 %d 条"
                      % (i + 1, len(images), name, len(rec["replaced"]), len(annotations)))
            else:
                summary["skipped"] += 1
                print("  [%d/%d] %s -> 跳过" % (i + 1, len(images), name))
        except Exception as e:
            rec["error"] = str(e)[:200]
            summary["failed"] += 1
            print("  [%d/%d] %s -> 失败: %s" % (i + 1, len(images), name, str(e)[:80]))
        summary["items"].append(rec)

    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    print("-" * 55)
    print("完成: 成功 %d / 跳过 %d / 失败 %d（共 %d），标注 %d 份"
          % (summary["ok"], summary["skipped"], summary["failed"],
             summary["total"], summary["annotated"]))
    print("输出: %s" % a.out)
    print("标注: %s" % ann_dir)
    return 0 if summary["ok"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
