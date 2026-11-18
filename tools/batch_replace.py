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
from sign_realism import paste_realistic                     # noqa: E402

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


def detect_shape(image, bbox):
    """判断原标牌的形状：circle / triangle / rect。

    做法：在 bbox 内做颜色分割（标牌通常是高饱和或高对比区域），
    取最大轮廓，按轮廓的「填充率 + 近似多边形顶点数」判定。

    为什么必须判形状：实测圆形图案贴到三角形标牌上时，三个角会露出来，
    一眼就能看出是假的。形状不匹配就不该替换。
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
        return "unknown"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 用 Otsu 分割出标牌主体（亮底 vs 暗环境，或反过来）
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 取包含中心的那一半
    h, w = th.shape
    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "unknown"
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    bx, by, bw, bh = cv2.boundingRect(c)
    if bw < 6 or bh < 6:
        return "unknown"
    fill = area / float(bw * bh)

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    n = len(approx)

    # 判定顺序：先三角（填充率 0.4~0.6），再圆（填充率 ~0.78），最后矩形
    aspect = bw / float(bh)
    if n == 3 or (0.35 <= fill <= 0.60 and 0.8 <= aspect <= 1.25):
        return "triangle"
    if fill >= 0.68 and 0.80 <= aspect <= 1.25:
        return "circle"
    return "rect"


def shape_matches(sign_shape, target_shape):
    """图案形状是否允许贴到该目标上 —— **严格同形状**。

    为什么严格：
      1) 几何上必然正确：圆对圆边缘完全重合，三角对三角三个角精确对应，不会露边露角。
         之前宽松放行「矩形目标」时，实测圆图案贴到椭圆广告牌上露出一圈椭圆边。
      2) 符合真实：道路标牌形状由国标规定（圆形=限速/禁止、三角=警告、矩形=指路），
         同一位置换同形状的牌，物理上就是合理的。
      3) 训练安全：硬把圆图案裁成三角形会产生现实中不存在的形状，反而给模型带偏。
    判不出目标形状时（unknown）不拦，避免因为检测失败而全部跳过。
    """
    if target_shape == "unknown":
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
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少张图")
    ap.add_argument("--scale", type=float, default=1.0, help="粘贴区域相对检测框的倍数")
    ap.add_argument("--strength", type=float, default=1.0, help="劣化强度 0~1.5")
    ap.add_argument("--sat-boost", type=float, default=1.18, help="饱和度提升（交通标志建议 1.1~1.25）")
    ap.add_argument("--min-px", type=int, default=24, help="标牌小于该像素宽就跳过")
    ap.add_argument("--all-signs", action="store_true", help="一张图里所有标牌都替换")
    ap.add_argument("--no-annotate", action="store_true", help="不输出标注")
    ap.add_argument("--no-shape-check", action="store_true", help="不校验形状匹配")
    a = ap.parse_args()

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
    summary = {"mode": mode, "country": a.country, "total": len(images), "ok": 0, "skipped": 0,
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
                else:
                    path = tsr_mod.material_for(d["class"], sign_table)
                    if not path:
                        rec["skipped"].append("%s 无匹配素材" % d["class"])
                        continue
                    sign_img = load_sign(path)
                    cls_name = tsr_mod.normalize_class_name(
                        os.path.splitext(os.path.basename(path))[0])

                sm = sign_mask(sign_img)
                # 形状约束：原标牌是三角形就不贴圆形图案（否则角会露出来）
                tgt_shape = detect_shape(im, d["bbox"])
                sign_shape = tsr_mod.shape_of(cls_name) if hasattr(tsr_mod, "shape_of") else "circle"
                if not shape_matches(sign_shape, tgt_shape):
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
                composed, soft = paste_realistic(base, sign_img[:, :, :3], sm, bbox,
                                                 strength=a.strength)
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
