#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_library_mtsd.py — 从 MTSD 构建「按国家分层」的标牌素材库

数据来源（网盘已有，无需下载）：
  /mnt/Data_Platform/zhj_datamining/Oversea_data_test/MTSD_Tsr/
    mtsd_by_country/<CC>_<中文名>/*.jpg       各国标牌裁切图
    mtsd_fully/annotations/<首字符>/<key>.json  全量标注（label + bbox + 遮挡标记）

产出结构：
  <out>/
    BR/speed_limit_60.png        巴西素材（带 alpha）
    BR/prohibition_no_right.png
    RU/warning_pedestrian.png
    _common/...                  无法归入某国但质量好的
    _report.json                 统计

类别映射（MTSD 层级 → 我们的类别名）：
  speed-limit--60--g1              → speed_limit_60
  regulatory--no-entry--g1         → prohibition_no_entry
  warning--pedestrians-crossing--g4 → warning_pedestrian
映射表见下面 LABEL_RULES，未命中的标签跳过（不做猜测）。

用法：
  python3 build_library_mtsd.py --mtsd <MTSD根> --out <素材库根> [--countries BR,RU,AR]
                                [--per-class 3] [--min-px 40]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules"))
try:
    import asset_library as alib
except Exception:
    alib = None

# MTSD 真实标签格式（实测，非猜测）：
#   regulatory--maximum-speed-limit-60--g1     限速60
#   regulatory--no-entry--g1                   禁止驶入
#   warning--pedestrians-crossing--g4          注意行人
#   information--parking--g1                   停车指示
#   complementary--maximum-speed-limit-20--g1  辅助牌上的限速（不是主牌，跳过）
#
# A+B 混合策略：
#   A) 通用类别（下面 MID_MAP 里有的）→ 映射成我们的简化名，如 speed_limit_60
#   B) 长尾类别（映射不到的）→ 保留 MTSD 原名，如 mtsd_regulatory__lane-control
#      好处：不丢信息、覆盖全部；代价：类别名带 mtsd_ 前缀与双下划线
#   辅助牌（complementary 开头的）默认跳过：它们是附在别的牌下面的小牌，
#   单独贴会变成「没有主牌的孤立小牌」，现实中不存在。

# 中层名 → 我们的子类名（按大类分组）
MID_MAP = {
    # ---- 限速（regulatory--maximum-speed-limit-<数字>）----
    "regulatory": {
        "no-entry": "prohibition_no_entry",
        "no-parking": "prohibition_no_parking",
        "no-stopping": "prohibition_no_stopping",
        "no-left-turn": "prohibition_no_left",
        "no-right-turn": "prohibition_no_right",
        "no-u-turn": "prohibition_no_u_turn",
        "no-uturn": "prohibition_no_u_turn",
        "do-not-enter": "prohibition_no_entry",
        "stop": "prohibition_stop",
        "yield": "prohibition_yield",
        "give-way": "prohibition_yield",
        "no-pedestrians": "prohibition_no_pedestrians",
        "go-straight": "mandatory_go_straight",
        "turn-left": "mandatory_turn_left",
        "turn-right": "mandatory_turn_right",
        "keep-left": "mandatory_keep_left",
        "keep-right": "mandatory_keep_right",
        "roundabout": "mandatory_roundabout",
        "dual-lanes-go-straight-on-right": "mandatory_go_straight",
        "height-limit": "prohibition_height_limit",
        "weight-limit": "prohibition_weight_limit",
        "width-limit": "prohibition_width_limit",
    },
    "warning": {
        "pedestrians-crossing": "warning_pedestrian",
        "pedestrian": "warning_pedestrian",
        "children": "warning_children",
        "bump": "warning_bump",
        "speed-bump": "warning_bump",
        "slippery": "warning_slippery",
        "roadworks": "warning_roadwork",
        "roadwork": "warning_roadwork",
        "left-curve": "warning_curve_left",
        "right-curve": "warning_curve_right",
        "double-curve": "warning_narrow",
        "narrow": "warning_narrow",
        "roundabout": "warning_roundabout",
        "crossroad": "warning_crossroad",
        "intersection": "warning_crossroad",
        "traffic-signals": "signal_red",
        "left-turn": "warning_curve_left",
        "right-turn": "warning_curve_right",
    },
    "information": {
        "parking": "guide_parking",
        "pedestrians-crossing": "crosswalk_sign",
        "motorway": "guide_highway",
        "end-of-motorway": "guide_highway",
        "bus-stop": "guide_bus_stop",
        "minimum-speed-40": "mandatory_min_speed",
    },
}

TOP_MAP = {
    "regulatory": "prohibition",
    "warning": "warning",
    "information": "guide",
    "complementary": "guide",
    "other-sign": "guide",
}

SUB_SHAPE = {
    # 形状提示：MTSD 不提供形状，按类别惯例给
    "regulatory": "circle",
    "warning": "triangle",
    "information": "rect",
}


def map_label(label, keep_longtail=True):
    """MTSD 标签 → (我们的类别名, 是否长尾保留)。

    'regulatory--maximum-speed-limit-60--g1' → ('speed_limit_60', False)
    'regulatory--lane-control--g1'           → ('mtsd_regulatory__lane-control', True)
    'complementary--xxx--g1'                 → (None, False)   辅助牌跳过
    """
    raw = str(label or "").strip()
    if not raw:
        return None, False
    parts = raw.split("--")
    if len(parts) < 2:
        return None, False
    top = parts[0].strip().lower()
    mid = parts[1].strip().lower()

    # 辅助牌单独存在没有意义，跳过
    if top == "complementary":
        return None, False

    # ⚠️ 先判 end-of：'end-of-maximum-speed-limit-30' 是解除限速牌，
    #    不是限速牌，必须在限速匹配之前拦掉（否则会被错认成 speed_limit_30）
    if mid.startswith("end-of-"):
        if keep_longtail and mid not in ("end-of-motorway",):
            return "mtsd_%s__%s" % (top, mid), True
        return None, False

    # 限速：maximum-speed-limit-60 / maximum-speed-limit-led-100
    if "maximum-speed-limit" in mid:
        digits = "".join(ch for ch in mid.split("maximum-speed-limit", 1)[1] if ch.isdigit())
        if digits:
            return "speed_limit_%s" % digits, False

    if False:  # 占位，保持下面的结构不变
        if keep_longtail and mid not in ("end-of-motorway",):
            return "mtsd_%s__%s" % (top, mid), True
        return None, False

    sub = MID_MAP.get(top, {}).get(mid)
    if sub:
        return sub, False

    # B) 长尾：保留原标签
    if keep_longtail:
        return "mtsd_%s__%s" % (top, mid), True
    return None, False


def shape_for(cls):
    """按类别名推断形状（圆/三角/矩形），用于生成 alpha 遮罩。"""
    if cls.startswith("speed_limit") or cls.startswith("prohibition") or \
            cls.startswith("mandatory") or cls.startswith("signal"):
        return "circle"
    if cls.startswith("warning"):
        return "triangle"
    if cls.startswith("mtsd_warning"):
        return "triangle"
    if cls.startswith("mtsd_regulatory"):
        return "circle"
    return "rect"

def load_annotations(mtsd_root, wanted_keys=None):
    """扫描 mtsd_fully/annotations，返回 {image_key: [objects]}。

    wanted_keys 给定时只收这些 key（省内存）。
    """
    ann_root = os.path.join(mtsd_root, "mtsd_fully", "annotations")
    if not os.path.isdir(ann_root):
        ann_root = os.path.join(mtsd_root, "mtsd_partially", "annotations")
    out = {}
    if not os.path.isdir(ann_root):
        return out
    for sub in sorted(os.listdir(ann_root)):
        d = os.path.join(ann_root, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            key = os.path.splitext(fn)[0]
            if wanted_keys is not None and key not in wanted_keys:
                continue
            try:
                with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                    out[key] = json.load(f)
            except Exception:
                continue
    return out


def read_country_manifest(cdir):
    """读某国家的 manifest.jsonl → [{file, image_key, ...}]。"""
    p = os.path.join(cdir, "manifest.jsonl")
    items = []
    if not os.path.isfile(p):
        return items
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def crop_sign(bgr, bbox, pad_ratio=0.06):
    """按标注框裁出标牌，留一点边。"""
    h, w = bgr.shape[:2]
    x1 = float(bbox.get("xmin", 0))
    y1 = float(bbox.get("ymin", 0))
    x2 = float(bbox.get("xmax", 0))
    y2 = float(bbox.get("ymax", 0))
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    px, py = bw * pad_ratio, bh * pad_ratio
    x1 = int(max(0, x1 - px))
    y1 = int(max(0, y1 - py))
    x2 = int(min(w, x2 + px))
    y2 = int(min(h, y2 + py))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return bgr[y1:y2, x1:x2]


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtsd", required=True, help="MTSD 根目录（含 mtsd_by_country）")
    ap.add_argument("--out", required=True, help="素材库输出根目录")
    ap.add_argument("--countries", default="", help="只处理这些国家码，逗号分隔；空=全部")
    ap.add_argument("--per-class", type=int, default=3, help="每国每类最多导出几张")
    ap.add_argument("--min-px", type=int, default=40, help="标牌裁切图最小边长")
    ap.add_argument("--no-longtail", action="store_true",
                    help="不导出长尾类别（默认导出，保留 MTSD 原名）")
    a = ap.parse_args()

    by_country = os.path.join(a.mtsd, "mtsd_by_country")
    if not os.path.isdir(by_country):
        print("找不到 mtsd_by_country: %s" % by_country)
        return 2
    os.makedirs(a.out, exist_ok=True)

    wanted = [c.strip().upper() for c in a.countries.split(",") if c.strip()]
    countries = []
    for name in sorted(os.listdir(by_country)):
        p = os.path.join(by_country, name)
        if not os.path.isdir(p):
            continue
        cc = name.split("_")[0].upper()
        if len(cc) != 2 or not cc.isalpha():
            continue
        if wanted and cc not in wanted:
            continue
        countries.append((cc, p))
    print("处理 %d 个国家" % len(countries))

    report = {"countries": {}, "total": 0, "skipped_no_ann": 0}
    stats = {"mapped": 0, "longtail": 0}
    for cc, cdir in countries:
        manifest = read_country_manifest(cdir)
        if not manifest:
            # 没有 manifest 就直接按文件名遍历（图片是整张？还是裁切？无法确定 → 跳过）
            report["skipped_no_ann"] += 1
            continue
        # 这里的关键：by_country 下的是**整张街景图**（width/height 与标注一致），
        # 需要按标注框裁出标牌。先只读本国的标注（省时间）。
        keys = set(m.get("image_key") for m in manifest if m.get("image_key"))
        anns = load_annotations(a.mtsd, keys)
        if not anns:
            report["skipped_no_ann"] += 1
            continue

        out_dir = os.path.join(a.out, cc)
        os.makedirs(out_dir, exist_ok=True)
        buckets = {}      # 类别 → [(质量分, 裁切图)]
        for m in manifest:
            key = m.get("image_key")
            ann = anns.get(key)
            if not ann:
                continue
            fpath = os.path.join(cdir, m.get("file", ""))
            if not os.path.isfile(fpath):
                continue
            img = cv2.imread(fpath)
            if img is None:
                continue
            for ob in ann.get("objects", []):
                props = ob.get("properties", {}) or {}
                if props.get("occluded") or props.get("out-of-frame"):
                    continue
                cls, is_longtail = map_label(ob.get("label"), keep_longtail=not a.no_longtail)
                if not cls:
                    continue
                # 长尾类别单独统计（便于判断映射是否够用）
                if is_longtail:
                    stats["longtail"] += 1
                else:
                    stats["mapped"] += 1
                crop = crop_sign(img, ob.get("bbox", {}))
                if crop is None:
                    continue
                if min(crop.shape[:2]) < a.min_px:
                    continue
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
                sharp = float(cv2.Laplacian(g, -1).var())
                bright = float(g.mean())
                score = sharp - abs(bright - 120.0) * 2.0
                buckets.setdefault(cls, []).append((score, crop))

        saved = 0
        for cls, items in buckets.items():
            items.sort(key=lambda t: -t[0])
            shape = "circle"
            try:
                import tsr as tsr_mod
                shape = tsr_mod.shape_of(cls)
            except Exception:
                pass
            for i, (score, crop) in enumerate(items[:a.per_class]):
                side = min(crop.shape[:2])
                crop = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_AREA)
                mask = np.zeros((192, 192), np.uint8)
                if shape == "circle":
                    cv2.circle(mask, (96, 96), 92, 255, -1)
                elif shape == "triangle":
                    pts = np.array([[96, 8], [10, 180], [182, 180]], np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                else:
                    mask[:, :] = 255
                bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = mask
                fn = cls if i == 0 else "%s_v%d" % (cls, i + 1)
                cv2.imwrite(os.path.join(out_dir, fn + ".png"), bgra)
                saved += 1
        if saved:
            report["countries"][cc] = {"dir": out_dir, "assets": saved,
                                       "classes": len(buckets)}
            report["total"] += saved
            print("  %-3s %-8s %3d 个类别 %4d 张"
                  % (cc, alib.COUNTRY_NAMES.get(cc, "") if alib else "", len(buckets), saved))

    with open(os.path.join(a.out, "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("-" * 55)
    print("共产出 %d 张，覆盖 %d 个国家 → %s"
          % (report["total"], len(report["countries"]), a.out))
    print("无标注跳过 %d 个国家" % report["skipped_no_ann"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
