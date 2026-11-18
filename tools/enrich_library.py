#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""enrich_library.py — 把官方图库里未分类（cid=none）的标志，
按各国编号体系补进素材库

背景：
  build_library_official.py 只收 cid 能映射到我们类别的标志，
  巴西 550 个标志里有 97 个 cid=none（主要是 1968/1941 历史版 + 新版未分类），
  导致素材库巴西只有 87 张。

本脚本处理两类补充：
  1) 编号体系归类：巴西 R-xx = 禁令、A-xx = 警告、D-xx = 指示
  2) 中文名关键字归类：从 zh 字段里认「限速」「禁止」「注意」等

只收「现行」标志，排除历史版（文件名含 1968 / 1941 / 1970 等）。

用法：
  python3 enrich_library.py --library /data/tsr_library_v2 --countries BR,AR,CL
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

BASE = "http://10.188.192.28:18080"
DATA_URL = BASE + "/v2/data/country/{cc}.js"

# 历史版标志的年份前缀，一律不收
HISTORIC_RE = re.compile(r"\b(19[0-9]{2}|20[0-1][0-9])\s*(BR|Brazil|Brazilian)?\s*(Road\s*sign|road\s*sign)", re.I)

# 中文名 → 类别（按优先级匹配，越靠前越优先）
ZH_RULES = [
    ("禁止停车", "prohibition_no_parking"),
    ("禁止长时停车", "prohibition_no_parking"),
    ("禁止驶入", "prohibition_no_entry"),
    ("禁止左转", "prohibition_no_left"),
    ("禁止右转", "prohibition_no_right"),
    ("禁止掉头", "prohibition_no_u_turn"),
    ("禁止超车", "prohibition_no_overtaking"),
    ("禁止行人", "prohibition_no_pedestrians"),
    ("禁止自行车", "prohibition_no_bicycles"),
    ("禁止货车", "prohibition_no_trucks"),
    ("禁止机动车", "prohibition_no_motor_vehicles"),
    ("禁止摩托车", "prohibition_no_motorcycles"),
    ("限高", "prohibition_height_limit"),
    ("限重", "prohibition_weight_limit"),
    ("限宽", "prohibition_width_limit"),
    ("限速", "speed_limit"),
    ("让行", "prohibition_yield"),
    ("停车让行", "prohibition_stop"),
    ("停止", "prohibition_stop"),
    ("注意行人", "warning_pedestrian"),
    ("人行横道", "warning_pedestrian"),
    ("注意儿童", "warning_children"),
    ("学校", "warning_children"),
    ("路面不平", "warning_bump"),
    ("减速", "warning_bump"),
    ("易滑", "warning_slippery"),
    ("施工", "warning_roadwork"),
    ("向左急弯", "warning_curve_left"),
    ("向右急弯", "warning_curve_right"),
    ("窄路", "warning_narrow"),
    ("环岛", "warning_roundabout"),
    ("交叉", "warning_crossroad"),
    ("铁路", "warning_railway"),
    ("陡坡", "warning_steep"),
    ("直行", "mandatory_go_straight"),
    ("左转", "mandatory_turn_left"),
    ("右转", "mandatory_turn_right"),
    ("靠左", "mandatory_keep_left"),
    ("靠右", "mandatory_keep_right"),
    ("最低限速", "mandatory_min_speed"),
    ("红灯", "signal_red"),
    ("信号灯", "signal_red"),
]

# 巴西编号体系：R=禁令（regulamentação）、A=警告（advertência）、D=指示（direção）
BR_CODE_RULES = [
    (r"\bR-1\b", "prohibition_no_entry"),
    (r"\bR-6[ab]?\b", "prohibition_stop"),
    (r"\bR-7\b", "prohibition_yield"),
    (r"\bR-24[ab]*\b", "prohibition_no_left"),
    (r"\bR-25[ab]*\b", "prohibition_no_right"),
    (r"\bR-26\b", "prohibition_no_u_turn"),
    (r"\bR-40\b", "prohibition_no_trucks"),
    (r"\bA-([0-9]+)", "warning_generic"),
    (r"\bD-([0-9]+)", "mandatory_generic"),
]

SHAPE_OF_PREFIX = {
    "speed_limit": "circle",
    "prohibition": "circle",
    "warning": "triangle",
    "mandatory": "circle",
    "signal": "circle",
}


def fetch_country(cc):
    req = urllib.request.Request(DATA_URL.format(cc=cc),
                                 headers={"User-Agent": "tsr-lib-builder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    s = text.find("= [")
    e = text.rfind("];")
    if s < 0 or e < 0:
        return []
    return json.loads(text[s + 2:e + 1])


def fetch_svg(file_field):
    name = file_field.split("/")[-1]
    cc = file_field.split("/")[-2] if "/" in file_field else ""
    url = "%s/official-road-signs/assets/%s/%s" % (BASE, cc, urllib.parse.quote(name))
    req = urllib.request.Request(url, headers={"User-Agent": "tsr-lib-builder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def classify_by_zh(item):
    """按中文名归类。"""
    zh = (item.get("zh") or "") + " " + (item.get("zd") or "")
    for kw, cls in ZH_RULES:
        if kw in zh:
            # 限速要把数字带上
            if cls == "speed_limit":
                m = re.search(r"限速\s*(\d{1,3})", zh)
                return ("speed_limit_%s" % m.group(1)) if m else "speed_limit"
            return cls
    return None


def classify_by_code(item, cc):
    """按编号体系归类（目前只做巴西）。"""
    if cc != "BR":
        return None
    name = item.get("t") or ""
    for pat, cls in BR_CODE_RULES:
        m = re.search(pat, name, re.I)
        if m:
            if cls.endswith("_generic"):
                return None       # 没有具体语义，不收
            return cls
    return None


def make_alpha(size, shape):
    import cv2
    import numpy as np
    a = np.zeros((size, size), np.uint8)
    if shape == "circle":
        cv2.circle(a, (size // 2, size // 2), int(size * 0.485), 255, -1)
    elif shape == "triangle":
        pts = np.array([[size // 2, int(size * 0.02)],
                        [int(size * 0.03), int(size * 0.95)],
                        [int(size * 0.97), int(size * 0.95)]], np.int32)
        cv2.fillPoly(a, [pts], 255)
    else:
        a[:, :] = 255
    return a


def main():
    import cv2
    import cairosvg
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--countries", default="BR,AR,CL,CO,PE,EC,PY,UY,VE,BO")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit-per-class", type=int, default=4)
    ap.add_argument("--include-historic", action="store_true")
    a = ap.parse_args()

    total_added = 0
    for cc in [c.strip().upper() for c in a.countries.split(",") if c.strip()]:
        try:
            items = fetch_country(cc)
        except Exception as e:
            print("  %-3s 拉取失败: %s" % (cc, str(e)[:50]), file=sys.stderr)
            continue
        if not items:
            continue

        out_dir = os.path.join(a.library, cc)
        os.makedirs(out_dir, exist_ok=True)
        existing = set(os.listdir(out_dir))

        # 找出尚未入库、且能归类的
        buckets = {}
        skipped_hist = 0
        for it in items:
            name = it.get("t") or ""
            if not a.include_historic and HISTORIC_RE.search(name):
                skipped_hist += 1
                continue
            cls = classify_by_zh(it)
            if not cls:
                cls = classify_by_code(it, cc)
            if not cls:
                continue
            buckets.setdefault(cls, []).append(it)

        added = 0
        for cls, group in buckets.items():
            shape = next((v for k, v in SHAPE_OF_PREFIX.items() if cls.startswith(k)), "circle")
            alpha = make_alpha(a.size, shape)
            # 统计该类别已有几张，避免重复
            have = sum(1 for f in existing if f.startswith(cls) and f.endswith(".png"))
            need = max(0, a.limit_per_class - have)
            if need == 0:
                continue
            for i, it in enumerate(group[:need]):
                try:
                    svg = fetch_svg(it.get("f") or "")
                    png = cairosvg.svg2png(bytestring=svg, output_width=a.size,
                                           output_height=a.size, background_color=None)
                    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
                    if arr is None:
                        continue
                    if arr.shape[2] == 4:
                        arr[:, :, 3] = np.minimum(arr[:, :, 3], alpha)
                    else:
                        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
                        arr[:, :, 3] = alpha
                    fn = "%s_x%d" % (cls, have + i + 1)
                    cv2.imwrite(os.path.join(out_dir, fn + ".png"), arr)
                    added += 1
                except Exception:
                    continue

        if added:
            total_added += added
            final = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
            print("  %-3s 新增 %3d 张  现有 %3d 张（跳过历史版 %d）" % (cc, added, final, skipped_hist),
                  flush=True)

    print("-" * 50)
    print("合计新增 %d 张" % total_added)
    return 0


if __name__ == "__main__":
    sys.exit(main())
