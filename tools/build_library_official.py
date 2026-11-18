#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_library_official.py — 从官方矢量图库构建高清标牌素材库

数据源：http://10.188.192.28:18080/v2/  「全球道路标志图库」
  - 52491 个标志，128 个地区，来源 Wikimedia Commons
  - 全部是 SVG 矢量文件（政府发布或公有领域），无损放大
  - 自带分类字段 cid（如 speed_limit / no_parking / stop）

为什么换这个源：
  之前用 MTSD 的 JPG 裁切（32~60px），贴到实拍里是一团模糊，
  素材本身分辨率不够，再好的算法也救不回来。
  SVG 可以渲染到任意尺寸，边缘锐利，是合成替换的正确素材形态。

产出：
  <out>/<CC>/<类别>.png   512×512 RGBA，带干净的 alpha（按形状裁剪）
  <out>/_report.json      统计

用法：
  python3 build_library_official.py --out /data/tsr_library_v2 \
      --countries BR,AR,CL,CO  [--size 512] [--limit-per-class 3]
"""
import argparse
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request

BASE = "http://10.188.192.28:18080"
DATA_URL = BASE + "/v2/data/country/{cc}.js"

# 图库的 cid → 我们的类别名（只映射能确定的；其余跳过，不猜）
CID_MAP = {
    # 限速
    "speed_limit": "speed_limit",
    "minimum_speed": "mandatory_min_speed",
    "truck_speed_limit": "speed_limit",
    # 禁止
    "no_parking": "prohibition_no_parking",
    "no_stopping": "prohibition_no_stopping",
    "no_entry": "prohibition_no_entry",
    "no_left_turn": "prohibition_no_left",
    "no_right_turn": "prohibition_no_right",
    "no_u_turn": "prohibition_no_u_turn",
    "no_overtaking": "prohibition_no_overtaking",
    "no_trucks": "prohibition_no_trucks",
    "no_pedestrians": "prohibition_no_pedestrians",
    "no_bicycles": "prohibition_no_bicycles",
    "no_motor_vehicles": "prohibition_no_motor_vehicles",
    "weight_limit": "prohibition_weight_limit",
    "width_limit": "prohibition_width_limit",
    "height_limit": "prohibition_height_limit",
    "stop": "prohibition_stop",
    "yield": "prohibition_yield",
    # 警告
    "pedestrian_crossing": "warning_pedestrian",
    "children": "warning_children",
    "speed_bump": "warning_bump",
    "bump": "warning_bump",
    "slippery": "warning_slippery",
    "roadworks": "warning_roadwork",
    "road_work": "warning_roadwork",
    "left_curve": "warning_curve_left",
    "right_curve": "warning_curve_right",
    "winding_road": "warning_narrow",
    "narrow_road": "warning_narrow",
    "roundabout": "warning_roundabout",
    "junction": "warning_crossroad",
    "crossroad": "warning_crossroad",
    "railway_crossing": "warning_railway",
    "steep_descent": "warning_steep",
    "steep_ascent": "warning_steep",
    # 指示
    "straight_ahead": "mandatory_go_straight",
    "turn_left": "mandatory_turn_left",
    "turn_right": "mandatory_turn_right",
    "keep_left": "mandatory_keep_left",
    "keep_right": "mandatory_keep_right",
    # 信号
    "traffic_signal": "signal_red",
    "traffic_light": "signal_red",
}

# 类别 → 形状（决定 alpha 遮罩）
SHAPE_MAP = {
    "speed_limit": "circle",
    "prohibition": "circle",
    "warning": "triangle",
    "mandatory": "circle",
    "signal": "circle",
    "guide": "rect",
    "crosswalk": "rect",
}


def fetch_country(cc):
    """取某国的标志清单（页面里是 window.RS_COUNTRIES["XX"] = [...]）。"""
    url = DATA_URL.format(cc=cc)
    req = urllib.request.Request(url, headers={"User-Agent": "tsr-lib-builder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    start = text.find("= [")
    end = text.rfind("];")
    if start < 0 or end < 0:
        return []
    try:
        return json.loads(text[start + 2:end + 1])
    except Exception:
        return []


def fetch_svg(file_field):
    """下载 SVG。file_field 是相对路径（../official-road-signs/assets/BR/xxx.svg）。"""
    name = file_field.split("/")[-1]
    cc = file_field.split("/")[-2] if "/" in file_field else ""
    url = "%s/official-road-signs/assets/%s/%s" % (BASE, cc, urllib.parse.quote(name))
    req = urllib.request.Request(url, headers={"User-Agent": "tsr-lib-builder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def shape_of(cls):
    for k, v in SHAPE_MAP.items():
        if cls.startswith(k):
            return v
    return "rect"


def make_alpha(size, shape):
    """按形状生成 alpha：圆形留圆内、三角留三角内、矩形全留。"""
    import cv2
    import numpy as np
    a = np.zeros((size, size), np.uint8)
    if shape == "circle":
        cv2.circle(a, (size // 2, size // 2), int(size * 0.485), 255, -1)
    elif shape == "triangle":
        # 警告牌：等边三角形，顶点贴顶、底边贴底
        pts = np.array([[size // 2, int(size * 0.02)],
                        [int(size * 0.03), int(size * 0.95)],
                        [int(size * 0.97), int(size * 0.95)]], np.int32)
        cv2.fillPoly(a, [pts], 255)
    else:
        a[:, :] = 255
    return a


def render_svg(svg_bytes, size):
    """SVG → PNG（RGBA）。带透明背景。"""
    import cairosvg
    png = cairosvg.svg2png(bytestring=svg_bytes,
                           output_width=size, output_height=size,
                           background_color=None)
    return png


def main():
    import cv2
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--countries", default="BR,AR,CL,CO,PE,EC,BO,PY,UY,VE,MX,RU,DE,FR")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit-per-class", type=int, default=3,
                    help="每个类别最多保存几个变体")
    ap.add_argument("--no-alpha", action="store_true", help="不按形状裁 alpha")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    report = {"countries": {}, "total": 0, "skipped_unmapped": 0, "failed": 0}

    ccs = [c.strip().upper() for c in a.countries.split(",") if c.strip()]
    for cc in ccs:
        try:
            items = fetch_country(cc)
        except Exception as e:
            print("  %-3s 取数据失败: %s" % (cc, str(e)[:60]), file=sys.stderr)
            continue
        if not items:
            print("  %-3s 无数据" % cc, file=sys.stderr)
            continue

        out_dir = os.path.join(a.out, cc)
        os.makedirs(out_dir, exist_ok=True)
        buckets = {}
        for it in items:
            cid = (it.get("cid") or "").strip().lower()
            base_cls = CID_MAP.get(cid)
            if not base_cls:
                report["skipped_unmapped"] += 1
                continue
            # 限速牌把数字带进类别名：speed_limit_60
            cls = base_cls
            if base_cls == "speed_limit":
                # ⚠️ 文件名形如 "Brasil R-19 (60).svg"：
                #    R-19 是巴西的限速标志编号，真正的限速值在括号里。
                #    之前用 r"-(\d{2,3})" 会先匹配到 R-19 的 19，产出错误的 speed_limit_19。
                name = it.get("t") or ""
                zh = it.get("zh") or ""
                num = None
                # 1) 括号里的数字最可靠："(60)"
                mm = re.search(r"\((\d{1,3})\)", name)
                if mm:
                    num = mm.group(1)
                # 2) 文件名里的 km/h 写法
                if not num:
                    mm = re.search(r"(\d{1,3})\s*(?:km|公里|km/h)", name + " " + zh, re.I)
                    if mm:
                        num = mm.group(1)
                # 3) 中文描述里的「限速N」
                if not num:
                    mm = re.search(r"限速\s*(\d{1,3})", zh)
                    if mm:
                        num = mm.group(1)
                cls = "speed_limit_%s" % num if num else base_cls
            buckets.setdefault(cls, []).append(it)

        saved = 0
        for cls, group in sorted(buckets.items()):
            shape = shape_of(cls)
            alpha = None if a.no_alpha else make_alpha(a.size, shape)
            for i, it in enumerate(group[:a.limit_per_class]):
                try:
                    svg = fetch_svg(it.get("f") or "")
                    png = render_svg(svg, a.size)
                    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
                    if arr is None:
                        report["failed"] += 1
                        continue
                    if arr.shape[2] == 4:
                        # SVG 自带透明：与形状遮罩取交集
                        if alpha is not None:
                            arr[:, :, 3] = np.minimum(arr[:, :, 3], alpha)
                    else:
                        bgr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
                        bgr[:, :, 3] = alpha if alpha is not None else 255
                        arr = bgr
                    fn = cls if i == 0 else "%s_v%d" % (cls, i + 1)
                    cv2.imwrite(os.path.join(out_dir, fn + ".png"), arr)
                    saved += 1
                except Exception as e:
                    report["failed"] += 1
        if saved:
            report["countries"][cc] = {"assets": saved, "classes": len(buckets)}
            report["total"] += saved
            print("  %-3s %3d 类 %4d 张" % (cc, len(buckets), saved), flush=True)

    with open(os.path.join(a.out, "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("-" * 50)
    print("共产出 %d 张，覆盖 %d 个国家 → %s"
          % (report["total"], len(report["countries"]), a.out))
    print("未映射跳过 %d 个，失败 %d 个"
          % (report["skipped_unmapped"], report["failed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
