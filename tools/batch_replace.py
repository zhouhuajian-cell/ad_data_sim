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
import cv2
import numpy as np                                                    # noqa: E402
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


def pick_by_shape(shape, sign_table, seed=None, prefer_cls=None):
    """SHAPE-AWARE: 按原标牌形状从素材库挑素材。

    shape: 'circle' | 'triangle' | 'square' | 'rect' | 'unknown'
    返回 (类别名, 路径) 或 (None, None)
    """
    if not sign_table:
        return None, None
    import hashlib
    import random as _rd

    _S2T = {"circle": "circle", "triangle": "triangle",
            "square": "square", "rect": "square", "unknown": None}
    want = _S2T.get(shape, None)

    pool = [(nm, p) for nm, p in sign_table.items()]
    if want:
        same = [(nm, p) for nm, p in pool if asset_shape(nm) == want]
        if same:
            pool = same
        else:
            return None, None      # 该形状完全没有素材

    # 优先同类（同 YOLO 大类）
    if prefer_cls:
        pref = [(nm, p) for nm, p in pool if nm == prefer_cls or nm.startswith(prefer_cls)]
        if pref:
            pool = pref

    if not pool:
        return None, None
    if len(pool) == 1:
        return pool[0]
    if seed is None:
        return _rd.choice(pool)
    h = int(hashlib.md5(str(seed).encode("utf-8")).hexdigest()[:8], 16)
    return pool[h % len(pool)]


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

    h, w = roi.shape[:2]

    # SHAPE-RED-FIRST: 先看有没有红圈/红边（禁令牌/限速牌）
    # 小 ROI 上 Otsu 会把红圈和灰背景合成一类，必须用 HSV 红掩码兜底
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _h, _s, _v = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]
    _red = (((_h <= 12) | (_h >= 160)) & (_s >= 50)) & (_v >= 30)
    if _red.sum() >= max(30, int(h * w * 0.08)):
        # 有足够的红色 → 用红色环的形状判定
        _rm = (_red.astype(np.uint8)) * 255
        _k2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, w // 8), max(3, h // 8)))
        _rm = cv2.morphologyEx(_rm, cv2.MORPH_CLOSE, _k2, iterations=1)
        _cnts, _ = cv2.findContours(_rm, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if _cnts:
            _c = max(_cnts, key=cv2.contourArea)
            _area = cv2.contourArea(_c)
            _per = cv2.arcLength(_c, True)
            if _area > 30 and _per > 10:
                _bx, _by, _bw, _bh = cv2.boundingRect(_c)
                _fill = _area / float(max(_bw * _bh, 1))
                _asp = _bw / float(max(_bh, 1))
                _circ = 4.0 * math.pi * _area / (_per * _per + 1e-6)
                # 圆环：包围框接近方形 + 圆形度高
                if 0.75 <= _asp <= 1.35 and _circ >= 0.35:
                    return "circle"
                # 三角：顶部窄底部宽
                _top = float((_rm[0:max(1, h // 5), :] > 127).sum())
                _bot = float((_rm[-max(1, h // 5):, :] > 127).sum())
                if _bot > 0 and _top / _bot < 0.45:
                    return "triangle"

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


def fit_to_inner(image, bbox, shrink=0.90):
    """把粘贴区域收缩到原标牌的「内圈」，避免盖住原有的深色外框。

    实测问题：
      原标牌是「深色外圈 + 白色内圈」的圆牌（49×49），
      新素材按 49×49 贴上去后，圆形 alpha 的四角露出原牌的深色外圈，
      视觉上就是「一圈重影」。
      把粘贴区域收缩到 88~92%，正好盖住白色内圈、不碰深色外圈，
      圆角处露出的就是原牌的深色描边 —— 看起来跟原牌融为一体。

    参数：
      shrink  收缩比例（0.90 = 收到 90%）
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    hw = (x2 - x1) * 0.5 * shrink
    hh = (y2 - y1) * 0.5 * shrink
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def measure_face(image, bbox, pad=0.25):
    """测量原标牌「牌面」的实际大小（白色/亮色内圈），返回精确的粘贴区域。

    为什么不按比例收缩：
      实测贴图溢出量在 -74% ~ +1737% 之间浮动，固定比例（0.88/0.90）完全是猜。
      原牌的真实结构是「深色描边 + 亮色牌面」，
      只要把亮色牌面量准了，新牌就能严丝合缝地盖住它，不凸出也不凹陷。

    做法：
      1) 在 bbox 外扩 pad 的范围内找亮度最高的连通域（牌面通常是亮色）
      2) 用它的外接矩形作为粘贴区域
      3) 量不准（面积过小/贴满全框）时退回 bbox 中心 90%
    """
    import cv2
    import numpy as np

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    bw0, bh0 = x2 - x1, y2 - y1
    px = int(bw0 * pad)
    ex1, ey1 = max(0, x1 - px), max(0, y1 - px)
    ex2, ey2 = min(W, x2 + px), min(H, y2 + px)
    roi = image[ey1:ey2, ex1:ex2]
    if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
        return [float(x1), float(y1), float(x2), float(y2)]

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = g.shape

    # 牌面 = 亮色区域。用 Otsu 拿阈值后取「偏亮」的一侧
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = th
    if bright[h // 2, w // 2] == 0:
        bright = cv2.bitwise_not(bright)
    # 闭运算把牌面的图案/文字填掉，只留整块牌面
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (max(3, w // 12) | 1, max(3, h // 12) | 1))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=2)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return _fallback(x1, y1, x2, y2, ex1, ey1)

    # 选包含 bbox 中心的最大连通域
    cx, cy = (x1 + x2) // 2 - ex1, (y1 + y2) // 2 - ey1
    cx = min(max(cx, 0), w - 1)
    cy = min(max(cy, 0), h - 1)
    lab_id = lab[cy, cx]
    if lab_id == 0:
        # 中心不在任何亮块里，退而求其次：面积最大的
        areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
        areas.sort(reverse=True)
        lab_id = areas[0][1]

    fx = stats[lab_id, cv2.CC_STAT_LEFT] + ex1
    fy = stats[lab_id, cv2.CC_STAT_TOP] + ey1
    fw = stats[lab_id, cv2.CC_STAT_WIDTH]
    fh = stats[lab_id, cv2.CC_STAT_HEIGHT]

    # 合理性检查
    if fw < bw0 * 0.45 or fh < bh0 * 0.45:
        return _fallback(x1, y1, x2, y2, ex1, ey1)     # 太小，肯定量错了
    if fw > bw0 * 1.5 or fh > bh0 * 1.5:
        return _fallback(x1, y1, x2, y2, ex1, ey1)     # 太大，把背景也算进来了

    return [float(fx), float(fy), float(fx + fw), float(fy + fh)]


def _fallback(x1, y1, x2, y2, ex1, ey1):
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    hw, hh = (x2 - x1) * 0.45, (y2 - y1) * 0.45
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def std_shape(image, bbox):
    """严格判断原标牌形状，只认标准 circle / square / triangle，其余 unknown。

    为什么必须严格：
      用户实测反馈，形状不匹配会直接穿帮（圆牌贴三角素材、椭圆贴圆）。
      宁可多判 unknown 跳过，也不能误判后乱贴。

    判定依据（按可靠性排序）：
      1) 宽高比必须接近 1（0.85~1.18）—— 先排除椭圆、竖长牌
      2) 填充率 fill = 轮廓面积 / 外接矩形面积
           圆    π/4 = 0.785
           方    1.0
           三角  0.5
      3) 圆形度 circularity = 4πA/P²（圆 1.0、方 0.785、三角 0.6）
      4) 三角必须同时满足：填充率高但顶角窄、底边宽 ——
         实测坑：最初只用「顶部中央是背景」，圆牌顶部也是背景，
         导致 50% 的圆牌被误判成三角。

    参数区间都是实测校准过的，改之前先跑验证。
    """
    import cv2
    import numpy as np

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = image[y1:y2, x1:x2]
    if roi.shape[0] < 24 or roi.shape[1] < 24:
        return "unknown"
    h, w = roi.shape[:2]

    asp = w / float(h)
    if not (0.85 <= asp <= 1.18):
        return "unknown"          # 椭圆 / 竖牌 / 横牌，不处理

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 10) | 1,
                                                      max(3, h // 10) | 1))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "unknown"
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    if area < 120 or per < 18:
        return "unknown"

    bx, by, bw, bh = cv2.boundingRect(c)
    if bw < 20 or bh < 20:
        return "unknown"
    fill = area / float(bw * bh)
    circ = 4.0 * np.pi * area / (per * per + 1e-6)

    # 菱形排除：四角全暗
    corners = [th[2, 2], th[2, w - 3], th[h - 3, 2], th[h - 3, w - 3]]
    if sum(1 for v in corners if v == 0) >= 3 and fill < 0.66:
        return "unknown"

    # ---- 三角：填充率 0.45~0.58 且 顶部窄底部宽 ----
    if 0.45 <= fill <= 0.58:
        # 顶部 20% 行的前景宽度 vs 底部 20% 行
        top_w = float((th[int(h * 0.15), :] > 0).sum())
        bot_w = float((th[int(h * 0.85), :] > 0).sum())
        if bot_w > 0 and top_w / bot_w < 0.45 and circ < 0.78:
            return "triangle"
        return "unknown"

    # ---- 圆：填充率贴近 π/4，圆形度高 ----
    if 0.70 <= fill <= 0.87 and circ >= 0.80:
        return "circle"

    # ---- 方：填充率高 ----
    if fill >= 0.87:
        return "square"

    return "unknown"


def fit_to_sign_contour(image, bbox, margin=0.0):
    """把检测框收缩到原标牌的实际轮廓。

    为什么不能直接用检测框：
      YOLO 的 bbox 通常留了几像素余量，实测原标牌轮廓比 bbox 小 5~15%。
      直接按 bbox 贴素材 → 新牌比原牌大一圈，
      原牌的深色描边/底板从边缘露出来，看起来就是「重影」。

    做法：在 bbox 内用 Otsu + 最大连通域找标牌本体，返回其外接矩形。
    找不到可靠轮廓时退回原 bbox（宁可不收缩，也不要乱收缩）。
    """
    import cv2
    import numpy as np

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 14 or roi.shape[1] < 14:
        return [float(v) for v in bbox]

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 中心是前景才保留，否则反相
    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)
    # 闭运算：把牌面内部的黑字/图案连起来，保住外轮廓
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 8) | 1,
                                                      max(3, h // 8) | 1))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return [float(v) for v in bbox]
    # 取最大的、且不是整幅画布的那个
    best = None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw_ = stats[i, cv2.CC_STAT_WIDTH]
        bh_ = stats[i, cv2.CC_STAT_HEIGHT]
        if bw_ >= w - 2 and bh_ >= h - 2:
            continue      # 整幅都算前景，说明分割失败
        if best is None or a > stats[best, cv2.CC_STAT_AREA]:
            best = i
    if best is None:
        return [float(v) for v in bbox]

    sx = stats[best, cv2.CC_STAT_LEFT]
    sy = stats[best, cv2.CC_STAT_TOP]
    sw = stats[best, cv2.CC_STAT_WIDTH]
    sh = stats[best, cv2.CC_STAT_HEIGHT]

    # 收缩过度保护：新框不小于原框的 55%
    if sw < w * 0.55 or sh < h * 0.55:
        return [float(v) for v in bbox]
    # 放大保护：不超过原框
    if sw > w or sh > h:
        return [float(v) for v in bbox]

    if margin:
        mx, my = int(sw * margin), int(sh * margin)
        sx, sy = max(0, sx - mx), max(0, sy - my)
        sw, sh = min(w - sx, sw + 2 * mx), min(h - sy, sh + 2 * my)

    return [float(x1 + sx), float(y1 + sy),
            float(x1 + sx + sw), float(y1 + sy + sh)]


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


def _frame_direct_pass(dets):
    """FRAME-PASS-DEF: 单帧高置信直通（不依赖跨帧轨迹）。"""
    out = []
    for d in dets:
        c = float(d.get("conf", 0))
        vr = float(d.get("_veh_ratio", 9.0))
        sr = float(d.get("_sign_ratio", 0.0))
        sp = float(d.get("_soft_prob", 0.0))
        b = d.get("bbox")
        if not b or len(b) < 4:
            continue
        side = min(b[2] - b[0], b[3] - b[1])
        if c >= 0.50 and vr <= 0.10 and (sr >= 0.30 or sp >= 0.15):
            out.append(d); continue
        if side >= 24 and c >= 0.45 and vr <= 0.15:
            out.append(d); continue
        if side < 24 and c >= 0.30 and vr <= 0.10 and sp >= 0.12:
            out.append(d); continue
        if sr >= 0.25 and vr <= 0.30:
            out.append(d); continue
    return out


_SAMPLER = None


def _get_sampler(asset_root):
    """懒加载全局 sampler。"""
    global _SAMPLER
    if _SAMPLER is not None:
        return _SAMPLER
    idx = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "asset_index.json")
    if not os.path.isfile(idx):
        print("  [ASSET-INDEX] 索引不存在: %s" % idx)
        return None
    try:
        _SAMPLER = AssetIndexSampler(idx, asset_root)
        print("  [ASSET-INDEX] 已加载索引 (%d 形状组)"
              % len(getattr(_SAMPLER, "by_shape", {})))
        return _SAMPLER
    except Exception as e:
        print("  [ASSET-INDEX] 加载失败:", str(e)[:60])
        return None


class AssetIndexSampler:
    """ASSET-INDEX: 按 形状+颜色+国家 查表采样。"""

    def __init__(self, index_json_path, asset_root):
        import json as _j
        from collections import defaultdict
        from pathlib import Path as _P
        self.asset_root = _P(asset_root)
        with open(index_json_path, "r", encoding="utf-8") as f:
            data = _j.load(f)
        self.by_country_shape_color = defaultdict(list)
        self.by_country_shape = defaultdict(list)
        self.by_shape_color = defaultdict(list)
        self.by_shape = defaultdict(list)
        for item in data["assets"]:
            p = str(self.asset_root / item["path"])
            c, sh, col = item["country"], item["shape"], item["color_type"]
            self.by_country_shape_color[(c, sh, col)].append(p)
            self.by_country_shape[(c, sh)].append(p)
            self.by_shape_color[(sh, col)].append(p)
            self.by_shape[sh].append(p)
        self.stats_used = {"exact": 0, "shape_only": 0, "miss": 0}

    def sample(self, shape, color_type=None, country=None,
               match_color=True, allow_fallback=True, seed=None):
        import random as _r
        _sh = {"square": "rect", "rect": "rect",
               "circle": "circle", "triangle": "triangle"}.get(shape, shape)
        if country:
            if match_color and color_type and color_type != "other":
                pool = self.by_country_shape_color.get(
                    (country, _sh, color_type), [])
                if pool:
                    self.stats_used["exact"] += 1
                    return (pool[seed % len(pool)] if seed is not None
                            else _r.choice(pool)), "exact"
                if not allow_fallback:
                    self.stats_used["miss"] += 1
                    return None, "miss"
            pool = self.by_country_shape.get((country, _sh), [])
            if pool:
                self.stats_used["shape_only"] += 1
                return (pool[seed % len(pool)] if seed is not None
                        else _r.choice(pool)), "country_shape"
            self.stats_used["miss"] += 1
            return None, "miss"
        if match_color and color_type and color_type != "other":
            pool = self.by_shape_color.get((_sh, color_type), [])
            if pool:
                self.stats_used["exact"] += 1
                return (pool[seed % len(pool)] if seed is not None
                        else _r.choice(pool)), "exact"
            if not allow_fallback:
                self.stats_used["miss"] += 1
                return None, "miss"
        pool = self.by_shape.get(_sh, [])
        if pool:
            self.stats_used["shape_only"] += 1
            return (pool[seed % len(pool)] if seed is not None
                    else _r.choice(pool)), "shape_only"
        self.stats_used["miss"] += 1
        return None, "miss"


_SAMPLER = None


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
    ap.add_argument("--temporal", action="store_true",
                    help="启用第3层时序过滤（连续>=3帧才放行，需>=5帧）")
    ap.add_argument("--match-color", action="store_true", default=True,
                    help="按 形状+底色 匹配素材（默认开）")
    ap.add_argument("--no-match-color", dest="match_color",
                    action="store_false",
                    help="关闭底色匹配，只按形状")
    ap.add_argument("--no-gate", action="store_true",
                    help="关闭三层质检漏斗（不推荐）")
    ap.add_argument("--no-semantic", action="store_true",
                    help="只做硬规则，跳过 Mask2Former 交叉验证（快）")
    ap.add_argument("--roi-match", action="store_true",
                    help="额外做 ROI 动态范围对齐（pipeline6 Step3 已处理后通常不需要）")
    ap.add_argument("--haze-blend", type=float, default=0.30,
                    help="雨雾融合度（晴天0.05，阴雨雾天0.15~0.25）")
    ap.add_argument("--blur-sigma", type=float, default=0.9,
                    help="镜头模糊（0.6~1.2）")
    ap.add_argument("--v6", action="store_true",
                    help="用 pipeline6 六步管线（形状校验→红圈精定位→三锚点→"
                         "退化→窄羽化→JPEG）")
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

    # GATE-WIRED: 三层漏斗过滤（专治 YOLO 误检）
    _gate = None
    _gate_seq = {}     # name -> dets（时序过滤前）
    if not getattr(a, "no_gate", False):
        try:
            from tsr_filter import SignGate
            _gate = SignGate(min_conf=max(a.conf, 0.60),
                             use_semantic=not getattr(a, "no_semantic", False))
            print("质检漏斗: 已启用 (conf>=%.2f)" % _gate.min_conf)
        except Exception as _eg:
            print("质检漏斗不可用:", str(_eg)[:60])

    # TRACKBACK-WIRED: 第3层 = Track-back 轨迹回溯 + 时序
    _temporal_keep = None
    _temporal_fill = None
    # FRAME-MODE-DEFAULT: 默认按帧处理，不依赖时序
    _use_temporal = getattr(a, "temporal", False)
    if _gate is not None and _use_temporal and len(images) >= 5:
        print("轨迹回溯: 预扫描 %d 帧..." % len(images))
        from tsr_filter import _seg_and_prob, _get_mapillary
        _seq = []          # 每帧通过的框
        _verified = []     # 每帧每个框是否通过 Mask2Former 交叉验证
        _ext = _get_mapillary()
        for _p in images:
            _im = cv2.imread(_p)
            if _im is None:
                _seq.append([])
                _verified.append([])
                continue
            _dd = detect_signs(_p, a.conf)
            _pass = _gate.filter_frame(_im, _dd)
            _seq.append(_pass)
            # NONE-FIX: _verified 必须与 _pass 等长（逐框一一对应）
            if not _pass:
                _verified.append([])
                continue
            try:
                _seg, _prob = _seg_and_prob(_im)
                _H, _W = _seg.shape[:2]
                _vf = []
                for _d in _pass:
                    try:
                        _x1, _y1, _x2, _y2 = _d["bbox"]
                    except Exception:
                        _vf.append(False)
                        continue
                    _cw = float(_x2 - _x1)
                    _ch = float(_y2 - _y1)
                    _ex = _cw * 0.5
                    _ey = _ch * 0.5
                    _ix1 = max(0, int(_x1 - _ex))
                    _iy1 = max(0, int(_y1 - _ey))
                    _ix2 = min(_W, int(_x2 + _ex))
                    _iy2 = min(_H, int(_y2 + _ey))
                    _re = _seg[_iy1:_iy2, _ix1:_ix2]
                    if _re.size == 0:
                        _vf.append(False)
                        continue
                    _sr = float(np.isin(_re, _ext._sign_ids).mean())
                    _vf.append(bool(_sr >= 0.20))
                while len(_vf) < len(_pass):
                    _vf.append(False)
                _verified.append(_vf)
            except Exception as _ee:
                print("    [verified] 失败:", str(_ee)[:60])
                _verified.append([False] * len(_pass))

        from tsr_trackback import trackback_filter
        # TRACKLEVEL-WIRED: 轨迹级确证 + 插值补全
        _keep, _fill, _tstats = trackback_filter(
            _seq, _verified, near_size=20.0, min_hits=2, max_hole=2)
        _temporal_keep = {}
        _temporal_fill = {}
        for _fi, _p in enumerate(images):
            _nm = os.path.splitext(os.path.basename(_p))[0]
            _keys = set()
            for _bi, _k in enumerate(_keep[_fi]):
                if not _k or _bi >= len(_seq[_fi]):
                    continue
                try:
                    _bb2 = _seq[_fi][_bi]["bbox"]
                    _keys.add((round(_bb2[0], 1), round(_bb2[1], 1)))
                except Exception:
                    continue
            _temporal_keep[_nm] = _keys
            # FILL-NONE-FIX: _fill 里含 None 占位符，必须过滤
            _temporal_fill[_nm] = [f for f in (_fill[_fi] or [])
                                   if f is not None]
        print("轨迹回溯: %d 条轨迹, 确认 %d 条, 放行 %d 框(含 %d 远景小目标), 插值补 %d 框"
              % (_tstats["tracks"], _tstats["confirmed_tracks"],
                 _tstats["kept"], _tstats["rescued_small"],
                 _tstats["filled_holes"]))

    for i, img_path in enumerate(images):
        name = os.path.splitext(os.path.basename(img_path))[0]
        rec = {"image": img_path, "replaced": [], "skipped": []}
        try:
            im = cv2.imread(img_path)
            dets = detect_signs(img_path, a.conf)
            # GATE-WIRED: 第1+2层过滤
            if _gate is not None and im is not None:
                _before = len(dets)
                dets = _gate.filter_frame(im, dets)
                # FRAME-MODE-DEFAULT: 单帧高置信直通（无需轨迹）
                dets = _frame_direct_pass(dets)
                # TRACKLEVEL-WIRED: 套用轨迹白名单
                if _temporal_keep is not None and name in _temporal_keep:
                    _allow = _temporal_keep[name]
                    dets = [d for d in dets
                            if (round(d["bbox"][0], 1),
                                round(d["bbox"][1], 1)) in _allow]
                # 插值补全：把补出来的框加进去（消除闪烁）
                if _temporal_fill is not None and name in _temporal_fill:
                    for _fb in _temporal_fill[name]:
                        if _fb is None or len(_fb) < 4:
                            continue
                        _w = _fb[2] - _fb[0]
                        _h = _fb[3] - _fb[1]
                        dets.append({
                            "bbox": _fb, "conf": 0.60,
                            "class": "speed_limit", "w": _w, "h": _h,
                            "interpolated": True})
                if len(dets) < _before:
                    rec["gate_dropped"] = _before - len(dets)
            usable = [d for d in dets if d["w"] >= a.min_px]
            if not usable:
                rec["skipped"].append("无可用检测")
                summary["skipped"] += 1
                summary["items"].append(rec)
                continue

            targets = usable if a.all_signs else [usable[0]]
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
                    # COVER-WIRED: 前景模板 IoU + 颜色投票判形
                    _tgt_color = None
                    try:
                        from tsr_shape_cover import (detect_shape_foreground,
                                                     detect_sign_color)
                        tgt_shape0, _shinfo = detect_shape_foreground(
                            im, d["bbox"], cls_hint=d["class"])
                        _tgt_color = detect_sign_color(im, d["bbox"])
                        if tgt_shape0 == "unknown":
                            tgt_shape0 = detect_shape(im, d["bbox"])
                    except Exception:
                        tgt_shape0 = detect_shape(im, d["bbox"])

                    # MATCH-COLOR-CALL: 优先查索引表，同形状+同底色
                    _used_idx = False
                    if getattr(a, "match_color", True):
                        _sp = _get_sampler(a.signs_dir)
                        if _sp is not None:
                            _pth, _mode = _sp.sample(
                                tgt_shape0, _tgt_color, a.country,
                                match_color=True,
                                allow_fallback=True,
                                seed=abs(hash(os.path.basename(img_path)
                                              + str(d["bbox"][0]))) % 100000)
                            if _pth:
                                # VAR-RENAME: 避免与 main 里的 _p 冲突
                                _base = os.path.basename(_pth)
                                path = _pth
                                cls_name = os.path.splitext(_base)[0]
                                _used_idx = True
                    if not _used_idx:
                        pass
                    if _used_idx:
                        path = path
                    else:
                        cls_name, path = pick_by_shape(
                            tgt_shape0, sign_table,
                            seed=os.path.basename(img_path) + str(d["bbox"][0]),
                            prefer_cls=d["class"] if a.vary_class else None)
                    if not path:
                        # 该形状无素材，退回原逻辑
                        if a.vary_class:
                            cls_name, path = pick_varied_asset(
                                d["class"], sign_table,
                                seed=os.path.basename(img_path))
                        else:
                            cls_name, path = pick_neutral_asset(d["class"], sign_table)
                    if not path:
                        rec["skipped"].append("%s 无匹配素材" % d["class"])
                        continue
                    sign_img = load_sign(path)

                sm = sign_mask(sign_img)
                # 形状约束：原标牌是三角形就不贴圆形图案（否则角会露出来）
                # SHAPE-ANNOT-FIX: 用前景模板+颜色投票（与采样同一套）
                try:
                    from tsr_shape_cover import detect_shape_foreground as _dsf
                    tgt_shape, _tsinfo = _dsf(im, d["bbox"], cls_hint=d["class"])
                    if tgt_shape == "unknown":
                        tgt_shape = detect_shape(im, d["bbox"])
                except Exception:
                    tgt_shape = detect_shape(im, d["bbox"])
                sign_shape = tsr_mod.shape_of(cls_name) if hasattr(tsr_mod, "shape_of") else "circle"
                if not shape_matches(sign_shape, tgt_shape, a.shape):
                    rec["skipped"].append("%s(图案%s) vs 场景%s 形状不符"
                                          % (cls_name, sign_shape, tgt_shape))
                    continue

                bbox = fit_bbox(d["bbox"], sm.shape, keep_ratio=True)

                # COVER-WIRED: 防露底 —— 素材 alpha 必须盖住原牌有色前景
                try:
                    from tsr_shape_cover import cover_check
                    bxs = bxs2 = [int(round(v)) for v in bbox]
                    _aw = bxs[2] - bxs[0]
                    _ah = bxs[3] - bxs[1]
                    if _aw > 4 and _ah > 4:
                        _alpha_r = cv2.resize(sm.astype(np.float32),
                                              (_aw, _ah),
                                              interpolation=cv2.INTER_AREA)
                        _okc, _leak, _cinfo = cover_check(im, bbox, _alpha_r)
                        if not _okc:
                            rec["skipped"].append(
                                "露底 %.0f%% > %.0f%%（%s 盖不住原牌）"
                                % (_leak * 100, 12, cls_name))
                            continue
                except Exception as _ec:
                    pass
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
                composed = None
                cinfo = {}
                # PIPELINE6-WIRED: 六步管线优先
                if a.v6:
                    try:
                        from pipeline6 import run_pipeline6
                        # 形状名映射
                        _sm_map = {"circle": "circle", "rect": "square",
                                   "square": "square",
                                   "triangle": "triangle",
                                   "invtriangle": "triangle"}
                        _dst = _sm_map.get(tgt_shape, "")
                        _out6, _info6 = run_pipeline6(
                            base, d["bbox"], sign_img, _dst,
                            jpeg_quality=60, pad_ratio=1.04)
                        if _info6.get("ok"):
                            composed = _out6
                            cinfo = {"ok": True, "v6": True,
                                     "bbox": _info6.get("bbox"),
                                     "steps": len(_info6.get("steps") or [])}
                        else:
                            print("    [v6] 跳过: %s" % _info6.get("reason", "?"))
                    except Exception as _e6:
                        print("    [v6] 异常，回退 v2: %s" % str(_e6)[:60])

                if composed is None:
                    composed, cinfo = paste_v2(
                        base, sign_img[:, :, :3], sm, bbox,
                        strength=a.strength, edge_glow=True)
                    if not cinfo.get("ok"):
                        def _deg(sig, alp, ref):
                            return make_realistic(sig, alp, ref,
                                                  strength=a.strength)
                        composed, cinfo = paste_with_occlusion(
                            base, sign_img[:, :, :3], sm, bbox,
                            degrade_fn=_deg,
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
                # ROI-MATCH-DEFAULT-OFF:
                # pipeline6 的 Step3 三锚点对齐已足够（白底压暗56、红圈降饱和）
                # 额外 ROI 后处理会二次冲淡，默认关闭，用 --roi-match 显式开启
                if getattr(a, "roi_match", False):
                    try:
                        from tsr_roi_match import apply_to_tsr_result
                        _bbox = annotations[-1].get("bbox") if annotations else None
                        if _bbox and composed is not None:
                            composed = apply_to_tsr_result(
                                im, composed, _bbox,
                                haze_blend=float(getattr(a, "haze_blend", 0.30) or 0.30),
                                blur_sigma=float(getattr(a, "blur_sigma", 1.1) or 1.1))
                    except Exception as _er:
                        print("    [ROI-MATCH] 失败:", str(_er)[:60])

                out_img = os.path.join(a.out, name + "_composed.jpg")
                cv2.imwrite(out_img, composed, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if not a.no_annotate and annotations:
                    h, w = composed.shape[:2]
                    ann = {"image": os.path.basename(out_img),
                           "source_image": os.path.basename(img_path),
                           "source_path": os.path.abspath(img_path),
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
            # TRACE-ON-OFF
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
