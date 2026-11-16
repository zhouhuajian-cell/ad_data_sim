# -*- coding: utf-8 -*-
# truth_schema.py — 统一真值字段定义 + 旧→新字段归一 + 难例判定（方案模块 5 的唯一口径）
# 干什么：
#   ① FIELD_SPEC / FRAME_SPEC：把「方案模块 5」要求的统一真值字段写死成一张表（类型、范围、是否必需），
#      下游（VOC 导出、难例 CSV、校验脚本）都从这张表取字段名，不许各自另编一套名字；
#   ② normalize_object()：把引擎现在写的旧字段（class / bbox_2d / occlusion）补齐成新字段
#      （class_name / bbox / occlusion_ratio / visible_ratio / distance / light_level / difficult_level /
#       weather / wear_level），缺什么给什么安全默认值；
#   ③ normalize_frame()：整帧归一（light_level / weather 从帧级 lighting 取，对象级字段可以覆盖它）；
#   ④ judge_difficult(ob, rules) / judge_frame()：难例判定 —— 阈值只在 RULES 里写一次，
#      post/hardcase_export.py 直接 import 这里的 RULES 与 judge_difficult，不重复实现第二套阈值；
#   ⑤ validate_frame()：批量校验真值文件，返回人类可读的错误列表（空列表 = 合法）。
# 怎么跑（本文件可单独当 CLI 用，批量体检整批真值 JSON）：
#   python common/truth_schema.py --meta /data/render/meta            # 校验旧/新真值（宽容口径）
#   python common/truth_schema.py --meta /data/render/meta --strict   # 严格口径：统一字段必须逐键存在
#   python common/truth_schema.py --meta /data/render/meta --normalized /tmp/meta_v2
#                                                                      # 另存归一后的新字段真值（不改原文件）
#   退出码：0 = 全部合法；1 = 有错误；2 = 路径/依赖等使用错误。
# 成功判据：--normalized 下每帧都有 <frame_id>.json，其 objects[] 同时含新旧两套字段且通过严格口径校验；
#           对正常帧 validate_frame 返回 []，对缺 width/bbox 的帧返回非空列表。
# 兼容：Python 3.8（不用 3.9+ 的 dict|dict、不用 3.10 的 X|Y 类型注解）；只依赖标准库。
#
# ⚠️ 方案原文（《UE5 全维度感知仿真技术方案》模块 5）与本仓库现状之间有几处**必须由人拍板**的口径问题，
#   代码按最保守的方式落地，并在对应常量上标「⚠️口径」。Lead 若改了方案口径，只需改这里的常量：
#   (a) light_level 的取值：方案规则写的是 {backlight, low_light}，但现有 config/lights.json 的 condition 是
#       {front_light, back_light, dusk, night, rain}（逆光是 back_light 不是 backlight；夜间是 night 不是 low_light）。
#       这里 **接受两套拼写**，输出统一写方案拼写（backlight / low_light），对照表见 LIGHT_LEVEL_MAP；
#       归属哪个是「标签口径」，不新增任何光照条件。
#   (b) 它要求目标类别，但现有类别表（batch_capture.SEMANTIC_STENCIL / queue_gen.CLASSES /
#       pack_dataset.CLASSES 同源）里**没有 Landline / 车道线类**。WEAR_CLASSES 里预先写了常见别名
#       （lane_line / lane_marking ...），引擎一旦新增该类即生效；当前数据里 wear_level 恒为 0，磨损规则命不中。
#   (c) 缺省值：现有帧级 lighting 与旧 manifest 都没有 wear / light_level 字段，默认 wear_level=0、
#       distance=0，这是「安全默认」而不是真实测量值 —— 引擎侧补字段前，磨损类难例永远判不出来。
#   (d) difficult 的 VOC 映射：VOC 的 <difficult> 只有 0/1，而 difficult_level 是 0/1/2/3，
#       映射为「difficult_level > 0 → difficult=1」。要还原原值请看扩展节点 <difficult_level>。
import argparse
import json
import os
import sys

# ══════════════════════════════════════════════════════════════════════════════════
# 一、枚举与阈值（全仓库唯一一份；hardcase_export / voc_export 一律 import 这里）
# ══════════════════════════════════════════════════════════════════════════════════

# 目标类别表：与 batch_capture.SEMANTIC_STENCIL、queue_gen.CLASSES、post/pack_dataset.CLASSES 同源，
# 改一处必须同步另外三处（现有四处都在文件头注明了这一点）。
# 方案《全维度感知仿真》新增「车道线 / 路面标识」类（模块 2），因此这里加 landline；
# traffic_sign 覆盖 TSR 交通标识，vehicle/pedestrian/cone 覆盖交通目标物。
CLASS_NAMES = ["road", "vehicle", "pedestrian", "traffic_sign", "cone", "vegetation", "landline"]
# class_id 沿用 COCO 的 1 起始编号（pack_dataset.to_coco 用 id = 序号+1），YOLO 侧用 序号 从 0 起。
CLASS_ID = dict((c, i + 1) for i, c in enumerate(CLASS_NAMES))

# 光照等级：canonical 是方案拼写（light_level 字段的输出值）
LIGHT_LEVELS = ["front_light", "backlight", "dusk", "night", "low_light", "rain"]
# 判定为「光照难例」的取值（方案规则：light_level ∈ {backlight, low_light}）
HARD_LIGHT_LEVELS = ("backlight", "low_light")
# config/lights.json 的 lighting.condition（以及 batch_capture.lighting_with_condition 能产出的值）
# → 统一 light_level。前五个是恒等映射，只有 back_light→backlight、night→low_light 是改名。
# ⚠️口径(a)：把 back_light 并入 backlight、把 night 并入 low_light。若方案认为夜间只是低照度、
#   不算强烈光照异常，把下面的 "night": "low_light" 改成 "night": "night" 即可（night 不在 HARD_LIGHT_LEVELS 里，
#   夜间就不再自动进光照难例）。
LIGHT_LEVEL_MAP = {
    "front_light": "front_light",
    "back_light": "backlight",
    "backlight": "backlight",
    "back_lighting": "backlight",
    "逆光": "backlight",
    "dusk": "dusk",
    "night": "low_light",
    "low_light": "low_light",
    "night_low_light": "low_light",
    "夜间": "low_light",
    "rain": "rain",
}
LIGHT_LEVEL_DEFAULT = "front_light"

# 天气：沿用 config/lights.json 的 weather 取值（clear / rain ...），原样透传，这里只给缺省值。
WEATHER_DEFAULT = "clear"

# 磨损判定只对 TSR（交通标志）/ 车道线一类目标有意义（方案规则：仅 TSR/车道线类）。
# ⚠️口径(b)：现有 CLASS_NAMES 里没有车道线类，别名先留着，类别扩展后自动生效。
WEAR_CLASSES = ("traffic_sign", "landline", "land_line", "lane_line", "lane_marking", "lane")

# 难例规则表：规则名 → 阈值与说明。阈值只在这里出现一次。
# 命中后 difficult_level = min(3, 命中规则条数)，理由写进 rules 列（分号连接）。
RULES = {
    "occlusion": {
        "threshold": 0.4,       # occlusion_ratio > 0.4
        "op": "gt",
        "field": "occlusion_ratio",
        "classes": None,        # None = 所有类别都适用
        "desc": "遮挡难例：occlusion_ratio > 0.4",
    },
    "far_small": {
        "threshold": 80.0,      # distance > 80 米
        "op": "gt",
        "field": "distance",
        "classes": None,
        "desc": "远距离小目标：distance > 80 m",
    },
    "light": {
        "threshold": None,      # 集合判定，不是数值阈值
        "op": "in",
        "field": "light_level",
        "values": HARD_LIGHT_LEVELS,
        "classes": None,
        "desc": "光照难例：light_level ∈ {%s}" % ", ".join(HARD_LIGHT_LEVELS),
    },
    "wear": {
        "threshold": 0.5,       # wear_level > 0.5，且仅 TSR / 车道线类
        "op": "gt",
        "field": "wear_level",
        "classes": WEAR_CLASSES,
        "desc": "磨损难例：wear_level > 0.5（仅 %s）" % "/".join(WEAR_CLASSES[:1] + WEAR_CLASSES[3:4]),
    },
    # ---- 以下两条为「方案之外补齐」：原规则只看距离，会漏掉"很近但很小"的目标（如远处的锥桶其实不大，
    #      但近处被树挡掉一半的标识也很小），以及贴边截断的目标；两者都是训练集里真实存在的难例。----
    "too_small": {
        "threshold": 1024.0,    # 1920×1080 画幅下框面积 < 32×32；其它画幅按面积比例缩放（见 relative_size）
        "op": "lt",
        "field": "bbox_area",
        "relative_size": True,  # True = 阈值按「画幅面积 / 1920×1080」缩放，避免小分辨率数据全被判成极小目标
        "classes": None,
        "desc": "极小目标：框面积 < 画幅面积的 0.049%（约 32×32 @1080p）",
    },
    "truncated": {
        "threshold": None,
        "op": "is_true",
        "field": "truncated",
        "classes": None,
        "desc": "目标被画面边缘截断（框不完整，训练时容易漏检）",
    },
}
RULE_NAMES = list(RULES.keys())

# 宽容口径下的「信息」区：validate_frame(..., info=INFO) 时把「缺字段已用默认值补齐」记进这里，
# 不参与判定（防空列表 = 有错误）；CLI 会把它们打成 INFO 行，方便看出旧真值靠哪些默认值跑通。
INFO = []

# ══════════════════════════════════════════════════════════════════════════════════
# 二、字段表
#    spec 项：kind 类型 / req 是否必需（校验期） / lo,hi 数值范围 / enum 允许值 /
#             default normalize 时缺失用的安全默认值 / desc 人话说明
# ══════════════════════════════════════════════════════════════════════════════════

FIELD_SPEC = {
    # —— 模块 5 明确点名的统一真值字段（对象级）——
    "class_name": {"kind": "str", "req": True, "enum": CLASS_NAMES, "default": "unknown",
                   "desc": "类别名（旧字段 class）"},
    "class_id": {"kind": "int", "req": True, "lo": 1, "hi": len(CLASS_NAMES), "default": 0,
                 "desc": "类别编号，沿用 COCO 1 起始（旧数据没有，按 CLASS_ID 算）"},
    "bbox": {"kind": "list4", "req": True, "default": None,
             "desc": "2D 框 [xmin,ymin,xmax,ymax]，像素、左上原点、右开（旧字段 bbox_2d）"},
    "occlusion_ratio": {"kind": "float", "req": True, "lo": 0.0, "hi": 1.0, "default": 0.0,
                        "desc": "遮挡比例 0~1（旧字段 occlusion）"},
    "visible_ratio": {"kind": "float", "req": True, "lo": 0.0, "hi": 1.0, "default": None,
                      "desc": "可见比例 0~1；缺省 = 1 - occlusion_ratio"},
    "distance": {"kind": "float", "req": True, "lo": 0.0, "hi": None, "default": 0.0,
                 "desc": "目标到相机的距离（米）"},
    "light_level": {"kind": "str", "req": True, "enum": LIGHT_LEVELS, "default": LIGHT_LEVEL_DEFAULT,
                    "desc": "光照等级；缺省取帧级 lighting.condition 映射后的值"},
    "difficult_level": {"kind": "int", "req": True, "lo": 0, "hi": 3, "default": None,
                        "desc": "难度 0~3 = 命中的难例规则条数；缺省由 judge_difficult 算"},
    "weather": {"kind": "str", "req": True, "enum": None, "default": WEATHER_DEFAULT,
                "desc": "天气（clear / rain ...）；缺省取帧级 lighting.weather"},
    "wear_level": {"kind": "float", "req": False, "lo": 0.0, "hi": 1.0, "default": 0.0,
                   "desc": "磨损程度 0~1（仅 TSR/车道线有意义；现状恒 0，见文件头口径(c)）"},
    # —— 以下为方案之外的既有字段，保留并在归一结果里透传（训练/抽检还要用）——
    "instance_id": {"kind": "int", "req": False, "lo": 0, "hi": None, "default": 0,
                    "desc": "实例号（引擎内生成目标时分配）"},
    "bbox_3d": {"kind": "list7", "req": False, "default": None,
                "desc": "3D 框 [cx,cy,cz,sx,sy,sz,yaw]，米/度（旧字段，原样保留）"},
    "truncated": {"kind": "bool", "req": False, "default": False,
                  "desc": "是否被画面边界截断（VOC 的 <truncated>）"},
    "occlusion": {"kind": "float", "req": False, "lo": 0.0, "hi": 1.0, "default": 0.0,
                  "desc": "旧字段名，归一后与 occlusion_ratio 同值（兼容只认旧名的下游）"},
    "bbox_2d": {"kind": "list4", "req": False, "default": None,
                "desc": "旧字段名，归一后与 bbox 同值（兼容只认旧名的下游）"},
    "class": {"kind": "str", "req": False, "enum": None, "default": None,
              "desc": "旧字段名，归一后与 class_name 同值"},
}

# 帧级字段：validate_frame 校验的帧结构（对象字段由 FIELD_SPEC 管）
FRAME_SPEC = {
    "frame_id": {"kind": "str", "req": True, "desc": "帧名，与 <frame_id>.json / <frame_id>.png 同主文件名"},
    "width": {"kind": "int", "req": True, "lo": 1, "hi": None, "desc": "画面宽（像素）"},
    "height": {"kind": "int", "req": True, "lo": 1, "hi": None, "desc": "画面高（像素）"},
    "objects": {"kind": "objects", "req": True, "desc": "目标列表（可以为空列表，不能缺这个键）"},
    "lighting": {"kind": "dict", "req": False, "desc": "光照段（condition/weather 是 light_level/weather 的来源）"},
    "speed": {"kind": "float", "req": False, "lo": 0.0, "hi": None, "desc": "车速 m/s（后处理算模糊核长）"},
}

# 旧字段 → 新字段（normalize_object 的迁移表）
LEGACY_ALIASES = {
    "class": "class_name",
    "bbox_2d": "bbox",
    "occlusion": "occlusion_ratio",
}


# ══════════════════════════════════════════════════════════════════════════════════
# 三、小工具
# ══════════════════════════════════════════════════════════════════════════════════

def _as_float(value, default=0.0):
    """宽松取浮点；坏值（None/字符串/NaN）→ default。真值文件是外部数据，不能假设类型。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:                    # NaN
        return default
    return v


def _as_int(value, default=0):
    """宽松取整数（四舍五入）。"""
    return int(round(_as_float(value, float(default))))


def _clamp(v, lo, hi):
    """裁剪到 [lo, hi]；hi=None 表示上不封顶。"""
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def canonical_light_level(value, default=LIGHT_LEVEL_DEFAULT):
    """任意光照写法 → 统一 light_level。

    认得的写法见 LIGHT_LEVEL_MAP（back_light/backlight、night/low_light 两套拼写都吃）；
    认不出的值原样小写下划线化后返回（不丢信息，但会被 validate_frame 标成非枚举值）。
    """
    if value is None:
        return default
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return default
    return LIGHT_LEVEL_MAP.get(key, key)


def light_level_of_frame(frame):
    """帧级 → light_level：优先 lighting.condition，其次 lighting.light_level，兜底 front_light。"""
    lig = frame.get("lighting") if isinstance(frame, dict) else None
    if isinstance(lig, dict):
        for key in ("condition", "light_level", "label"):
            if lig.get(key):
                return canonical_light_level(lig.get(key))
    if isinstance(frame, dict) and frame.get("light_level"):
        return canonical_light_level(frame.get("light_level"))
    return LIGHT_LEVEL_DEFAULT


def weather_of_frame(frame):
    """帧级 → weather：优先 lighting.weather，其次帧级 weather，兜底 clear。"""
    lig = frame.get("lighting") if isinstance(frame, dict) else None
    if isinstance(lig, dict) and lig.get("weather"):
        return str(lig.get("weather")).strip().lower()
    if isinstance(frame, dict) and frame.get("weather"):
        return str(frame.get("weather")).strip().lower()
    return WEATHER_DEFAULT


def class_id_of(class_name):
    """类别名 → class_id（0 表示不在类别表里）。"""
    return CLASS_ID.get(str(class_name or ""), 0)


def is_wear_class(class_name):
    """该类是否参与磨损判定（仅 TSR / 车道线）。"""
    return str(class_name or "").lower() in WEAR_CLASSES


# ══════════════════════════════════════════════════════════════════════════════════
# 四、难例判定（阈值只读 RULES）
# ══════════════════════════════════════════════════════════════════════════════════

def rule_triggered(rule, ob):
    """单条规则是否被这个目标命中。ob 需要已归一（至少含 rule["field"] 这个键）。"""
    field = rule.get("field")
    if rule.get("op") == "in":
        return ob.get(field) in (rule.get("values") or ())
    if rule.get("op") == "is_true":
        return bool(ob.get(field))
    if rule.get("classes") and not is_wear_class(ob.get("class_name")):
        # classes 限定了适用范围（目前只有 wear 用），其它类别一律不算
        if field == "wear_level":
            return False
    threshold = rule.get("threshold")
    if field == "bbox_area":
        # 框面积由 bbox 现算，允许真值里没有 bbox_area 字段
        v = _as_float(ob.get("bbox_area"), bbox_area_of(ob.get("bbox") or ob.get("bbox_2d")))
        if rule.get("relative_size") and threshold is not None:
            # 按画幅面积缩放阈值：1080p 下 1024px²；320×180 这类小画幅按比例缩小
            w = _as_float(ob.get("frame_width"), 1920.0)
            h = _as_float(ob.get("frame_height"), 1080.0)
            if w > 0 and h > 0:
                threshold = float(threshold) * (w * h) / (1920.0 * 1080.0)
    else:
        v = _as_float(ob.get(field), 0.0)
    if rule.get("op") == "gt":
        return v > float(threshold)
    if rule.get("op") == "ge":
        return v >= float(threshold)
    if rule.get("op") == "lt":
        return v < float(threshold)
    return False


def bbox_area_of(bbox):
    """[xmin,ymin,xmax,ymax] → 像素面积；非法框返回 0。"""
    try:
        x1, y1, x2, y2 = [float(v) for v in (bbox or [])[:4]]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def judge_difficult(ob, rules=None, frame_width=None, frame_height=None):
    """难例判定：返回 (difficult_level, [命中的规则名...])，规则名顺序同 RULES。

    ob 传归一后的对象（含 class_name / occlusion_ratio / distance / light_level / wear_level）。
    frame_width/height：画幅尺寸，供 too_small 这类「按画幅比例」的规则缩放阈值；不给则按 1920×1080。
    difficult_level = min(3, 命中规则条数)：方案只给了「命中即难例」，没给分级，
    这里把「几条规则同时命中」当作难度分级，0 好、1 一般、2 难、3 极难，VOC 只用到 >0。
    """
    rules = RULES if rules is None else rules
    probe = dict(ob or {})
    if frame_width:
        probe.setdefault("frame_width", frame_width)
    if frame_height:
        probe.setdefault("frame_height", frame_height)
    hit = [name for name in RULE_NAMES if name in rules and rule_triggered(rules[name], probe)]
    return min(3, len(hit)), hit


def judge_frame(frame):
    """整帧判定：返回 [(object_index, difficult_level, [规则名...]), ...]，只含判定过的目标。

    帧宽高会传给对象级判定，供 too_small 按画幅比例缩放阈值。
    """
    out = []
    fw = frame.get("width")
    fh = frame.get("height")
    for i, ob in enumerate(frame.get("objects") or []):
        if not isinstance(ob, dict):
            continue
        lvl, hits = judge_difficult(ob, frame_width=fw, frame_height=fh)
        out.append((i, lvl, hits))
    return out


# ══════════════════════════════════════════════════════════════════════════════════
# 五、归一（旧字段 → 新字段）
# ══════════════════════════════════════════════════════════════════════════════════

def normalize_object(ob, light_level=None, weather=None):
    """单个目标：旧字段补齐成新字段；新旧两套键同时保留（下游只认旧名的照样能跑）。

    参数：
      ob          真值 objects[] 里的一项（可以是旧格式，也可以是已经归一过的）
      light_level 帧级光照（light_level_of_frame 的结果）；对象自己带了就优先用对象的
      weather     帧级天气（weather_of_frame 的结果）；同上

    安全默认（宁可给保守值，不让下游 KeyError）：
      class_name      → ob.class → "unknown"（不在 CLASS_NAMES 里时原样保留，由 validate_frame 报错）
      class_id        → CLASS_ID[class_name] → 0
      bbox            → ob.bbox_2d → None（没有框就没法标，validate_frame 会报错）
      occlusion_ratio → ob.occlusion → 0.0，并裁到 0~1
      visible_ratio   → 1 - occlusion_ratio（缺失时）
      distance        → 0.0，负值归 0
      light_level     → 帧级值 → front_light
      weather         → 帧级值 → clear
      wear_level      → 0.0（现状引擎不产出，见文件头口径(c)）
      difficult_level → judge_difficult 现算（不信任外部写进来的值，避免与规则漂移）
    返回值是新的 dict，不改入参。
    """
    src = ob if isinstance(ob, dict) else {}
    out = dict(src)             # 先把未知字段原样留下（forward-compat）

    cls = src.get("class_name")
    if not cls:
        cls = src.get("class")
    out["class_name"] = str(cls) if cls else FIELD_SPEC["class_name"]["default"]
    out["class"] = out["class_name"]          # 旧名同步，保证只认旧名的下游不炸

    cid = src.get("class_id")
    if cid is None:
        cid = class_id_of(out["class_name"])
    out["class_id"] = _as_int(cid, 0)

    bbox = src.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        bbox = src.get("bbox_2d")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        out["bbox"] = [round(_as_float(v, 0.0), 2) for v in bbox]
        out["bbox_2d"] = list(out["bbox"])
    else:
        out["bbox"] = None
        out.setdefault("bbox_2d", None)

    occ_raw = src.get("occlusion_ratio")
    if occ_raw is None:
        occ_raw = src.get("occlusion")
    occ = _clamp(_as_float(occ_raw, FIELD_SPEC["occlusion_ratio"]["default"]), 0.0, 1.0)
    out["occlusion_ratio"] = round(occ, 4)
    out["occlusion"] = out["occlusion_ratio"]     # 旧名同步

    vis_raw = src.get("visible_ratio")
    if vis_raw is None:
        vis = 1.0 - out["occlusion_ratio"]        # 遮挡与可见互补（同一块面积的两种说法）
    else:
        vis = _as_float(vis_raw, 1.0 - out["occlusion_ratio"])
    out["visible_ratio"] = round(_clamp(vis, 0.0, 1.0), 4)

    out["distance"] = round(max(0.0, _as_float(src.get("distance"), 0.0)), 3)

    lv = src.get("light_level")
    out["light_level"] = canonical_light_level(lv if lv else light_level)

    wx = src.get("weather")
    out["weather"] = str(wx).strip().lower() if wx else (weather or WEATHER_DEFAULT)

    wear = src.get("wear_level")
    out["wear_level"] = round(_clamp(_as_float(wear, 0.0), 0.0, 1.0), 4) \
        if is_wear_class(out["class_name"]) else 0.0

    out["truncated"] = bool(src.get("truncated", False))
    out["instance_id"] = _as_int(src.get("instance_id"), 0)

    lvl, _hits = judge_difficult(out)
    out["difficult_level"] = lvl
    return out


def normalize_frame(frame):
    """整帧归一：返回新 dict（对象全部替换为归一后的对象），不改入参。

    帧级 width/height/lighting 原样保留；额外写帧级 light_level / weather，方便按帧做统计与检索。
    """
    src = frame if isinstance(frame, dict) else {}
    out = dict(src)
    lv = light_level_of_frame(src)
    wx = weather_of_frame(src)
    out["light_level"] = lv
    out["weather"] = wx
    objs = []
    for ob in (src.get("objects") or []):
        objs.append(normalize_object(ob, light_level=lv, weather=wx))
    out["objects"] = objs
    return out


# ══════════════════════════════════════════════════════════════════════════════════
# 六、校验
# ══════════════════════════════════════════════════════════════════════════════════

def _check_value(where, name, spec, value, errs):
    """按 spec 检查一个值；errs 是累积列表（不抛异常，一次列全问题）。"""
    kind = spec.get("kind")
    if kind == "str":
        if not isinstance(value, str) or not value.strip():
            errs.append("%s.%s 应为非空字符串（实际 %r）" % (where, name, value))
            return
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            errs.append("%s.%s=%r 不在允许取值 %s 内" % (where, name, value, enum))
    elif kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            errs.append("%s.%s 应为整数（实际 %r）" % (where, name, value))
            return
        lo, hi = spec.get("lo"), spec.get("hi")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            errs.append("%s.%s=%d 超出范围 [%s, %s]" % (where, name, value, lo, hi))
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            errs.append("%s.%s 应为数值（实际 %r）" % (where, name, value))
            return
        lo, hi = spec.get("lo"), spec.get("hi")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            errs.append("%s.%s=%s 超出范围 [%s, %s]" % (where, name, value, lo, hi))
    elif kind == "bool":
        if not isinstance(value, bool):
            errs.append("%s.%s 应为布尔（实际 %r）" % (where, name, value))
    elif kind == "list4":
        if not (isinstance(value, (list, tuple)) and len(value) == 4):
            errs.append("%s.%s 应为 4 个数的框（实际 %r）" % (where, name, value))
            return
        try:
            nums = [float(v) for v in value]
        except (TypeError, ValueError):
            errs.append("%s.%s 含非数字（实际 %r）" % (where, name, value))
            return
        if nums[2] <= nums[0] or nums[3] <= nums[1]:
            errs.append("%s.%s 退化框（xmax<=xmin 或 ymax<=ymin）：%s" % (where, name, nums))
        if min(nums) < 0:
            errs.append("%s.%s 含负坐标：%s" % (where, name, nums))
    elif kind == "list7":
        if not (isinstance(value, (list, tuple)) and len(value) == 7):
            errs.append("%s.%s 应为 7 个数（实际 %r）" % (where, name, value))


def _legacy_name(field):
    """新字段 → 旧字段名（没有旧名返回 None）。"""
    for old, new in LEGACY_ALIASES.items():
        if new == field:
            return old
    return None


def _field_value(ob, name):
    """取字段值：新名字优先，再退回旧名字（bbox ← bbox_2d / class_name ← class ...）。

    返回 (是否有值, 值)。注意用「键是否存在」判断而不是 bool(value) —— 0 / 0.0 / False 都是合法值。
    """
    if name in ob:
        return True, ob[name]
    legacy = _legacy_name(name)
    if legacy and legacy in ob:
        return True, ob[legacy]
    return False, None


def validate_frame(frame, strict_required=True, normalize_first=False, info=None):
    """校验一帧真值；返回错误列表（空列表 = 合法）。

    参数：
      frame           真值 JSON 的 dict。旧格式（class / bbox_2d / occlusion）可直接吃。
      strict_required 缺字段口径，两档：
        True （默认，严格）—— 统一字段必须逐键存在：class_name / class_id / bbox / occlusion_ratio /
               visible_ratio / distance / light_level / difficult_level / weather 缺一个报一条，
               wear_level 仍按 FIELD_SPEC 记为可选（非 TSR 类不产出它）。用于验收「统一输出」。
        False（宽容）—— 只要求「旧字段或新字段」有其一：引擎刚写出的旧真值不报缺字段，
               但值本身仍逐项校验（范围/枚举/框合理性/难例自洽）。适合体检历史批次。
      normalize_first True 时先 normalize_frame 再校验 —— 「归一后文件必然通过严格校验」的保证。
      info            可选的 list：宽容口径下「缺字段已用默认值补齐」的记录会追加进去（不算错误）。
                      CLI 用它打印 INFO 行，便于看出旧真值靠哪些默认值跑通。
    返回的错误信息都以 frame_id 开头，可以直接 grep / 贴进日志。
    """
    strict = bool(strict_required)           # 统一落成局部名字，后续分支只看它（避免被同名变量遮蔽）
    if normalize_first:
        frame = normalize_frame(frame)
    errs = []
    if not isinstance(frame, dict):
        return ["帧不是对象（实际 %r）" % type(frame).__name__]
    fid = frame.get("frame_id") or "<无 frame_id>"
    where = str(fid)

    for fkey, fspec in FRAME_SPEC.items():
        if fspec.get("req") and fkey not in frame:
            errs.append("%s: 缺帧级必需字段 %s（%s）" % (where, fkey, fspec.get("desc")))
            continue
        if fkey not in frame:
            continue
        fval = frame[fkey]
        if fspec.get("kind") == "objects" and not isinstance(fval, list):
            errs.append("%s.objects 应为列表（实际 %r）" % (where, type(fval).__name__))
        _check_value(where, fkey, fspec, fval, errs)

    w = _as_float(frame.get("width"), 0.0)
    h = _as_float(frame.get("height"), 0.0)
    objs = frame.get("objects")
    if not isinstance(objs, list):
        return errs
    for idx, ob in enumerate(objs):
        ow = "%s.objects[%d]" % (where, idx)
        if not isinstance(ob, dict):
            errs.append("%s 不是对象（实际 %r）" % (ow, type(ob).__name__))
            continue
        norm_ob = normalize_object(ob, light_level_of_frame(frame), weather_of_frame(frame))
        # 每个字段单独判断「文件里给没给值」：给了就校验值；没给就记进 absent（不全靠默认值糊过去）
        absent = []                        # 本对象在文件里缺失的必需字段名
        for fname, fspec in FIELD_SPEC.items():
            have, value = _field_value(ob, fname)
            if have:
                _check_value(ow, fname, fspec, value, errs)
                continue
            if not fspec.get("req"):
                continue                   # 可选字段：缺了不报
            absent.append(fname)           # 旧名（class/bbox_2d/occlusion）也算已提供，两套都缺才记
        # 必需字段缺失的收口：严格口径 = 错误（缺一个报一条）；宽容口径 = 信息（默认值补齐，不算错）
        for mname in absent:
            msg = "%s 缺字段 %s（%s）" % (ow, mname, FIELD_SPEC[mname].get("desc"))
            if strict:
                errs.append(msg)
            elif info is not None:
                info.append(msg + " —— 已用默认值补齐")
            else:
                INFO.append(msg + " —— 已用默认值补齐")
        # 框必须落在画面内（越界说明投影或裁切有问题，训练时会喂进错框）
        box = norm_ob.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4 and w > 0 and h > 0:
            try:
                x1, y1, x2, y2 = [float(v) for v in box]
            except (TypeError, ValueError):
                pass
            else:
                if x2 > w + 1e-6 or y2 > h + 1e-6:
                    errs.append("%s 框超出画面：box=%s 画面=%sx%s" % (ow, list(box), int(w), int(h)))
        # 难例难度的自洽性：difficult_level 与现算结果不一致就报（规则改了但文件没重跑）
        if "difficult_level" in ob:
            lvl, _hits = judge_difficult(norm_ob)
            if _as_int(ob.get("difficult_level"), -1) != lvl:
                errs.append("%s difficult_level=%r 与规则现算 %d 不一致（规则改了没重跑？）"
                            % (ow, ob.get("difficult_level"), lvl))
        if "visible_ratio" in ob and "occlusion_ratio" in ob:
            vis, occ = _as_float(ob.get("visible_ratio"), -1.0), _as_float(ob.get("occlusion_ratio"), -1.0)
            if abs(vis + occ - 1.0) > 1e-3:
                errs.append("%s visible_ratio(%.3f) + occlusion_ratio(%.3f) != 1"
                            % (ow, vis, occ))
    return errs


def validate_file(path, strict_required=True, normalize_first=False):
    """读一个真值 JSON 文件并校验；解析失败也算一条错误。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            frame = json.load(f)
    except Exception as e:                    # 坏 JSON 不能让批量体检中断
        return ["%s: JSON 解析失败：%s" % (os.path.basename(path), e)]
    errs = validate_frame(frame, strict_required=strict_required,
                          normalize_first=normalize_first)
    return ["%s: %s" % (os.path.basename(path), e) for e in errs]


# ══════════════════════════════════════════════════════════════════════════════════
# 七、CLI
# ══════════════════════════════════════════════════════════════════════════════════

def iter_frame_files(meta_dir):
    """meta 目录下的逐帧 JSON（按文件名排序；没有 frame_id 的 settings/params 类 JSON 跳过）。"""
    for fn in sorted(os.listdir(meta_dir)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(meta_dir, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                fr = json.load(f)
        except Exception:
            continue
        if isinstance(fr, dict) and fr.get("frame_id"):
            yield p, fr


def main(argv=None):
    ap = argparse.ArgumentParser(description="统一真值字段：校验 / 归一")
    ap.add_argument("--meta", required=True, help="逐帧真值 JSON 目录")
    ap.add_argument("--normalized", default="", help="把归一后的新字段真值另存到这个目录（不改原文件）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 帧（0=全部）")
    ap.add_argument("--strict", action="store_true",
                    help="严格口径：统一字段必须逐键存在（验收归一后文件用；默认宽容口径只校验已写出的值）")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.meta):
        sys.stderr.write("目录不存在：%s\n" % a.meta)
        return 2

    n_frame = n_obj = n_err = n_info = 0
    rule_hits = dict((k, 0) for k in RULE_NAMES)
    loose_mode = not a.strict              # 默认宽容：旧真值也能扫，只是不报缺字段
    mode = "严格口径（统一字段必须逐键存在）" if a.strict else "宽容口径（旧字段/帧级字段可替代新字段）"
    print("truth_schema: 校验口径 = %s" % mode)
    for p, fr in iter_frame_files(a.meta):
        if a.limit and n_frame >= a.limit:
            break
        n_frame += 1
        infos = []
        errs = validate_frame(fr, strict_required=not loose_mode, info=infos)
        n_err += len(errs)
        for e in errs:
            print("FAIL %s" % e)
        n_info += len(infos)
        nf = normalize_frame(fr)
        errs2 = validate_frame(nf, strict_required=not loose_mode)
        n_err += len(errs2)
        for e in errs2:
            print("FAIL(归一后) %s" % e)
        for ob in nf["objects"]:
            n_obj += 1
            _lvl, hits = judge_difficult(ob)
            for r in hits:
                rule_hits[r] += 1
        if a.normalized:
            os.makedirs(a.normalized, exist_ok=True)
            dst = os.path.join(a.normalized, "%s.json" % fr["frame_id"])
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(nf, f, ensure_ascii=False, indent=1)
                f.write("\n")
            os.replace(tmp, dst)
    print("truth_schema: %d 帧 / %d 目标，错误 %d 条，缺字段补齐 %d 处%s" % (
        n_frame, n_obj, n_err, n_info,
        "，已归一另存 → %s" % a.normalized if a.normalized else ""))
    if n_info:
        print("  提示：上面 %d 处缺失已用默认值补齐（宽容口径不算错；要验收统一格式请加 --strict）"
              % n_info)
    for r in RULE_NAMES:
        print("  规则 %-10s 命中目标 %d 个 —— %s" % (r, rule_hits[r], RULES[r]["desc"]))
    if n_frame == 0:
        sys.stderr.write("WARN 目录下没有带 frame_id 的真值 JSON：%s\n" % a.meta)
        return 2
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
