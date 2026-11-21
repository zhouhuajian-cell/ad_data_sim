"""素材库：统一加载 OBJ / GND 素材，支持多视角。

目录规范（你往里放）：

/data/obj_assets/
  <class>/
    <variant>/                     ← 变体（如 orange / white / rusty）
      meta.json                    ← 必须：真实尺寸、视角列表
      front.png                    ← 正面视角（必须）
      front_night.png              ← 夜间变体（可选）
      side_left.png / side_right.png
      rear.png
    <variant>.json                 ← 也支持放同级的 meta

/data/gnd_assets/
  <class>/
    <variant>/
      meta.json
      tex.png                      ← 重复单元（纵向 = 长度）
  meta.json                        ← 全局默认

meta.json 格式：
{
  "class": "cone",
  "size_m": {"width": 0.28, "height": 0.70, "length": 0.28},
  "anchor": "bottom_center",
  "views": ["front", "side_left", "rear"],
  "lighting": ["day", "night"],
  "source": "kenney-cc0",
  "license": "CC0"
}
"""
import json
import os

OBJ_ROOT = "/data/obj_assets"
GND_ROOT = "/data/gnd_assets"

VIEW_PRIORITY = ["front", "side_left", "side_right", "rear", "top"]
LIGHT_PRIORITY = ["day", "night", "dusk"]


def _read_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def scan(root):
    """扫描素材根目录 → {class: {variant: meta}}"""
    out = {}
    if not os.path.isdir(root):
        return out

    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        out[cls] = {}

        # 情况 A：<class>/<variant>/ 子目录
        for var in sorted(os.listdir(cdir)):
            vdir = os.path.join(cdir, var)
            if not os.path.isdir(vdir):
                continue
            meta_p = os.path.join(vdir, "meta.json")
            meta = _read_json(meta_p)
            if meta is None:
                # 没 meta 也收，推断尺寸
                meta = _infer_meta(cls, var, vdir)
            else:
                meta = _merge_meta(cls, var, vdir, meta)
            out[cls][var] = meta

        # 情况 B：<class>/<variant>.png + .json
        files = [f for f in os.listdir(cdir) if f.lower().endswith(".png")]
        for f in files:
            var = os.path.splitext(f)[0]
            if var in out[cls]:
                continue
            mp = os.path.join(cdir, var + ".json")
            meta = _read_json(mp) or _infer_meta(cls, var, cdir, f)
            meta = _merge_meta(cls, var, cdir, meta, single=f)
            out[cls][var] = meta

    return out


def _infer_meta(cls, var, d, single=None):
    """没有 meta 时的默认值。"""
    DEFAULT_SIZE = {
        "cone": (0.28, 0.70),
        "cone_flat": (0.30, 0.15),
        "car": (1.80, 1.50),
        "suv": (1.90, 1.75),
        "truck": (2.50, 3.20),
        "bus": (2.55, 3.30),
        "van": (2.00, 2.30),
        "pedestrian": (0.55, 1.70),
        "rider": (0.70, 1.75),
        "three": (1.30, 1.60),
    }
    w, h = DEFAULT_SIZE.get(cls.split("_")[0], (0.6, 1.0))
    return {
        "class": cls,
        "variant": var,
        "size_m": {"width": w, "height": h, "length": w},
        "anchor": "bottom_center",
        "views": ["front"],
        "lighting": ["day"],
        "_inferred": True,
    }


def _merge_meta(cls, var, d, meta, single=None):
    """补全并核实文件存在。"""
    m = dict(meta or {})
    m.setdefault("class", cls)
    m.setdefault("variant", var)
    sz = m.get("size_m") or {}
    base = _infer_meta(cls, var, d)["size_m"]
    for k in ("width", "height", "length"):
        if k not in sz:
            sz[k] = base[k]
    m["size_m"] = sz
    m.setdefault("anchor", "bottom_center")

    # 扫可用视角
    files = {}
    if single:
        files["front"] = os.path.join(d, single)
    else:
        for f in os.listdir(d):
            if not f.lower().endswith(".png"):
                continue
            stem = os.path.splitext(f)[0]
            # 解析 <view>[_<light>]
            light = "day"
            view = stem
            for L in LIGHT_PRIORITY[1:]:
                if stem.endswith("_" + L):
                    view = stem[:-(len(L) + 1)]
                    light = L
                    break
            if view in VIEW_PRIORITY:
                files[view + "|" + light] = os.path.join(d, f)

    m["_files"] = files
    m["_dir"] = d
    if "views" not in m:
        m["views"] = sorted(set(k.split("|")[0] for k in files)) or ["front"]
    return m


def pick(cls, view="front", light="day", root=OBJ_ROOT):
    """选一个素材文件。返回 (path, meta) 或 (None, None)。

    回退链：精确匹配 → 白天同视角 → front → 任意
    """
    lib = scan(root) if not hasattr(pick, "_cache") else pick._cache
    if not hasattr(pick, "_cache"):
        pick._cache = lib

    c = lib.get(cls) or {}
    if not c:
        return None, None

    # 优先选带该视角的变体
    for var, m in c.items():
        key = view + "|" + light
        if key in m["_files"]:
            return m["_files"][key], m
    for var, m in c.items():
        key = view + "|day"
        if key in m["_files"]:
            return m["_files"][key], m
    for var, m in c.items():
        if "front|day" in m["_files"]:
            return m["_files"]["front|day"], m
    for var, m in c.items():
        if m["_files"]:
            k = sorted(m["_files"])[0]
            return m["_files"][k], m
    return None, None


def refresh():
    """素材更新后调用。"""
    if hasattr(pick, "_cache"):
        del pick._cache


def list_all(root=OBJ_ROOT):
    """返回可读的清单。"""
    lib = scan(root)
    out = []
    for cls, vars_ in lib.items():
        for var, m in vars_.items():
            out.append({
                "class": cls, "variant": var,
                "size_m": m["size_m"],
                "views": m.get("views", []),
                "lighting": m.get("lighting", []),
                "files": len(m.get("_files", {})),
                "inferred": bool(m.get("_inferred")),
            })
    return out


def ensure_dirs():
    """创建目录规范（幂等）。"""
    os.makedirs(OBJ_ROOT, exist_ok=True)
    os.makedirs(GND_ROOT, exist_ok=True)
    readme = os.path.join(OBJ_ROOT, "_README.txt")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write(OPEN_SPEC)


OPEN_SPEC = """OBJ 素材目录规范
===================

路径：/data/obj_assets/<class>/<variant>/

【最简单】把 PNG 直接扔进 /data/obj_assets/cone/ 即可，会自动识别。

推荐结构（多视角）：
  /data/obj_assets/cone/orange/
      meta.json
      front.png          正面（必须）
      front_night.png    夜间
      side_left.png      左侧
      rear.png           后方

meta.json 示例：
{
  "class": "cone",
  "size_m": {"width": 0.28, "height": 0.70, "length": 0.28},
  "anchor": "bottom_center",
  "views": ["front", "side_left", "rear"],
  "lighting": ["day", "night"],
  "source": "kenney-cc0",
  "license": "CC0"
}

【PNG 要求】
  - RGBA，背景完全透明
  - 物体底边贴 PNG 底边（这是接地点）
  - 高度 >= 200px
  - 正对车头视角

【class 建议值】
  cone / cone_flat / car / suv / truck / bus / van
  pedestrian / rider / three（三轮车）

GND 素材目录规范
==================

路径：/data/gnd_assets/<class>/<variant>/
  tex.png              重复单元，纵向=长度，横向=宽度
  meta.json

meta.json 示例：
{
  "class": "lane_dashed_white",
  "width_m": 0.15,
  "tile_length_m": 4.0,
  "dash_m": 6.0,
  "gap_m": 9.0,
  "color": "white"
}
"""


if __name__ == "__main__":
    import sys
    if "--init" in sys.argv:
        ensure_dirs()
        print("已创建:", OBJ_ROOT, GND_ROOT)
        print("说明:", os.path.join(OBJ_ROOT, "_README.txt"))

    print("\n=== OBJ 素材 ===")
    items = list_all(OBJ_ROOT)
    if not items:
        print("  (空) 把 PNG 放进 %s/<class>/" % OBJ_ROOT)
    for it in items:
        print("  %-12s %-12s %s  views=%s files=%d%s"
              % (it["class"], it["variant"],
                 "%.2fx%.2fm" % (it["size_m"]["width"], it["size_m"]["height"]),
                 ",".join(it["views"]), it["files"],
                 "  (推断)" if it["inferred"] else ""))

    print("\n=== GND 素材 ===")
    g = list_all(GND_ROOT)
    if not g:
        print("  (空) 把 tex.png 放进 %s/<class>/<variant>/" % GND_ROOT)
    for it in g:
        print("  %-24s %s" % (it["class"], it["variant"]))
