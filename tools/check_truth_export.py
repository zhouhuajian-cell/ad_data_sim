# -*- coding: utf-8 -*-
# check_truth_export.py — 真值导出链路端到端校验（不依赖 UE，本机/服务器都能跑）
# 干什么：用 tools/mock_frames.py 造 8 帧假数据，再叠加一份「手工真值」覆盖（遮挡 / 远距离 / 逆光 /
#   磨损 / 退化框 各造一个典型目标），然后真实跑 post/voc_export.py 与 post/hardcase_export.py，
#   对产物逐条断言：XML 份数、可解析性、object 数、bndbox 合法性、difficult 与难例判定一致、
#   hardcase CSV 行数与 rules 列、truth_schema.validate_frame 的正反例。任何一条不符立即 FAIL。
# 怎么跑：python tools/check_truth_export.py [--frames 8] [--keep] [--out DIR]
# 成功判据：末行「真值导出校验通过：PASS n / FAIL 0」，退出码 0。产物默认跑完清理，--keep 保留便于排查。
# 与 dryrun_check.py 的分工：dryrun 管「引擎外主链路」（队列→后处理→封装），本脚本只管
#   「真值 JSON → VOC XML / hardcase CSV / 统一字段」这一段（方案模块 5），互不重复。
# ⚠️ 通过只证明脚本契约自洽，不证明引擎内真值字段与方案一致（例如磨损字段引擎尚未产出，见 common/truth_schema.py 文件头）。
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PASS, FAIL, SKIP = [], [], []
META_DIR = ""          # 校验用的真值目录（main 里赋值，供断言直接回读磁盘上的真值）

FRAME_FILES = ["post/voc_export.py", "post/hardcase_export.py", "common/truth_schema.py",
               "tools/check_truth_export.py"]


def _env():
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"     # 子进程输出统一 utf-8，避免 Windows 本地调试乱码
    e["PYTHONUTF8"] = "1"
    return e


def run(args, cwd=ROOT, expect=0):
    p = subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=_env())
    if p.returncode != expect:
        raise AssertionError("命令退出码 %d（预期 %d）\n  cmd: %s\n  stdout:\n%s\n  stderr:\n%s"
                             % (p.returncode, expect, " ".join(args),
                                (p.stdout or "").strip()[-1500:],
                                (p.stderr or "").strip()[-1500:]))
    return p


def check(name, fn):
    try:
        fn()
    except Exception as e:
        FAIL.append(name)
        msg = str(e) or repr(e)
        print("FAIL %s\n     %s" % (name, msg.replace("\n", "\n     ")))
    else:
        PASS.append(name)
        print("PASS %s" % name)


def skip(name, why):
    SKIP.append(name)
    print("SKIP %s（%s）" % (name, why))


def load_module(name, rel_path):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def meta_path(frame_id):
    """校验用的真值目录（main 里赋值）。"""
    return os.path.join(META_DIR, "%s.json" % frame_id)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def count_files(d, suffix):
    return sum(1 for f in os.listdir(d) if f.endswith(suffix)) if os.path.isdir(d) else 0


def find_image(img_dir, fr):
    """与 voc_export.find_image 同口径：先成品 <frame_id>.png，再原始 <frame_id>_rgb.png。"""
    for name in ("%s.png" % fr["frame_id"], "%s_rgb.png" % fr["frame_id"]):
        if os.path.exists(os.path.join(img_dir, name)):
            return name
    return None


# ── 手工真值覆盖：让 8 帧里必然出现 4 类难例各至少一个，另有 1 个退化框 ─────────────────
# 按帧号下标写入，帧 = mock_frames 造的 av_XXXXXX（严格递增，输出顺序确定）。
OVERRIDES = {
    0: {"lighting": {"condition": "front_light", "weather": "clear"},
        "objects": [
            {"class": "vehicle", "instance_id": 1, "bbox_2d": [10.0, 10.0, 110.0, 90.0],
             "occlusion": 0.0, "distance": 12.0, "truncated": False},                          # 正常帧，0 难例
        ]},
    1: {"lighting": {"condition": "front_light", "weather": "clear"},
        "objects": [
            {"class": "pedestrian", "instance_id": 2, "bbox_2d": [20.0, 20.0, 80.0, 120.0],
             "occlusion": 0.9, "distance": 15.0, "truncated": False},                         # 遮挡难例
            {"class": "vehicle", "instance_id": 3, "bbox_2d": [100.0, 40.0, 200.0, 100.0],
             "occlusion": 0.1, "distance": 12.0, "truncated": False},                         # 正常
        ]},
    2: {"lighting": {"condition": "front_light", "weather": "clear"},
        "objects": [
            {"class": "cone", "instance_id": 4, "bbox_2d": [30.0, 30.0, 60.0, 60.0],
             "occlusion": 0.0, "distance": 100.0, "truncated": False},                        # 远距离难例
            {"class": "vehicle", "instance_id": 5, "bbox_2d": [5.5, 5.5, 60.2, 60.9],
             "occlusion": 0.0, "distance": 8.0, "truncated": False},                          # 坐标取整的边界用例
        ]},
    3: {"lighting": {"condition": "back_light", "weather": "clear"},
        "objects": [
            {"class": "traffic_sign", "instance_id": 6, "bbox_2d": [40.0, 20.0, 70.0, 60.0],
             "occlusion": 0.2, "distance": 20.0, "truncated": False},                         # 逆光（back_light→backlight）
        ]},
    4: {"lighting": {"condition": "night", "weather": "rain"},
        "objects": [
            {"class": "vehicle", "instance_id": 7, "bbox_2d": [0.0, 0.0, 120.0, 90.0],
             "occlusion": 0.5, "distance": 95.0, "truncated": False},                         # 夜间→low_light + 三条规则
            {"class": "cone", "const": 1, "instance_id": 8, "bbox_2d": [300.0, 150.0, 319.0, 179.0],
             "occlusion": 0.0, "distance": 5.0, "truncated": False},                          # 贴右下角的框
        ]},
    5: {"lighting": {"condition": "dusk", "weather": "clear"},
        "objects": [
            # 磨损难例：只有 TSR/车道线类才判 → 0.7 命中，同帧 vehicle 的 0.7 不命中（一次验证两条口径）
            {"class": "traffic_sign", "instance_id": 9, "bbox_2d": [50.0, 30.0, 90.0, 80.0],
             "occlusion": 0.0, "distance": 30.0, "truncated": False, "wear_level": 0.7},
            {"class": "vehicle", "instance_id": 10, "bbox_2d": [130.0, 60.0, 200.0, 130.0],
             "occlusion": 0.0, "distance": 25.0, "truncated": False, "wear_level": 0.7},
        ]},
    6: {"lighting": {"condition": "front_light", "weather": "clear"},
        "objects": [
            {"class": "vehicle", "instance_id": 11, "bbox_2d": [20.0, 20.0, 101.0, 80.0],
             "occlusion": 0.0, "distance": 9.0, "truncated": False},
            {"class": "cone", "instance_id": 12, "bbox_2d": [50.0, 50.0, 50.6, 90.0],
             "occlusion": 0.0, "distance": 7.0, "truncated": False},                          # 退化框（宽 0.6px）
        ]},
    7: {"lighting": {"condition": "back_light", "weather": "clear"},
        "objects": [
            {"class": "pedestrian", "instance_id": 13, "bbox_2d": [150.0, 80.0, 170.0, 150.0],
             "occlusion": 0.0, "distance": 150.0, "truncated": True},                         # 远距离 + 逆光
        ]},
}


def apply_overrides(meta_dir):
    """把 OVERRIDES 写进 mock 出来的 meta/<frame_id>.json（只改 objects / lighting）。"""
    names = sorted(fn for fn in os.listdir(meta_dir)
                   if fn.endswith(".json") and read_json(os.path.join(meta_dir, fn)).get("frame_id"))
    for idx, patch in OVERRIDES.items():
        path = os.path.join(meta_dir, names[idx])
        fr = read_json(path)
        for key, val in patch.items():
            fr[key] = val
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fr, f, ensure_ascii=False, indent=1)
    return [read_json(os.path.join(meta_dir, n)) for n in names]


def expected_frame(fr, ts):
    """独立于 voc_export 的期望值：直接调 truth_schema 归一后自己算框（不复用 voc_export.to_voc_box）。

    注意这里是「同一套 schema 的第二次计算」而不是复制实现：框的换算用下面的内联公式，
    与 voc_export 的写法独立，用来交叉验证坐标口径（右开 → VOC 含端点）。
    """
    nf = ts.normalize_frame(fr)
    w, h = int(fr["width"]), int(fr["height"])
    boxes = []
    for ob in nf["objects"]:
        x1, y1, x2, y2 = [float(v) for v in ob["bbox"]]
        x1 = min(max(x1, 0.0), w)
        x2 = min(max(x2, 0.0), w)
        y1 = min(max(y1, 0.0), h)
        y2 = min(max(y2, 0.0), h)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        # 右开浮点框 → 含端点整数框：xmin = 第一个覆盖像素的 1-based 序号，xmax = ceil(x2)
        xmin = int(x1) + 1
        xmax = int(x2) if float(x2).is_integer() else int(x2) + 1
        ymin = int(y1) + 1
        ymax = int(y2) if float(y2).is_integer() else int(y2) + 1
        boxes.append({"class_name": ob["class_name"], "box": (xmin, ymin, xmax, ymax),
                      "difficult_level": ob["difficult_level"]})
    return nf, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--keep", action="store_true", help="保留校验目录")
    ap.add_argument("--out", default="", help="校验目录（默认临时目录）")
    a = ap.parse_args()
    if a.frames < 8:
        raise SystemExit("--frames 至少 8（难例覆盖表按帧号 0~7 构造）")
    base = os.path.abspath(a.out) if a.out else tempfile.mkdtemp(prefix="ue5_truth_check_")
    os.makedirs(base, exist_ok=True)
    print("== 校验目录 %s" % base)
    print("== Python %s" % sys.version.split()[0])

    # ---------- 0. 四个文件在不在（服务器上可能只同步了部分目录） ----------
    missing = [p for p in FRAME_FILES if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        sys.stderr.write("校验缺文件：%s\n把仓库根目录（post/、common/、tools/）整体同步过来再跑。\n"
                         % "、".join(missing))
        return 2
    ts = load_module("check_truth_schema", "common/truth_schema.py")

    # ---------- 1. mock 假帧 + 手工覆盖 ----------
    data = os.path.join(base, "data")
    run(["tools/mock_frames.py", "--out", data, "--frames", str(a.frames), "--bit", "8"])
    meta = os.path.join(data, "meta")
    raw = os.path.join(data, "raw")
    global META_DIR
    META_DIR = meta
    frames = apply_overrides(meta)
    print("== 真值：%d 帧（已叠加难例覆盖）" % len(frames))

    voc_out = os.path.join(base, "voc")
    ann_dir = os.path.join(voc_out, "Annotations")
    hc_csv = os.path.join(base, "hardcase_list.csv")
    voc_stdout = ""
    hc_stdout = ""

    def t_voc_run():
        nonlocal voc_stdout
        p = run(["post/voc_export.py", "--meta", meta, "--images", raw, "--out", voc_out, "--clear"])
        voc_stdout = p.stdout
        assert "VOC: %d 帧" % len(frames) in p.stdout, p.stdout
    check("voc_export 正常跑完（%d 帧）" % len(frames), t_voc_run)

    def t_hc_run():
        nonlocal hc_stdout
        p = run(["post/hardcase_export.py", "--meta", meta, "--out", hc_csv, "--images", raw])
        hc_stdout = p.stdout
        assert os.path.exists(hc_csv), "没写出 CSV"
    check("hardcase_export 正常跑完", t_hc_run)

    # ---------- 2. XML 数量 / 可解析 / object 数 ----------
    def t_xml_count():
        assert count_files(ann_dir, ".xml") == len(frames), \
            "XML 份数 %d != 帧数 %d" % (count_files(ann_dir, ".xml"), len(frames))
        for fr in frames:
            p = os.path.join(ann_dir, "%s.xml" % fr["frame_id"])
            assert os.path.exists(p), "缺 XML: %s" % p
            try:
                ET.parse(p)
            except Exception as e:
                raise AssertionError("XML 解析失败 %s: %s" % (p, e))
        sets = os.path.join(voc_out, "ImageSets", "Main", "train.txt")
        lines = [ln for ln in open(sets, encoding="utf-8").read().splitlines() if ln.strip()]
        assert len(lines) == len(frames), "ImageSets/Main/train.txt 行数 %d != 帧数" % len(lines)
    check("XML 份数 == 帧数、全部可解析、ImageSets 一致", t_xml_count)

    def t_xml_objects():
        for fr in frames:
            root = ET.parse(os.path.join(ann_dir, "%s.xml" % fr["frame_id"])).getroot()
            assert root.tag == "annotation", "根节点不是 annotation"
            for tag in ("folder", "filename", "path", "source", "size", "segmented"):
                assert root.find(tag) is not None, "%s 缺标准节点 <%s>" % (fr["frame_id"], tag)
            assert int(root.find("size/width").text) == int(fr["width"])
            assert int(root.find("size/height").text) == int(fr["height"])
            assert int(root.find("size/depth").text) == 3
            _nf, want = expected_frame(fr, ts)
            objs = root.findall("object")
            assert len(objs) == len(want), "%s object 数 %d != 期望 %d（退化框应跳过）" % (
                fr["frame_id"], len(objs), len(want))
            for ob, w in zip(objs, want):
                assert ob.findtext("name") == w["class_name"], "类别名不符"
                assert ob.findtext("pose") == "Unspecified"
                for tag in ("distance", "occlusion", "visible_ratio", "light_level",
                            "weather", "wear_level", "difficult_level"):
                    assert ob.find(tag) is not None, "%s 缺扩展节点 <%s>" % (fr["frame_id"], tag)
                # 扩展节点必须在 <bndbox> 之后（VOC 解析器按顺序找标准节点，扩展放后面最安全）
                tags = [c.tag for c in ob]
                assert tags.index("bndbox") < tags.index("distance"), "扩展节点没放在 bndbox 之后"
    check("XML 结构 + object 数 == 有效框数（退化框被跳过）", t_xml_objects)

    def t_bndbox():
        for fr in frames:
            root = ET.parse(os.path.join(ann_dir, "%s.xml" % fr["frame_id"])).getroot()
            w, h = int(fr["width"]), int(fr["height"])
            _nf, want = expected_frame(fr, ts)
            for i, (ob, exp) in enumerate(zip(root.findall("object"), want)):
                vals = []
                for tag in ("xmin", "ymin", "xmax", "ymax"):
                    txt = ob.findtext("bndbox/%s" % tag)
                    assert txt is not None and txt.strip().lstrip("-").isdigit(), \
                        "%s object[%d] %s 不是整数: %r" % (fr["frame_id"], i, tag, txt)
                    vals.append(int(txt))
                xmin, ymin, xmax, ymax = vals
                assert xmin >= 1 and ymin >= 1, "坐标必须 1-based: %s" % (vals,)
                assert xmin < xmax <= w, "x 越界或退化: %s 宽=%d" % (vals, w)
                assert ymin < ymax <= h, "y 越界或退化: %s 高=%d" % (vals, h)
                assert tuple(vals) == exp["box"], "%s object[%d] 框 %s != 期望 %s" % (
                    fr["frame_id"], i, vals, exp["box"])
    check("bndbox 全部为整数、1<=min<max<=宽高、且与坐标口径换算一致", t_bndbox)

    # ---------- 3. difficult 与难例判定一致 ----------
    def t_difficult():
        for fr in frames:
            root = ET.parse(os.path.join(ann_dir, "%s.xml" % fr["frame_id"])).getroot()
            nf, want = expected_frame(fr, ts)
            for i, (ob, exp) in enumerate(zip(root.findall("object"), want)):
                d = int(ob.findtext("difficult"))
                lvl = int(ob.findtext("difficult_level"))
                assert d == (1 if exp["difficult_level"] > 0 else 0), "%s object[%d] difficult=%d" % (
                    fr["frame_id"], i, d)
                assert lvl == exp["difficult_level"]
                if lvl == 0:
                    assert d == 0, "正常目标不该标 difficult"
        # 两帧定点验证：遮挡帧的难例目标 difficult=1，正常帧全部 difficult=0
        def objects_of(i):
            return ET.parse(os.path.join(ann_dir, "%s.xml" % frames[i]["frame_id"])).getroot()
        occ = objects_of(1)
        ped = [o for o in occ.findall("object") if o.findtext("name") == "pedestrian"][0]
        assert ped.findtext("difficult") == "1", "occlusion=0.9 的目标 difficult 应为 1"
        assert float(ped.findtext("occlusion")) > ts.RULES["occlusion"]["threshold"]
        normal = objects_of(0)
        assert [o.findtext("difficult") for o in normal.findall("object")] == ["0"], \
            "正常帧 difficult 应全为 0"
    check("difficult 与 judge_difficult 一致（遮挡帧=1 / 正常帧=0）", t_difficult)

    # ---------- 4. hardcase CSV ----------
    def t_hc_csv():
        rows = read_csv(hc_csv)
        with open(hc_csv, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["frame_id", "image_file", "object_index", "class_name",
                                         "rules", "occlusion_ratio", "visible_ratio", "distance",
                                         "light_level", "wear_level", "difficult_level"], \
                "CSV 列不符: %s" % (reader.fieldnames,)
        # 独立计算期望：逐目标跑判规则（object_index 用原始下标，与 hardcase_export 同口径）
        want = []
        for fr in frames:
            nf = ts.normalize_frame(fr)
            for idx, ob in enumerate(nf["objects"]):
                _lvl, hits = ts.judge_difficult(ob)
                if not hits:
                    continue
                want.append({"frame_id": str(nf["frame_id"]), "object_index": str(idx),
                             "class_name": ob["class_name"], "rules": ";".join(hits),
                             "difficult_level": str(_lvl)})
        assert len(rows) == len(want), "CSV 行数 %d != 命中难例的目标数 %d" % (len(rows), len(want))
        for r, w in zip(rows, want):
            assert r["frame_id"] == w["frame_id"], "行顺序/帧号不符: %s vs %s" % (r, w)
            assert r["object_index"] == w["object_index"], "object_index 不符: %s vs %s" % (r, w)
            assert r["class_name"] == w["class_name"], r
            assert r["rules"] == w["rules"], "rules 列不符: %s vs %s" % (r["rules"], w["rules"])
            assert int(r["difficult_level"]) == int(w["difficult_level"]), r
            assert r["image_file"].endswith(".png"), "image_file 应指向图片: %s" % r["image_file"]
    check("hardcase CSV 行数 == 命中目标数、rules 列正确", t_hc_csv)

    def t_hc_rules():
        rows = read_csv(hc_csv)
        by_key = dict(((r["frame_id"], r["class_name"]), r) for r in rows)
        f_occ = str(frames[1]["frame_id"])
        f_far = str(frames[2]["frame_id"])
        f_back = str(frames[3]["frame_id"])
        f_night = str(frames[4]["frame_id"])
        f_wear = str(frames[5]["frame_id"])
        assert by_key[(f_occ, "pedestrian")]["rules"] == "occlusion", by_key[(f_occ, "pedestrian")]
        # 100m 的 cone：既超远距离，框也极小（画幅 320×180 下相对面积低于 0.049%），两条都成立才对
        far_rules = set(by_key[(f_far, "cone")]["rules"].split(";"))
        assert "far_small" in far_rules, by_key[(f_far, "cone")]
        assert far_rules <= {"far_small", "too_small"}, "远距离帧不该命中其它规则: %s" % far_rules
        assert by_key[(f_back, "traffic_sign")]["rules"] == "light", "back_light 应映射成 backlight"
        assert by_key[(f_back, "traffic_sign")]["light_level"] == "backlight"
        got = set(by_key[(f_night, "vehicle")]["rules"].split(";"))
        assert got == {"occlusion", "light", "far_small"}, "夜间+遮挡+远距离应命中 3 条: %s" % got
        assert by_key[(f_night, "vehicle")]["light_level"] == "low_light", "night 应映射成 low_light"
        assert by_key[(f_night, "vehicle")]["difficult_level"] == "3", "命中 3 条规则 → difficult_level 封顶 3"
        # 磨损只对 TSR 生效：同帧 vehicle 的 wear_level=0.7 不该被算成难例
        assert by_key[(f_wear, "traffic_sign")]["rules"] == "wear", by_key[(f_wear, "traffic_sign")]
        assert (f_wear, "vehicle") not in by_key or "wear" not in by_key[(f_wear, "vehicle")]["rules"], \
            "非 TSR 类不应命中磨损规则"
        summary_line = [ln for ln in hc_stdout.splitlines() if ln.strip().startswith("occlusion ")]
        assert summary_line, "hardcase_export 没打印每条规则的汇总行:\n%s" % hc_stdout
    check("难例规则逐条命中正确（遮挡/远距离/逆光/夜间/磨损/类别限定）", t_hc_rules)

    # ---------- 5. truth_schema 校验正反例 ----------
    def t_schema_normal():
        for f0 in frames:
            # 一律回读磁盘上的真值，确保被校验的就是引擎/覆盖写出来的那一份
            fr = read_json(meta_path(f0["frame_id"]))
            # ① 宽容口径（体检引擎刚写出的旧真值）：旧字段齐 → 不报缺字段
            errs = ts.validate_frame(fr, strict_required=False)
            assert errs == [], "%s 宽容口径应为空，实际：%s" % (fr["frame_id"], errs[:3])
            # ② 严格口径对旧真值必须报缺（否则「统一输出」就没被验收）
            assert ts.validate_frame(fr, strict_required=True), \
                "%s 旧字段真值在严格口径下应报缺统一字段" % fr["frame_id"]
            # ③ 归一后：统一字段逐键齐全，严格口径必须为空
            nf = ts.normalize_frame(fr)
            for key in ("class_name", "class_id", "bbox", "occlusion_ratio", "visible_ratio",
                        "distance", "light_level", "difficult_level", "weather", "wear_level"):
                for ob in nf["objects"]:
                    assert key in ob, "归一后缺统一字段 %s" % key
            errs2 = ts.validate_frame(nf, strict_required=True)
            assert errs2 == [], "归一后文件应通过严格校验：%s" % errs2[:3]
            # ④ normalize_first 的等价写法
            assert ts.validate_frame(fr, strict_required=True, normalize_first=True) == [], \
                "%s 归一后严格校验应为空" % fr["frame_id"]
    check("validate_frame：旧真值宽容通过 / 严格报缺 / 归一后严格通过", t_schema_normal)

    def t_schema_bad():
        fr = dict(frames[0])
        bad1 = dict(fr)
        bad1.pop("width")                       # 缺帧级必需字段
        assert ts.validate_frame(bad1), "缺 width 应报错"
        bad2 = dict(fr)
        bad2["objects"] = [{"class": "vehicle", "instance_id": 1, "occlusion": 0.0,
                            "distance": 1.0}, {"bbox_2d": [1, 2, 3, 4], "occlusion": 0.0}]  # 缺类别 / 缺框
        errs = ts.validate_frame(bad2)
        assert errs, "缺 class 与 bbox_2d 应报错"
        assert any("class_name" in e for e in errs), errs
        assert any("bbox" in e for e in errs), errs
        bad3 = dict(fr)
        bad3["objects"] = [{"class": "vehicle", "class_id": 2, "bbox_2d": [-5, 0, 10, 10],
                            "occlusion": 1.5, "distance": 3.0}]
        errs3 = ts.validate_frame(bad3)
        assert any("occlusion" in e for e in errs3), "occlusion>1 应报错: %s" % errs3
        assert any("负坐标" in e for e in errs3), "负坐标应报错: %s" % errs3
        bad4 = dict(fr)
        bad4["objects"] = [dict(fr["objects"][0], difficult_level=2)]   # 难度与规则不符
        assert any("difficult_level" in e for e in ts.validate_frame(bad4)), "难度不自洽应报错"
        bad5 = dict(fr)
        bad5["objects"] = [dict(fr["objects"][0], class_name="bogus")]  # 类别不在表里
        assert any("class_name" in e for e in ts.validate_frame(bad5)), "非法类别应报错"
    check("validate_frame 对缺字段/越界/难度不自洽帧返回非空", t_schema_bad)

    def t_schema_cli():
        out = os.path.join(base, "meta_v2")
        # 宽容口径扫旧真值：旧字段齐 → 错误 0 条，同时另存归一结果
        p = run(["common/truth_schema.py", "--meta", meta, "--normalized", out])
        assert "错误 0 条" in p.stdout, p.stdout[-800:]
        assert count_files(out, ".json") == len(frames), "归一另存的帧数不对"
        nf = read_json(os.path.join(out, "%s.json" % frames[4]["frame_id"]))
        assert nf["objects"][0]["light_level"] == "low_light", "night 帧应归一成 low_light"
        assert nf["objects"][0]["visible_ratio"] == round(1.0 - nf["objects"][0]["occlusion_ratio"], 4)
        # 严格口径扫旧真值：必须报缺（与上面的严格判定一致），但归一后的文件必须全过
        p_old = run(["common/truth_schema.py", "--meta", meta, "--strict"], expect=1)
        assert "严格口径" in p_old.stdout
        p_new = run(["common/truth_schema.py", "--meta", out, "--strict"])
        assert "错误 0 条" in p_new.stdout, p_new.stdout[-800:]
        # 对归一后的目录再跑一遍难例导出（要能直接吃新字段文件）
        csv2 = os.path.join(base, "hardcase_from_meta_v2.csv")
        run(["post/hardcase_export.py", "--meta", out, "--out", csv2])
        r1 = read_csv(hc_csv)
        r2 = read_csv(csv2)
        assert [(r["frame_id"], r["class_name"], r["rules"]) for r in r1] == \
               [(r["frame_id"], r["class_name"], r["rules"]) for r in r2], "归一前后难例清单不一致"
    check("truth_schema CLI：宽容扫旧=0 错、严格扫旧=报缺、归一再跑难例清单一致", t_schema_cli)

    # ---------- 6. CLI 边界：没有难例时只有表头；缺图帧跳过 ----------
    def t_hc_empty():
        empty_meta = os.path.join(base, "meta_clean")
        os.makedirs(empty_meta, exist_ok=True)
        fr = dict(frames[0])
        with open(os.path.join(empty_meta, "clean.json"), "w", encoding="utf-8") as f:
            json.dump(fr, f, ensure_ascii=False)
        out = os.path.join(base, "hardcase_empty.csv")
        p = run(["post/hardcase_export.py", "--meta", empty_meta, "--out", out])
        rows = read_csv(out)
        assert rows == [], "没有难例时应只有表头，实际 %d 行" % len(rows)
        assert "命中 0 个目标" in p.stdout, p.stdout
    check("没有难例时产出只有表头的 CSV", t_hc_empty)

    def t_voc_limit_missing():
        out = os.path.join(base, "voc_limit")
        p = run(["post/voc_export.py", "--meta", meta, "--out", out, "--limit", "3"])
        assert "VOC: 3 帧" in p.stdout, p.stdout
        assert count_files(os.path.join(out, "Annotations"), ".xml") == 3
        # 图片目录给个空目录：全部帧缺图应报错退出 2，且不产出半个 XML
        empty_img = os.path.join(base, "no_images")
        os.makedirs(empty_img, exist_ok=True)
        out2 = os.path.join(base, "voc_noimg")
        p2 = run(["post/voc_export.py", "--meta", meta, "--images", empty_img, "--out", out2], expect=2)
        assert "缺成品图" in p2.stderr, p2.stderr
        assert count_files(os.path.join(out2, "Annotations"), ".xml") == 0, "缺图时不该产出 XML"
    check("voc_export --limit 生效；图片全缺时退出码 2 且不产出 XML", t_voc_limit_missing)

    print("-" * 60)
    if FAIL:
        print("真值导出校验失败：PASS %d / FAIL %d / SKIP %d" % (len(PASS), len(FAIL), len(SKIP)))
        print("失败项：" + "；".join(FAIL))
    else:
        print("真值导出校验通过：PASS %d / FAIL 0 / SKIP %d" % (len(PASS), len(SKIP)))
    if a.keep:
        print("保留产物：%s" % base)
    else:
        shutil.rmtree(base, ignore_errors=True)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
