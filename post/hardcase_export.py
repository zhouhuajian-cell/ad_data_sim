# -*- coding: utf-8 -*-
# hardcase_export.py — 难例判定 + hardcase_list.csv（方案模块 5：难例清单）
# 干什么：扫 meta/*.json，逐目标按 common/truth_schema.RULES 判难例（阈值只在 truth_schema 里写一次，
#   本脚本 import 它，绝不复制第二套阈值），命中任意一条就写一行 CSV，并打印「每条规则命中多少帧/多少目标」。
#   规则（见 truth_schema.RULES，改阈值请改那里）：
#     occlusion  遮挡     occlusion_ratio  > 0.4
#     far_small  远距离小目标 distance      > 80 m
#     light      光照     light_level ∈ {backlight, low_light}
#     wear       磨损     wear_level    > 0.5（仅 traffic_sign 等 TSR/车道线类）
#   difficult_level = min(3, 命中规则条数)，与 VOC 的 <difficult> 同源同算法。
# 怎么跑：python post/hardcase_export.py --meta /data/render/meta --out /data/render/hardcase_list.csv \
#             [--images /data/render/final] [--limit N] [--rules occlusion,far_small,light,wear] [--strict]
#   默认扫 meta 目录本身；若 meta 下没有逐帧 JSON 但有 meta/post_v2 子目录（归一后的真值），自动改扫它。
# 成功判据：
#   * 输出 CSV 列固定为 frame_id, image_file, object_index, class_name, rules, occlusion_ratio,
#     visible_ratio, distance, light_level, wear_level, difficult_level；
#   * 行数 == 命中难例的目标数；rules 列为分号连接的规则名（单条命中也只写一个名字）；
#   * 没有难例时照样产出只有表头的 CSV（退出码 0）；
#   * --strict 时额外把 truth_schema.validate_frame 报错的帧写进 WARN 并让退出码变 1（默认只打印不拦）。
# ⚠️ 口径（详见 common/truth_schema.py 文件头）：
#   light_level 统一写方案拼写 backlight/low_light；现有 config 的 back_light/night 会自动映射；
#   wear_level 现状恒 0（现有类别表里没有车道线类），磨损规则暂时命不中任何目标。
# 兼容：Python 3.8；只依赖标准库。
import argparse
import csv
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common import truth_schema as ts  # noqa: E402  字段表/归一/难例判定的唯一口径

CSV_COLUMNS = ["frame_id", "image_file", "object_index", "class_name", "rules",
               "occlusion_ratio", "visible_ratio", "distance", "light_level",
               "wear_level", "difficult_level"]


def image_name(frame_id):
    """成品图文件名：与 meta 同主文件名（postprocess 的命名契约）。"""
    return "%s.png" % frame_id


def find_image(image_dir, frame_id, rgb_name=None):
    """返回该帧在 image_dir 里的实际图片名：先找后处理成品 <frame_id>.png，再找原始 <frame_id>_rgb.png。

    png 的来源是二选一（跑了 postprocess 就只有成品，没跑就只有 raw），所以这里做存在性核对，
    不硬编码一条路径；目录缺失时退回成品名（CSV 里至少有个可读的名字）。
    """
    names = []
    for n in ("%s.png" % frame_id, rgb_name):
        if n and n not in names:
            names.append(n)
    if image_dir and os.path.isdir(image_dir):
        for n in names:
            if os.path.exists(os.path.join(image_dir, n)):
                return os.path.join(image_dir, n)
    return os.path.join(image_dir, names[0]) if image_dir else names[0]


def load_frames(meta_dir):
    """读逐帧真值（跳过无 frame_id 的 JSON），按 frame_id 排序保证输出稳定。"""
    frames = []
    for fn in sorted(os.listdir(meta_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(meta_dir, fn), "r", encoding="utf-8") as f:
                fr = json.load(f)
        except Exception:
            continue
        if isinstance(fr, dict) and fr.get("frame_id"):
            frames.append(fr)
    frames.sort(key=lambda f: str(f.get("frame_id")))
    return frames


def resolve_meta(meta_dir):
    """meta 目录解析：优先 meta 本身，其次 meta/post_v2（归一后的真值）。"""
    if not os.path.isdir(meta_dir):
        return None
    for fn in os.listdir(meta_dir):
        if fn.endswith(".json"):
            return meta_dir
    sub = os.path.join(meta_dir, "post_v2")
    if os.path.isdir(sub):
        return sub
    return meta_dir


def collect(frames, rules, images_dir=""):
    """逐帧判难例 → (rows, 每规则统计, 校验错误列表, 目标总数)。

    每规则统计：{"frames": set(frame_id), "objects": int}；frames 用 set 去重（同一帧多目标只算 1 帧）。
    """
    rows = []
    stats = dict((name, {"frames": set(), "objects": 0}) for name in rules)
    errs = []
    n_obj = 0
    for fr in frames:
        errs.extend(ts.validate_frame(fr))
        norm = ts.normalize_frame(fr)
        fid = str(norm.get("frame_id"))
        img_files = norm.get("image_files") if isinstance(norm.get("image_files"), dict) else {}
        img = find_image(images_dir, fid, img_files.get("rgb"))
        for idx, ob in enumerate(norm.get("objects") or []):
            n_obj += 1
            lvl, hits = ts.judge_difficult(ob, rules)
            if not hits:
                continue
            for name in hits:
                stats[name]["frames"].add(fid)
                stats[name]["objects"] += 1
            rows.append({
                "frame_id": fid,
                "image_file": img,
                "object_index": idx,
                "class_name": ob.get("class_name"),
                "rules": ";".join(hits),
                "occlusion_ratio": "%.4f" % float(ob.get("occlusion_ratio") or 0.0),
                "visible_ratio": "%.4f" % float(ob.get("visible_ratio") or 0.0),
                "distance": "%.3f" % float(ob.get("distance") or 0.0),
                "light_level": ob.get("light_level"),
                "wear_level": "%.4f" % float(ob.get("wear_level") or 0.0),
                "difficult_level": int(lvl),
            })
    return rows, stats, errs, n_obj


def write_csv(path, rows):
    """写 CSV：utf-8-sig（Excel 直接打开不乱码），换行统一 \\n；没有难例时只有表头。"""
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)
    return len(rows)


def select_rules(names):
    """--rules 校验：未知规则名直接报错（不静默忽略，否则会以为跑全了）。"""
    if not names:
        return dict(ts.RULES)
    picked = {}
    for n in [x.strip() for x in names.split(",") if x.strip()]:
        if n not in ts.RULES:
            raise SystemExit("未知规则 %r，可选：%s" % (n, ", ".join(ts.RULE_NAMES)))
        picked[n] = ts.RULES[n]
    return picked


def print_summary(stats, n_frame, n_obj, n_rows, rules):
    """打印汇总：每条规则的命中帧数 / 命中目标数 + 描述。"""
    print("难例汇总（共 %d 帧 / %d 个目标，命中 %d 个目标）：" % (n_frame, n_obj, n_rows))
    for name in rules:
        st = stats[name]
        print("  %-10s 帧 %4d / 目标 %5d   —— %s" % (
            name, len(st["frames"]), st["objects"], rules[name]["desc"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="难例判定 → hardcase_list.csv")
    ap.add_argument("--meta", required=True, help="逐帧真值 JSON 目录")
    ap.add_argument("--out", default="", help="输出 CSV 路径（默认 <meta>/../hardcase_list.csv）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 帧（0=全部）")
    ap.add_argument("--rules", default="", help="只启用部分规则（逗号分隔：%s）" % ",".join(ts.RULE_NAMES))
    ap.add_argument("--images", default="", help="成品图/原始图目录（CSV 的 image_file 用它拼路径）")
    ap.add_argument("--strict", action="store_true", help="真值校验有错时退出码 1（默认只 WARN）")
    a = ap.parse_args(argv)

    meta = resolve_meta(a.meta)
    if meta is None:
        sys.stderr.write("真值目录不存在：%s\n" % a.meta)
        return 2
    rules = select_rules(a.rules)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(meta)), "hardcase_list.csv")

    frames = load_frames(meta)
    if a.limit and a.limit > 0:
        frames = frames[:a.limit]
    if not frames:
        sys.stderr.write("没有可用帧（%s 下没有带 frame_id 的真值 JSON）\n" % meta)
        return 2

    rows, stats, errs, n_obj = collect(frames, rules, images_dir=a.images)
    n_rows = write_csv(out, rows)
    for e in errs[:20]:                      # 只打前 20 条，避免整批刷屏
        print("WARN 真值校验：%s" % e, file=sys.stderr)
    if len(errs) > 20:
        print("WARN 真值校验还有 %d 条未显示（用 python common/truth_schema.py --meta %s 看全量）"
              % (len(errs) - 20, meta), file=sys.stderr)
    print_summary(stats, len(frames), n_obj, n_rows, rules)
    print("hardcase: %d 行 → %s" % (n_rows, os.path.abspath(out)))
    if a.strict and errs:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
