# -*- coding: utf-8 -*-
# queue_gen.py — 任务清单生成（方案 4.4 四维笛卡尔展开）
# 干什么：场景 × 光照工况 × 目标组合 × 相机位姿 展开成 tasks.jsonl，每行一个任务。
# 怎么跑：python queue_gen.py --scenes scenes.json --lights lights.json --targets targets.json \
#              --cameras cameras.json -o /data/queue/tasks.jsonl --seed 20261113
# 成功判据：输出行数 = 四个输入的组合数；幂等（同参数重跑内容一致）；输入缺字段直接报错不写文件。
# 字段契约见 config/README.md；类别表与 batch_capture.SEMANTIC_STENCIL、pack_dataset.CLASSES 同源，
# 改一处必须同步另外两处（三处都在各自文件头注明）。
import argparse
import itertools
import json
import os
import sys

CLASSES = ["road", "vehicle", "pedestrian", "traffic_sign", "cone", "vegetation"]


def load_json(p):
    if not os.path.exists(p):
        raise SystemExit("配置文件不存在: %s —— 先按需编辑四维配置（场景=关卡路径，光照=日/晨/昏/夜，目标=生成组合，相机=焦距位姿）" % p)
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else list(d.values())


def _is_num_seq(v, n):
    return isinstance(v, (list, tuple)) and len(v) == n and all(
        isinstance(x, (int, float)) for x in v)


def validate(scenes, lights, targets, cameras):
    """输入契约校验：失败一次列全，避免生成半批任务后才发现字段错。"""
    errs = []
    if not scenes:
        errs.append("scenes 为空")
    for i, s in enumerate(scenes):
        if isinstance(s, str):
            if not s.strip():
                errs.append("scenes[%d] 是空字符串" % i)
        elif isinstance(s, dict):
            if not isinstance(s.get("map"), str) or not s["map"].strip():
                errs.append("scenes[%d] 缺 map 字段（关卡资产路径）" % i)
        else:
            errs.append("scenes[%d] 类型应为字符串或对象" % i)
    if not lights:
        errs.append("lights 为空")
    for i, l in enumerate(lights):
        if not isinstance(l, dict):
            errs.append("lights[%d] 应为对象（含 condition/sun_* 等）" % i)
            continue
        if not isinstance(l.get("condition"), str):
            errs.append("lights[%d] 缺 condition（后处理按它取退化参数）" % i)
    if not targets:
        errs.append("targets 为空")
    for i, combo in enumerate(targets):
        if not isinstance(combo, list) or not combo:
            errs.append("targets[%d] 应为非空的目标列表" % i)
            continue
        for j, item in enumerate(combo):
            tag = "targets[%d][%d]" % (i, j)
            if not isinstance(item, dict):
                errs.append("%s 应为对象" % tag)
                continue
            if item.get("class") not in CLASSES:
                errs.append("%s.class=%r 不在类别表 %s 中" % (tag, item.get("class"), CLASSES))
            if not isinstance(item.get("actor_class"), str) or not item["actor_class"]:
                errs.append("%s 缺 actor_class（要 spawn 的蓝图路径）" % tag)
            for key in ("location", "rotation"):
                if not _is_num_seq(item.get(key), 3):
                    errs.append("%s.%s 应为 3 个数字的数组" % (tag, key))
            if "jitter" in item and not _is_num_seq(item["jitter"], 4):
                errs.append("%s.jitter 应为 4 个数字 [dx,dy,dz,dyaw]" % tag)
    if not cameras:
        errs.append("cameras 为空")
    for i, c in enumerate(cameras):
        if not isinstance(c, dict):
            errs.append("cameras[%d] 应为对象" % i)
            continue
        for key in ("location", "rotation"):
            if not _is_num_seq(c.get(key), 3):
                errs.append("cameras[%d].%s 应为 3 个数字的数组" % (i, key))
        for key in ("focal_length", "film_back_width"):
            v = c.get(key)
            if not isinstance(v, (int, float)) or v <= 0:
                errs.append("cameras[%d].%s 应为正数" % (i, key))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="关卡列表 json")
    ap.add_argument("--lights", required=True, help="光照工况列表 json")
    ap.add_argument("--targets", required=True, help="目标组合列表 json")
    ap.add_argument("--cameras", required=True, help="相机位姿列表 json")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--prefix", default="av")
    a = ap.parse_args()
    scenes, lights = load_json(a.scenes), load_json(a.lights)
    targets, cameras = load_json(a.targets), load_json(a.cameras)
    errs = validate(scenes, lights, targets, cameras)
    if errs:
        for e in errs:
            print("FAIL 输入: %s" % e, file=sys.stderr)
        print("%d 处输入不合法，未写文件" % len(errs), file=sys.stderr)
        return 1
    total = len(scenes) * len(lights) * len(targets) * len(cameras)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    # 用固定序号生成 id → 同参数重跑得到逐字节相同的清单（可复现）
    with open(a.out, "w", encoding="utf-8") as f:
        for i, (s, l, tg, c) in enumerate(itertools.product(scenes, lights, targets, cameras)):
            tid = "%s_%06d" % (a.prefix, i)
            row = {"id": tid, "frame_id": tid, "seed": a.seed,
                   "map": s if isinstance(s, str) else s.get("map"),
                   "lighting": l, "targets": tg, "camera": c}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    if n != total:
        print("FAIL 写出 %d 条，与组合数 %d 不符" % (n, total), file=sys.stderr)
        return 1
    print("已写 %s：%d 个任务（%d 场景×%d 工况×%d 目标组合×%d 相机）" % (
        a.out, n, len(scenes), len(lights), len(targets), len(cameras)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
