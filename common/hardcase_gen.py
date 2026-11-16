# -*- coding: utf-8 -*-
# hardcase_gen.py — 难例定向批次生成器（方案模块 6 的"按需求生成"部分）
# 干什么：现有 queue_gen.py 只做四维全组合展开（均匀覆盖），难例只能靠碰运气。
#   本脚本按「难例类型」定向生成任务行：指定要哪种难例、要多少帧，就把参数推到该难例的区间里，
#   让每一帧都落在目标难例附近，而不是随机铺开。
# 怎么跑：
#   python3 common/hardcase_gen.py --kind far_small --count 40 -o /data/queue/hc_far.jsonl
#   python3 common/hardcase_gen.py --kind all --count 60 --scenes config/scenes.json \
#       --cameras config/cameras.json -o /data/queue/hc_all.jsonl
#   自定义场景（"我自己需要的场景"）：--custom 传一个 JSON，直接描述要什么
#   python3 common/hardcase_gen.py --custom my_case.json -o /data/queue/custom.jsonl
# 成功判据：输出的每个任务行都能被 batch_capture.py 消费；同一 seed 重跑结果逐字节一致。
#
# 难度定向手段：在难例参数区间内均匀采样，其余维度（场景/工况/目标组合/相机）轮转覆盖。
# ⚠️ 这是"提出请求"，不是"保证命中"：最终是否成为难例由真值判定（common/truth_schema.py）决定，
#   渲染后跑 hardcase_export.py 才知道真实命中率。命中率低说明参数区间还没压到难点上，回头调区间。
import argparse
import itertools
import json
import os
import random
import sys

# 难例类型 → 参数区间。字段含义：
#   distance_m     目标距离（米），batch_capture 会按它摆放目标
#   target_scale   目标尺寸倍率（1.0 = 标称尺寸）
#   occlusion      期望遮挡（由引擎摆放决定，这里只记录意图，真值以像素/框计算为准）
#   light          期望光照工况
#   wear           磨损等级（仅标识/车道线）
#   note           说明
HARDCASE_KINDS = {
    "far_small": {
        "desc": "远距离小目标：80~160m 的标识/锥桶，框通常小于 64px",
        "distance_m": (80.0, 160.0),
        "target_scale": (0.8, 1.0),
        "light": ["front_light", "dusk"],
        "wear": 0.0,
    },
    "occlusion": {
        "desc": "遮挡：用更近的前景物体挡住目标 40%~80%",
        "distance_m": (15.0, 50.0),
        "target_scale": (1.0, 1.0),
        "occlusion": (0.4, 0.8),
        "light": ["front_light", "backlight"],
        "wear": 0.0,
    },
    "backlight": {
        "desc": "逆光：太阳在目标背后，目标呈暗剪影",
        "distance_m": (20.0, 80.0),
        "target_scale": (1.0, 1.0),
        "light": ["back_light"],
        "wear": 0.0,
    },
    "night": {
        "desc": "夜间低照度：路灯/车灯照明，信噪比低",
        "distance_m": (15.0, 70.0),
        "target_scale": (1.0, 1.0),
        "light": ["night"],
        "wear": 0.0,
    },
    "rain": {
        "desc": "雨天：积水反光、能见度下降",
        "distance_m": (15.0, 70.0),
        "target_scale": (1.0, 1.0),
        "light": ["rain", "dusk"],
        "wear": 0.0,
    },
    "wear": {
        "desc": "磨损残缺：标识/车道线褪色破损",
        "distance_m": (10.0, 60.0),
        "target_scale": (1.0, 1.0),
        "light": ["front_light", "dusk"],
        "wear": (0.5, 0.95),
    },
    "tiny": {
        "desc": "极小目标：近距离小物体，框面积 < 32×32 像素",
        "distance_m": (8.0, 25.0),
        "target_scale": (0.25, 0.5),
        "light": ["front_light"],
        "wear": 0.0,
    },
    "edge": {
        "desc": "画面边缘截断：目标中心靠近画面边缘，框被裁",
        "distance_m": (10.0, 40.0),
        "target_scale": (1.0, 1.0),
        "light": ["front_light"],
        "wear": 0.0,
        "edge_bias": (0.75, 0.98),   # 侧向偏移占半视野的比例
    },
}


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else list(d.values())


def _pick_light(lights, allowed):
    """从配置的光照工况里挑出符合难例要求的那个（按 condition 匹配）。"""
    want = set(allowed)
    hits = [l for l in lights if isinstance(l, dict) and str(l.get("condition")) in want]
    return hits or lights


def _uniform(rng, spec, fallback):
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return rng.uniform(spec[0], spec[1])
    return fallback


def make_row(rng, tid, kind, spec, scenes, lights, cameras, prefix):
    """生成一条任务行：难例维度按区间采样，其余维度轮转。"""
    scene = scenes[rng.randrange(len(scenes))]
    light_pool = _pick_light(lights, spec.get("light") or [])
    light = light_pool[rng.randrange(len(light_pool))]
    camera = cameras[rng.randrange(len(cameras))]

    dist = _uniform(rng, spec.get("distance_m"), 30.0)
    scale = _uniform(rng, spec.get("target_scale"), 1.0)
    wear = _uniform(rng, spec.get("wear"), 0.0)
    occlusion = _uniform(rng, spec.get("occlusion"), 0.0)

    # 侧向偏移：edge 难例把目标推到画面边缘，其余在画面内随机
    if spec.get("edge_bias"):
        side = rng.choice([-1.0, 1.0])
        lateral = side * dist * (rng.uniform(*spec["edge_bias"]))
    else:
        lateral = rng.uniform(-0.25, 0.25) * dist

    # 目标：用一个"难例靶标" + 可选的前景物（制造遮挡）
    targets = [{
        "class": "traffic_sign",
        "actor_class": "/Game/Props/BP_TrafficSign.BP_TrafficSign_C",
        "location": [round(dist, 3), round(lateral, 3), round(2.2 * scale, 3)],
        "rotation": [0.0, 180.0, 0.0],
        "size_scale": round(scale, 4),
        "wear_level": round(wear, 4),
        "jitter": [0.5, 0.5, 0.2, 6.0],
    }]
    if occlusion > 0.0:
        # 在相机与目标之间放一个前景遮挡物（距离目标的 40%~70% 处）
        fg_dist = dist * rng.uniform(0.4, 0.7)
        targets.insert(0, {
            "class": "vegetation",
            "actor_class": "/Game/Props/BP_Bush.BP_Bush_C",
            "location": [round(fg_dist, 3), round(lateral * 0.9, 3), 0.0],
            "rotation": [0.0, rng.uniform(0.0, 360.0), 0.0],
            "size_scale": round(rng.uniform(1.0, 2.0), 4),
            "jitter": [0.3, 0.3, 0.0, 15.0],
        })

    return {
        "id": tid,
        "frame_id": tid,
        "seed": rng.randrange(1, 2 ** 31),
        "map": scene if isinstance(scene, str) else scene.get("map"),
        "lighting": light,
        "targets": targets,
        "camera": camera,
        # 额外字段：真值侧可以直接读，便于统计与复现（batch_capture 会忽略不认识的键）
        "_hardcase": {
            "kind": kind,
            "intent": {"distance_m": round(dist, 3), "lateral_m": round(lateral, 3),
                       "target_scale": round(scale, 4), "wear": round(wear, 4),
                       "occlusion": round(occlusion, 4),
                       "light_condition": light.get("condition") if isinstance(light, dict) else None},
        },
    }


def gen_kind(kind, count, scenes, lights, cameras, seed, prefix, start_index=0):
    """生成某一类难例的 N 个任务行。"""
    spec = HARDCASE_KINDS[kind]
    rng = random.Random(seed * 1000 + sorted(HARDCASE_KINDS).index(kind))
    rows = []
    for i in range(count):
        tid = "%s_%s_%04d" % (prefix, kind, start_index + i)
        rows.append(make_row(rng, tid, kind, spec, scenes, lights, cameras, prefix))
    return rows


def load_custom(path, scenes, cameras):
    """自定义场景：直接读一个 JSON，覆盖默认生成规则。

    支持两种写法：
      1) 单条场景描述：{"name": "...", "light_condition": "backlight", "distance_m": [30,60],
                        "targets": [...], "count": 10}
      2) 场景列表：[{...}, {...}]，每个元素一个场景，各自生成 count 条
    这给了"我自己需要的场景"一个直接的入口：不用改代码，写 JSON 就能生成。
    """
    # ⚠️ 不能用 load_json：它会把 dict 展览成 values() 列表（那是给 scenes/lights 这类「字典即列表」的配置用的），
    #    自定义场景是单个对象，拆开会变成字符串列表。
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise SystemExit("自定义场景 JSON 必须是对象或对象数组：%s" % path)
    rows = []
    for ci, case in enumerate(data):
        name = str(case.get("name") or ("custom%d" % ci))
        count = int(case.get("count") or 10)
        seed = int(case.get("seed") or 20261113)
        rng = random.Random(seed * 7919 + ci)
        for i in range(count):
            tid = "custom_%s_%04d" % (name, i)
            scene = case.get("map") or scenes[rng.randrange(len(scenes))]
            light = case.get("lighting")
            if not light:
                cond = case.get("light_condition")
                if cond:
                    pool = [l for l in lights_global if isinstance(l, dict) and l.get("condition") == cond]
                    light = pool[0] if pool else {"condition": cond}
                else:
                    light = lights_global[rng.randrange(len(lights_global))]
            cam = case.get("camera") or cameras[rng.randrange(len(cameras))]
            targets = case.get("targets") or []
            # 自定义 targets 里若给了 distance_m，转成 location 的 X
            norm_targets = []
            for t in targets:
                t2 = dict(t)
                if "distance_m" in t2:
                    t2.setdefault("location", [float(t2.pop("distance_m")), 0.0, 1.5])
                else:
                    t2.pop("distance_m", None)
                norm_targets.append(t2)
            rows.append({
                "id": tid, "frame_id": tid, "seed": rng.randrange(1, 2 ** 31),
                "map": scene if isinstance(scene, str) else scene.get("map"),
                "lighting": light, "targets": norm_targets, "camera": cam,
                "_custom": {"name": name, "index": i},
            })
    return rows


lights_global = []


def main():
    global lights_global
    ap = argparse.ArgumentParser(description="难例定向批次生成")
    ap.add_argument("--kind", default="all",
                    help="难例类型：%s 或 all" % "/".join(sorted(HARDCASE_KINDS)))
    ap.add_argument("--count", type=int, default=30, help="每种难例生成多少帧（all 时为每类各 count 帧）")
    ap.add_argument("--custom", default="", help="自定义场景 JSON（给了就用它，忽略 --kind）")
    ap.add_argument("--scenes", default="", help="场景配置，留空用 config/scenes.json")
    ap.add_argument("--lights", default="", help="光照配置，留空用 config/lights.json")
    ap.add_argument("--cameras", default="", help="相机配置，留空用 config/cameras.json")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--prefix", default="hc")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = os.path.join(root, "config")
    scenes = load_json(a.scenes or os.path.join(cfg, "scenes.json"))
    lights_global = load_json(a.lights or os.path.join(cfg, "lights.json"))
    cameras = load_json(a.cameras or os.path.join(cfg, "cameras.json"))

    if a.custom:
        rows = load_custom(a.custom, scenes, cameras)
    elif a.kind == "all":
        rows = []
        for ci, kind in enumerate(sorted(HARDCASE_KINDS)):
            rows.extend(gen_kind(kind, a.count, scenes, lights_global, cameras, a.seed, a.prefix,
                                 start_index=ci * 10000))
    else:
        if a.kind not in HARDCASE_KINDS:
            print("未知难例类型 %r，可选：%s" % (a.kind, sorted(HARDCASE_KINDS)), file=sys.stderr)
            return 2
        rows = gen_kind(a.kind, a.count, scenes, lights_global, cameras, a.seed, a.prefix)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 统计各类难例的条数，方便核对
    stat = {}
    for r in rows:
        k = (r.get("_hardcase") or {}).get("kind") or ("custom:" + (r.get("_custom") or {}).get("name", "?"))
        stat[k] = stat.get(k, 0) + 1
    print("已写 %s：%d 条任务" % (a.out, len(rows)))
    for k in sorted(stat):
        print("  %-12s %d 帧" % (k, stat[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
