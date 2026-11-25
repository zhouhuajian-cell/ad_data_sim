"""vehicle_lights.py — 车灯组驱动（Blender 侧）

用法（在 render_canonical_task.py 里）：
    from vehicle_lights import load_light_index, apply_lights
    apply_lights(objs, uid, frame_idx, fps, light_cfg)

light_cfg 字段（由 render_obj_task.py 从 config.json 透传）：
    enabled     : bool
    brightness  : 0..1
    blink_hz    : Hz
    mode        : steady | blink
    target      : all | head | tail | turn
"""
import json
import math
import os

LIGHT_INDEX = "/data/assets_global/OBJ_3D/light_index.json"

_INDEX_CACHE = None

# 各灯组默认颜色与最大自发光强度
GROUP_STYLE = {
    "head": {"color": (1.00, 0.96, 0.88), "emissive": 18.0},
    "tail": {"color": (0.95, 0.08, 0.05), "emissive": 12.0},
    "turn": {"color": (1.00, 0.55, 0.05), "emissive": 14.0},
}


def load_light_index(path=None):
    """加载灯组索引（带缓存）"""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    p = path or LIGHT_INDEX
    try:
        with open(p, encoding="utf-8") as fh:
            _INDEX_CACHE = json.load(fh)
    except Exception:
        _INDEX_CACHE = {"assets": {}}
    return _INDEX_CACHE


def get_asset_groups(uid, path=None):
    """返回该模型的 {head:[], tail:[], turn:[]}"""
    idx = load_light_index(path)
    a = (idx.get("assets") or {}).get(uid) or {}
    g = a.get("groups") or {}
    return {
        "head": list(g.get("head") or []),
        "tail": list(g.get("tail") or []),
        "turn": list(g.get("turn") or []),
    }


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_factor(t_sec, cfg):
    """返回当前时刻 [0,1] 的亮度系数"""
    if not cfg.get("enabled", True):
        return 0.0
    b = _clamp(float(cfg.get("brightness", 1.0)), 0.0, 1.0)
    hz = max(0.0, float(cfg.get("blink_hz", 0.0)))
    mode = str(cfg.get("mode", "steady"))
    if mode == "steady" or hz <= 1e-6:
        return b
    duty = _clamp(float(cfg.get("duty", 0.5)), 0.05, 0.95)
    phase = (t_sec * hz) % 1.0
    return b if phase < duty else 0.0


def frame_group_factors(frame_idx, fps, cfg):
    """计算该帧三个灯组各自的亮度系数"""
    fps = float(fps) if fps and fps > 0 else 10.0
    t = float(frame_idx) / fps
    base = compute_factor(t, cfg)
    target = str(cfg.get("target", "all"))
    out = {}
    for grp in ("head", "tail", "turn"):
        active = cfg.get("enabled", True) and (
            target == "all" or target == grp)
        out[grp] = base if active else 0.0
    return out


def apply_lights(objs, uid, frame_idx, fps, cfg, log=None):
    """把亮度写进 Blender 材质的 Emission

    返回 {"head": n, "tail": n, "turn": n} 表示各灯组实际改了几个材质
    """
    if not cfg or not cfg.get("enabled", True):
        return {}
    groups = get_asset_groups(uid)
    if not any(groups.values()):
        return {}

    factors = frame_group_factors(frame_idx, fps, cfg)

    # 材质名 -> 组
    name2grp = {}
    for grp, names in groups.items():
        for n in names:
            name2grp[n.lower()] = grp

    # 遍历所有对象上的材质
    touched = {"head": 0, "tail": 0, "turn": 0}
    seen = set()
    all_names = []
    for o in objs:
        mats = []
        try:
            mats = list(o.data.materials or [])
        except Exception:
            continue
        for _m in mats:
            if _m is not None:
                all_names.append(_m.name)
        for m in mats:
            if m is None:
                continue
            key = (m.name or "").lower()
            if key in seen:
                continue
            grp = name2grp.get(key)
            if not grp:
                continue
            seen.add(key)
            _set_emission(m, grp, factors[grp])
            touched[grp] += 1

    if log and not any(touched.values()):
        # LIGHT-DEBUG: 没匹配上时打印实际材质名
        log("  lights 未匹配! 模型 %s 有 %d 个材质, 索引要 %s" % (
            uid, len(all_names), list(name2grp.keys())[:6]))
        log("    实际材质: %s" % all_names[:12])
    if log:
        log("  lights f%-3d %s  factor head=%.2f tail=%.2f turn=%.2f  "
            "mats %d/%d/%d" % (
                frame_idx,
                uid.split("_")[0],
                factors["head"], factors["tail"], factors["turn"],
                touched["head"], touched["tail"], touched["turn"]))
    return touched


def _set_emission(mat, grp, factor):
    """设置 Principled BSDF 的 Emission Color / Strength"""
    if not mat.use_nodes or not mat.node_tree:
        return
    style = GROUP_STYLE.get(grp) or GROUP_STYLE["head"]
    r, g, b = style["color"]
    strength = style["emissive"] * float(factor)

    bsdf = None
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            bsdf = n
            break

    if bsdf is not None:
        # 4.x: Emission Color / Emission Strength
        for cname in ("Emission Color", "Emission"):
            if cname in bsdf.inputs and not bsdf.inputs[cname].is_linked:
                try:
                    bsdf.inputs[cname].default_value = (r, g, b, 1.0)
                except Exception:
                    pass
                break
        if "Emission Strength" in bsdf.inputs and \
                not bsdf.inputs["Emission Strength"].is_linked:
            bsdf.inputs["Emission Strength"].default_value = strength
    else:
        # 无 Principled：找任意 Emission 节点
        for n in mat.node_tree.nodes:
            if n.type == "EMISSION":
                try:
                    n.inputs["Color"].default_value = (r, g, b, 1.0)
                    n.inputs["Strength"].default_value = strength
                except Exception:
                    pass
                break

    # 关灯时确保不透明，避免玻璃材质发虚
    if factor > 0.01:
        try:
            mat.blend_method = "OPAQUE"
        except Exception:
            pass


def config_from_task(params):
    """从任务 params 提取车灯配置，含轨迹自动推导

    params 里可能有：
      lights: {enabled, brightness, blink_hz, mode, target}
      trajectory_mode: straight | turn | right
    """
    lp = params.get("lights") or {}
    cfg = {
        "enabled": bool(lp.get("enabled", False)),
        "brightness": float(lp.get("brightness", 1.0)),
        "blink_hz": float(lp.get("blink_hz", 0.0)),
        "mode": str(lp.get("mode", "steady")),
        "duty": float(lp.get("duty", 0.5)),
        "target": str(lp.get("target", "all")),
    }
    # TRAJ-AUTO-LIGHT: 左转/右转自动开转向灯闪烁
    tm = str(params.get("trajectory_mode") or "")
    if tm in ("turn", "right") and not lp:
        cfg["enabled"] = True
        cfg["target"] = "turn"
        cfg["mode"] = "blink"
        cfg["blink_hz"] = 1.5
        cfg["brightness"] = 1.0
    return cfg
