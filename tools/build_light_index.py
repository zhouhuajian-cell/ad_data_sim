"""LIGHTS-INDEX: 扫描 50 个 canonical glb，提取灯组材质，写索引。"""
import glob
import json
import os

OUT = "/data/assets_global/OBJ_3D/light_index.json"

# KW-FALSE-POSITIVE-FIX: 去掉 amber（lambert 误命中），加排除词
KW = {
    "head": ("headlight", "head_light", "front_light", "high_beam",
             "low_beam", "drl", "fog_light", "foglight", "fog lights",
             "head_lamp", "headlamp", "frontlamp", "headlights",
             "front_lights", "ext_front_light"),
    "tail": ("taillight", "tail_light", "tail_lamp", "taillamp",
             "tailights", "tail lights", "rear_light", "rear_lamp",
             "rearlamp", "brakelight", "brake_light", "brake lights",
             "stop_lamp", "stoplight"),
    "turn": ("turn_signal", "turn signal", "turnlight", "indicator",
             "blinker", "hazard", "signal_light", "signallights",
             "direction_light"),
}

# 排除：这些不是灯（车身/刹车盘/标牌/内饰等）
EXCLUDE = ("brake_disc", "brake rotor", "brakerotor", "brake_caliper",
           "brakepads", "brake pads", "brakerotor", "lambert",
           "body_", "carpaint", "paint", "chrome", "logo", "seat",
           "interior", "indoor", "carbon")

# 亮色关键字（用于粗略分类：红=尾灯，白/暖黄=前灯）
def classify(name):
    low = (name or "").lower()
    # KW-FALSE-POSITIVE-FIX: 先排除非灯材质
    if any(x in low for x in EXCLUDE):
        return None
    hits = []
    for grp, kws in KW.items():
        if any(k in low for k in kws):
            hits.append(grp)
    if not hits:
        return None
    # 转向优先（turn 常含 light 字样）
    if "turn" in hits:
        return "turn"
    if "tail" in hits:
        return "tail"
    if "head" in hits:
        return "head"
    return hits[0]


def main():
    import trimesh
    fs = sorted(glob.glob("/data/assets_global/OBJ_3D/canonical/*.glb"))
    idx = {}
    ok = 0
    for f in fs:
        uid = os.path.splitext(os.path.basename(f))[0]
        try:
            scene = trimesh.load(f, force="scene")
        except Exception as e:
            idx[uid] = {"error": str(e)[:120], "groups": {}, "n_materials": 0}
            continue
        groups = {"head": [], "tail": [], "turn": []}
        names = set()
        tri = 0
        for gname, geom in scene.geometry.items():
            faces = getattr(geom, "faces", None)
            if faces is not None:
                tri += int(len(faces))
            cands = []
            if gname:
                cands.append(gname)
            vis = geom.visual
            if hasattr(vis, "material") and vis.material is not None:
                mn = getattr(vis.material, "name", None)
                if isinstance(mn, str) and mn:
                    cands.append(mn)
            for c in cands:
                if c in names:
                    continue
                names.add(c)
                g = classify(c)
                if g:
                    groups[g].append(c)
        n_light = sum(len(v) for v in groups.values())
        idx[uid] = {
            "n_materials": len(names),
            "n_triangles": tri,
            "n_light_mats": n_light,
            "supported": n_light >= 2,
            "groups": groups,
        }
        if n_light >= 2:
            ok += 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "total": len(idx),
                   "supported": ok, "assets": idx}, fh,
                  ensure_ascii=False, indent=1)
    print("写入 %s" % OUT)
    print("  模型 %d 个，可控灯 %d 个" % (len(idx), ok))

    # 打印明细
    for uid, d in sorted(idx.items()):
        if d.get("supported"):
            g = d["groups"]
            print("  ★ %-34s head=%d tail=%d turn=%d (材质 %d)"
                  % (uid, len(g["head"]), len(g["tail"]),
                     len(g["turn"]), d["n_materials"]))
            for grp in ("head", "tail", "turn"):
                if g[grp]:
                    print("       %-4s: %s" % (grp, ", ".join(g[grp][:6])))


if __name__ == "__main__":
    main()
