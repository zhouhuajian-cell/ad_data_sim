"""Normalize an external GLB into a reviewable staging output with Blender.

MERGED-VERSION: 以 external_asset_canonicalize_blender.py 为核心，
补入 auto_upright / head-to-x 检测（来自 ingest_external.py）。

用法（程序调用）:
    blender -b --python merge_ingest.py -- \
        --src <input.glb> --out <output.glb> --meta <meta.json> \
        [--uid <uid>] [--class <cls>] [--yaw <deg>] [--length <m>] \
        [--index <asset_index.json>] [--dry]

处理顺序（关键）:
    ① 解父级保留 matrix_world（PARENT-FIRST）
    ② auto_upright：Y-up 模型绕 X 转 -90°
    ③ yaw_fix：绕 Z 转（车头指向 +X）
    ④ head-to-x：长边若仍在 Y，再转 90°
    ⑤ 缩放到 X 轴长度 = target_m（等比）
    ⑥ 居中 + min_z = 0（断言 < 0.001）
    ⑦ 导出 GLB + 写 meta
    ⑧ 可选：写入 asset_index_obj3d.json

不复用 transform_apply，全部走 matrix_world 复合，保留材质与多 mesh。
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector


DEFAULT_TARGET_LEN = {
    "car": 4.6, "suv": 4.8, "pickup": 5.4, "van": 5.1,
    "truck": 8.5, "bus": 12.0,
    "tricycle": 2.6, "motorcycle": 2.1, "bicycle": 1.8,
    "rickshaw": 2.8, "low_speed_ev": 2.9,
    "scooter": 1.2, "skateboard": 0.8,
    "pedestrian": 1.7, "rider": 1.8, "child": 1.2,
    "camel": 2.2, "kangaroo": 1.6, "animal": 1.5,
    "cone": 0.7, "barrier": 1.0,
}


def log(s):
    print("[MERGE-INGEST] %s" % s)


def bounds(meshes):
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    if not pts:
        raise ValueError("model contains no meshes")
    return (Vector([min(p[k] for p in pts) for k in range(3)]),
            Vector([max(p[k] for p in pts) for k in range(3)]))


def apply_world(meshes, M):
    """对每个 mesh 的 matrix_world 统一左乘 M。"""
    for ob in meshes:
        ob.matrix_world = M @ ob.matrix_world
    bpy.context.view_layer.update()


def auto_upright(meshes):
    """Y-up 自动摆正：若 X 轴最矮（说明 Y 是向上），绕 X 转 -90。"""
    lo, hi = bounds(meshes)
    ext = hi - lo
    if ext.x < ext.y and ext.x < ext.z and ext.y >= ext.z * 0.9:
        log("检测到 Y-up，绕 X 转 -90")
        apply_world(meshes, Matrix.Rotation(math.radians(-90), 4, "X"))
        return True
    return False


def head_to_x(meshes):
    """长边若在 Y 轴，绕 Z 转 90 使车头朝 +X。"""
    lo, hi = bounds(meshes)
    ext = hi - lo
    if ext.y > ext.x * 1.1:
        log("长边在 Y -> 绕 Z 转 90 使车头朝 +X")
        apply_world(meshes, Matrix.Rotation(math.radians(90), 4, "Z"))
        return True
    return False


def main():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []

    src = out_fp = meta_fp = index_fp = None
    uid = cls = None
    yaw = 0.0
    target = None
    dry = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--src":
            src = args[i + 1]; i += 2
        elif a == "--out":
            out_fp = args[i + 1]; i += 2
        elif a == "--meta":
            meta_fp = args[i + 1]; i += 2
        elif a == "--uid":
            uid = args[i + 1]; i += 2
        elif a == "--class":
            cls = args[i + 1]; i += 2
        elif a == "--yaw":
            yaw = float(args[i + 1]); i += 2
        elif a == "--length":
            target = float(args[i + 1]); i += 2
        elif a == "--index":
            index_fp = args[i + 1]; i += 2
        elif a == "--dry":
            dry = True; i += 1
        else:
            i += 1

    if not src or not os.path.isfile(src):
        raise ValueError("missing or invalid --src")

    # 兼容旧式位置参数: input output metadata yaw_deg target_length_m
    if out_fp is None and len(args) >= 5 and "=" not in args[0]:
        src, out_fp, meta_fp = args[0], args[1], args[2]
        yaw = float(args[3])
        target = float(args[4])

    if target is None:
        target = DEFAULT_TARGET_LEN.get(cls or "car", 4.6)
    if not math.isfinite(yaw) or not math.isfinite(target) or target <= 0:
        raise ValueError("invalid yaw or target length")

    log("src=%s  目标长=%.2fm  yaw=%.1f" % (src, target, yaw))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    meshes = [x for x in bpy.context.scene.objects if x.type == "MESH"]
    if not meshes:
        raise ValueError("no mesh objects")

    # ① PARENT-FIRST: 解父级但保留世界矩阵（materials 与 multi-mesh 不变）
    for ob in meshes:
        m = ob.matrix_world.copy()
        ob.parent = None
        ob.matrix_world = m
    bpy.context.view_layer.update()

    # ② Y-up 自动摆正
    auto_upright(meshes)

    # ③ 用户指定 yaw（车头指向 +X）
    if abs(yaw) > 1e-6:
        apply_world(meshes, Matrix.Rotation(math.radians(yaw), 4, "Z"))

    # ④ 长边强制落到 X 轴
    head_to_x(meshes)

    # ⑤ 缩放到 X 轴长度
    lo, hi = bounds(meshes)
    length = hi.x - lo.x
    if not math.isfinite(length) or length <= 1e-9:
        raise ValueError("model has invalid X extent")
    scale = target / length
    apply_world(meshes, Matrix.Diagonal(
        Vector((scale, scale, scale, 1.0))))

    # ⑥ 居中 + 接地
    lo, hi = bounds(meshes)
    center = Matrix.Translation((-(lo.x + hi.x) / 2.0,
                                 -(lo.y + hi.y) / 2.0,
                                 -lo.z))
    apply_world(meshes, center)

    lo, hi = bounds(meshes)
    if abs(lo.z) > 0.001:
        raise ValueError("ground alignment failed: min_z=%.6f" % lo.z)

    ext = hi - lo
    log("结果: LWH=(%.3f, %.3f, %.3f) min_z=%.6f 缩放=%.4f" % (
        ext.x, ext.y, ext.z, lo.z, scale))

    # ⑦ 导出（DRY-KEEP-ARTIFACT: dry 也导出 glb，只是不写索引）
    if out_fp:
        os.makedirs(os.path.dirname(os.path.abspath(out_fp)), exist_ok=True)
        bpy.ops.object.select_all(action="DESELECT")
        for ob in meshes:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.export_scene.gltf(
            filepath=out_fp, export_format="GLB",
            use_selection=True, export_apply=True)
        log("导出: %s" % out_fp)

    meta = {
        "source": src, "path": out_fp, "uid": uid,
        "true_class": cls,
        "yaw_fix_deg": yaw,
        "target_length_m": target,
        "scale_factor": round(scale, 6),
        "lwh": [round(float(v), 4) for v in ext],
        "min_z": round(float(lo.z), 6),
        "preview_only": False,
        "training_eligible": False,
    }
    if meta_fp:
        with open(meta_fp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log("meta: %s" % meta_fp)

    # ⑧ 写索引
    if index_fp and uid and not dry:
        idx = {"assets": []}
        if os.path.isfile(index_fp):
            try:
                idx = json.load(open(index_fp, encoding="utf-8"))
            except Exception:
                idx = {"assets": []}
        assets = [a for a in idx.get("assets") or [] if a.get("uid") != uid]
        assets.append({
            "uid": uid,
            "true_class": cls or "car",
            "lwh": [round(float(v), 4) for v in ext],
            "path": out_fp,
            "yaw_fix_deg": yaw,
            "source": "external",
        })
        idx["assets"] = assets
        with open(index_fp, "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=1, ensure_ascii=False)
        log("索引已更新: %d 个资产" % len(assets))


if __name__ == "__main__":
    main()
