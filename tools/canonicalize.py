#!/usr/bin/env python3
"""按 canonical_manifest.csv 规范化 50 个精品车辆资产并入库。

流程（每个模型）：
  1. 导入 GLB，应用父子变换
  2. 合并所有 MESH 计算真实 bbox
  3. 绕 Z 旋转 yaw_fix_deg，使车头对齐 +X
  4. 按 true_class 等比例缩放到标准车长
  5. 平移到中心 (0,0)，底面 min_z = 0（硬断言）
  6. 导出 canonical/<uid>.glb
  7. 写 asset_index_obj3d.json

用法:
    blender --background --python canonicalize.py
"""
import bpy
import csv
import glob
import json
import math
import os
import sys
from mathutils import Vector, Matrix

MANIFEST = "/data/assets_global/OBJ_3D/canonical_manifest.csv"
# SRC-SELECTED: 模型已重组到 SELECTED/<class>/<name>/
SRC_ROOT = "/data/assets_global/OBJ_3D/SELECTED"
OUT_DIR = "/data/assets_global/OBJ_3D/canonical"
INDEX = "/data/assets_global/OBJ_3D/asset_index_obj3d.json"

# 标准车长（米）
TARGET_LEN = {
    "car": 4.6, "suv": 4.8, "pickup": 5.4, "van": 5.1,
}


def log(s):
    print("[CANON] %s" % s)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    # 应用所有变换
    bpy.ops.object.select_all(action="SELECT")
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True,
                                       scale=True)
    except Exception:
        pass
    # 解除父子
    for o in list(bpy.context.scene.objects):
        if o.type == "MESH" and o.parent:
            o.matrix_world = o.matrix_world.copy()
            o.parent = None
    bpy.context.view_layer.update()
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def world_bbox(objs):
    mn = Vector((1e18,) * 3)
    mx = Vector((-1e18,) * 3)
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    return mn, mx


def merge_meshes(objs):
    """把所有 mesh 合并成一个对象（保留材质）。"""
    if not objs:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        try:
            bpy.ops.object.join()
        except Exception as e:
            log("  join fail: %s" % str(e)[:50])
    return bpy.context.view_layer.objects.active


def find_source(uid):
    """SRC-SELECTED: 在 SELECTED/<class>/<idx>_<cat>_<hash>/mesh.glb 里找。"""
    idx = uid.split("_")[0]
    hits = glob.glob(os.path.join(SRC_ROOT, "*", "%s_*" % idx))
    for c in hits:
        g = os.path.join(c, "mesh.glb")
        if os.path.isfile(g):
            return g
    return None


def canon_one(uid, true_class, yaw_fix):
    src = find_source(uid)
    if not src:
        return None, "no_source"

    clear_scene()
    try:
        objs = import_glb(src)
    except Exception as e:
        return None, "import:%s" % str(e)[:60]
    if not objs:
        return None, "no_mesh"

    ob = merge_meshes(objs)
    if ob is None:
        return None, "merge_fail"

    # LYING-FIX: 若 z 远小于水平轴，说明模型是躺平的，先绕 X 轴立起来
    ob.rotation_mode = "XYZ"
    bpy.context.view_layer.update()
    _mn, _mx = world_bbox([ob])
    _e = _mx - _mn
    # LYING-FIX2: 只有 z 是【最短轴】才判为躺平
    #   正常车: z(高1.2) 是中等轴   x(4.6) > z(1.2) > y(1.9)? 不对
    #   实际: 车长 > 车宽 > 车高，z 应是最短，但要排除"真扁平"
    #   判据: z < 0.30 * 第二长轴 —— 只有极端扁平才立起来
    _sorted = sorted([_e.x, _e.y, _e.z], reverse=True)
    _second = _sorted[1] if len(_sorted) > 1 else _sorted[0]
    if _e.z < _second * 0.30:
        ob.rotation_euler.x += math.radians(90)
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=True, rotation=True,
                                       scale=True)
        _mn, _mx = world_bbox([ob])
        _e = _mx - _mn
        log("  LYING -> 立起 (%.2f,%.2f,%.2f)" % (_e.x, _e.y, _e.z))

    # 1) 绕 Z 旋转
    ob.rotation_euler.z += math.radians(yaw_fix)
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True,
                                   scale=True)
    bpy.context.view_layer.update()

    # 2) 计算 bbox
    mn, mx = world_bbox([ob])
    ext = mx - mn
    # 判断车长轴：最长水平轴
    if ext.y > ext.x:
        # 未对齐 +X，再转 90 度
        ob.rotation_euler.z += math.radians(90)
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=True, rotation=True,
                                       scale=True)
        mn, mx = world_bbox([ob])
        ext = mx - mn

    # WIDTH-GUARD: 车宽异常（> 2.7m 或 < 1.4m）说明横置了，再转 90 度
    if ext.y > 2.7 or ext.y < 1.4:
        for _ in range(2):
            ob.rotation_euler.z += math.radians(90)
            bpy.context.view_layer.update()
            bpy.ops.object.transform_apply(location=True, rotation=True,
                                           scale=True)
            mn, mx = world_bbox([ob])
            ext = mx - mn
            if 1.4 <= ext.y <= 2.7 and ext.x > ext.y:
                break

    # 3) 缩放到标准车长
    target = TARGET_LEN.get(true_class, 4.6)
    L = max(ext.x, 0.01)
    s = target / L
    ob.scale = (s, s, s)
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(scale=True)
    mn, mx = world_bbox([ob])
    ext = mx - mn

    # 4) 中心对齐 (0,0)，底面 z=0
    cx = (mn.x + mx.x) / 2
    cy = (mn.y + mx.y) / 2
    ob.location.x -= cx
    ob.location.y -= cy
    ob.location.z -= mn.z
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True)

    mn2, mx2 = world_bbox([ob])
    # 硬断言
    if abs(mn2.z) >= 1e-3:
        return None, "minz=%.5f" % mn2.z

    L2 = mx2.x - mn2.x
    W2 = mx2.y - mn2.y
    H2 = mx2.z - mn2.z

    # 5) 导出
    os.makedirs(OUT_DIR, exist_ok=True)
    out_fp = os.path.join(OUT_DIR, "%s.glb" % uid)
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.export_scene.gltf(
        filepath=out_fp, export_format="GLB",
        use_selection=True, export_apply=True)
    return {
        "uid": uid, "true_class": true_class,
        "yaw_fix_deg": yaw_fix,
        "lwh": [round(L2, 4), round(W2, 4), round(H2, 4)],
        "path": out_fp,
        "src": src,
    }, "ok"


def main():
    rows = []
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("keep", "")).strip() != "1":
                continue
            rows.append(r)
    log("保留 %d 个模型" % len(rows))

    os.makedirs(OUT_DIR, exist_ok=True)
    index = []
    n_ok = 0
    for i, r in enumerate(rows):
        uid = r["uid"].strip()
        cls = r["true_class"].strip()
        yaw = float(r["yaw_fix_deg"])
        rec, status = canon_one(uid, cls, yaw)
        if rec:
            index.append(rec)
            n_ok += 1
            log("%02d %-22s %-7s lwh=%s" % (
                i + 1, uid, cls, rec["lwh"]))
        else:
            log("%02d %-22s FAIL %s" % (i + 1, uid, status))

    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump({"total": len(index), "assets": index},
                  f, ensure_ascii=False, indent=1)

    log("=== 完成 %d/%d -> %s ===" % (n_ok, len(rows), INDEX))


main()
