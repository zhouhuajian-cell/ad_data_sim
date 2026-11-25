#!/usr/bin/env python3
"""通用外部 3D 资产规范化入库入口。

用法（在服务器上用 blender 跑）：
    blender --background --python ingest_external.py -- \
        --src /data/assets_global/OBJ_3D/raw_external \
        --out-glb /data/assets_global/OBJ_3D/canonical \
        --index /data/assets_global/OBJ_3D/asset_index_obj3d.json \
        --target-len '{"car":4.6,"suv":4.8,"pickup":5.4,"van":5.1,"truck":8.5}'

输入目录结构（raw_external/ 下）：
    raw_external/
      <uid>/                     例如 60_truck_abc123
        mesh.glb                 模型（必需）
        meta.json                {"true_class":"truck", "yaw_fix_deg":0}  （可选）
      <uid>.glb                  或直接放 flat glb（必需配套 <uid>.meta.json）

处理流程：
    ① 导入 GLB
    ② 按 yaw_fix_deg 旋转（把车头/正面转到 +X）
    ③ 按 true_class 的长度缩放到真实米制尺寸（等比）
    ④ 居中到 (0,0)，强制 min_z = 0（接地）
    ⑤ 导出到 canonical/<uid>.glb
    ⑥ 写入 asset_index_obj3d.json

不做的事（需人工先做）：
    - 朝向判定：先跑 board_render.py 出看板，人工填 yaw_fix_deg 和 keep
    - 质量筛选：keep=0 的不入库
"""
import glob
import json
import math
import os
import sys

import bpy
from mathutils import Vector


# ============================================================
# 类目标准长度（米）
# ============================================================
DEFAULT_TARGET_LEN = {
    "car": 4.6,
    "suv": 4.8,
    "pickup": 5.4,
    "van": 5.1,
    "truck": 8.5,          # 中型货车默认；重卡填 16.5
    "bus": 12.0,
    "tricycle": 2.6,       # 电动三轮
    "motorcycle": 2.1,
    "bicycle": 1.8,
    "rickshaw": 2.8,       # 人力/机动三轮（异行车）
    "low_speed_ev": 2.9,   # 老头乐
    "scooter": 1.2,        # 电动滑板车
    "skateboard": 0.8,
    "pedestrian": 1.7,
    "rider": 1.8,          # 骑车人（含车）
    "child": 1.2,          # 孩子（你要求 1.5m 左右可传 1.5）
    "camel": 2.2,          # 骆驼（肩高）
    "kangaroo": 1.6,       # 袋鼠（站高）
    "animal": 1.5,         # 其他动物兜底
    "cone": 0.7,
    "barrier": 1.0,
}


def log(s):
    print("[INGEST] %s" % s)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    """导入 GLB，返回 (对象列表, 合并后的 bbox)。"""
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=path)
    new = [bpy.data.objects[k] for k in bpy.data.objects.keys()
           if k not in before and bpy.data.objects[k].type == "MESH"]
    if not new:
        return None, None

    # PARENT-FIRST-FIX: 必须先解父级保留世界矩阵，再 apply
    #   否则父级的缩放不会被计入，导致尺寸算错
    bpy.ops.object.select_all(action="DESELECT")
    for o in new:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]

    # ① 先解父级，保留世界变换
    for o in new:
        if o.parent:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
    bpy.context.view_layer.update()

    # ② 再把世界变换烘进几何
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True,
                                       scale=True)
    except Exception:
        pass
    bpy.context.view_layer.update()
    return new, None


def world_bbox(objs):
    mn = Vector((1e18, 1e18, 1e18))
    mx = Vector((-1e18, -1e18, -1e18))
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    return mn, mx


def rotate(objs, deg, axis="z"):
    """VERTEX-ROT-FIX: 直接旋转顶点数据（最可靠，不依赖 transform_apply）。
    """
    if abs(deg) < 1e-6:
        return
    from mathutils import Matrix
    rot = {"x": Matrix.Rotation(math.radians(deg), 4, "X"),
           "y": Matrix.Rotation(math.radians(deg), 4, "Y"),
           "z": Matrix.Rotation(math.radians(deg), 4, "Z")}[axis]
    for o in objs:
        o.data.transform(rot)
        o.data.update()
    bpy.context.view_layer.update()


def rotate_z(objs, deg):
    rotate(objs, deg, "z")


def auto_upright(objs):
    """Y-up 模型自动摆正：若 Z 轴最短且 Y 轴接近竖直，绕 X 转 -90。"""
    mn, mx = world_bbox(objs)
    ext = mx - mn
    # 车辆"高度"通常是三个尺寸里最小的（车高 < 车宽 < 车长）
    # 若 X 轴是最小（说明 Y 是向上），转 -90 让 Z 向上
    if ext.x < ext.y and ext.x < ext.z and ext.y >= ext.z * 0.9:
        log("自动摆正：检测到 Y-up，绕 X 转 -90")
        rotate(objs, -90, "x")
        return True
    return False


def scale_to_length(objs, target_m):
    """等比缩放到最长水平边 = target_m。"""
    mn, mx = world_bbox(objs)
    ext = mx - mn
    cur = max(ext.x, ext.y)
    if cur < 1e-6:
        return 1.0
    k = target_m / cur
    for o in objs:
        o.scale = (o.scale.x * k, o.scale.y * k, o.scale.z * k)
    bpy.context.view_layer.update()
    return k


def center_and_ground(objs):
    """水平居中到 (0,0)，min_z 归零。"""
    mn, mx = world_bbox(objs)
    cx = (mn.x + mx.x) / 2.0
    cy = (mn.y + mx.y) / 2.0
    for o in objs:
        o.location = (o.location.x - cx,
                      o.location.y - cy,
                      o.location.z - mn.z)
    bpy.context.view_layer.update()


def export_glb(objs, out_fp):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out_fp, export_format="GLB",
        use_selection=True, export_apply=True)


def find_inputs(src):
    """返回 [(uid, glb_path, meta_dict)]"""
    out = []
    if not os.path.isdir(src):
        return out
    for name in sorted(os.listdir(src)):
        p = os.path.join(src, name)
        if os.path.isdir(p):
            glb = os.path.join(p, "mesh.glb")
            if not os.path.isfile(glb):
                cand = glob.glob(os.path.join(p, "*.glb"))
                glb = cand[0] if cand else None
            if not glb:
                continue
            meta = {}
            mf = os.path.join(p, "meta.json")
            if os.path.isfile(mf):
                try:
                    meta = json.load(open(mf, encoding="utf-8"))
                except Exception:
                    meta = {}
            out.append((name, glb, meta))
        elif name.lower().endswith(".glb"):
            uid = os.path.splitext(name)[0]
            meta = {}
            mf = os.path.join(src, uid + ".meta.json")
            if os.path.isfile(mf):
                try:
                    meta = json.load(open(mf, encoding="utf-8"))
                except Exception:
                    meta = {}
            out.append((uid, p, meta))
    return out


def main():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []

    src = "/data/assets_global/OBJ_3D/raw_external"
    out_glb = "/data/assets_global/OBJ_3D/canonical"
    index_fp = "/data/assets_global/OBJ_3D/asset_index_obj3d.json"
    target_len = dict(DEFAULT_TARGET_LEN)
    dry = False

    for i, a in enumerate(args):
        if a == "--src" and i + 1 < len(args):
            src = args[i + 1]
        elif a == "--out-glb" and i + 1 < len(args):
            out_glb = args[i + 1]
        elif a == "--index" and i + 1 < len(args):
            index_fp = args[i + 1]
        elif a == "--target-len" and i + 1 < len(args):
            try:
                target_len.update(json.loads(args[i + 1]))
            except Exception as e:
                log("target-len 解析失败: %s" % e)
        elif a == "--dry":
            dry = True

    inputs = find_inputs(src)
    log("发现 %d 个输入" % len(inputs))
    if not inputs:
        log("raw_external 为空，退出")
        return

    # 读现有索引
    idx = {"assets": []}
    if os.path.isfile(index_fp):
        try:
            idx = json.load(open(index_fp, encoding="utf-8"))
        except Exception:
            idx = {"assets": []}
    existing = {a["uid"]: a for a in idx.get("assets") or []}

    ok, skip = 0, 0
    for uid, glb, meta in inputs:
        true_class = meta.get("true_class") or "car"
        yaw_fix = float(meta.get("yaw_fix_deg", 0.0))
        keep = int(meta.get("keep", 1))
        if not keep:
            log("SKIP %-28s keep=0" % uid)
            skip += 1
            continue

        clear_scene()
        objs, _ = import_glb(glb)
        if not objs:
            log("SKIP %-28s 导入失败" % uid)
            skip += 1
            continue

        # 先自动摆正（Y-up -> Z-up）
        auto_upright(objs)
        # 再按 yaw_fix 绕 Z 转（车头指向 +X）
        rotate_z(objs, yaw_fix)
        tgt = float(target_len.get(true_class, 4.6))
        k = scale_to_length(objs, tgt)
        # HEAD-TO-X-FIX: 车头统一朝 +X（长边在 X 轴）
        mn0, mx0 = world_bbox(objs)
        e0 = mx0 - mn0
        if e0.y > e0.x * 1.1:
            log("  长边在 Y -> 绕 Z 转 90 使车头朝 +X")
            rotate_z(objs, 90)
        center_and_ground(objs)

        mn, mx = world_bbox(objs)
        ext = mx - mn
        fp = os.path.join(out_glb, uid + ".glb")
        if not dry:
            export_glb(objs, fp)
        log("%-28s cls=%-10s 目标%.1fm 缩放%.3f 尺寸=(%.2f, %.2f, %.2f) min_z=%.4f" % (
            uid, true_class, tgt, k, ext.x, ext.y, ext.z, mn.z))

        if not dry:
            existing[uid] = {
                "uid": uid,
                "true_class": true_class,
                "lwh": [round(ext.x, 3), round(ext.y, 3),
                        round(ext.z, 3)],
                "path": fp,
                "yaw_fix_deg": yaw_fix,
                "source": "external",
            }
        ok += 1

    if not dry:
        idx["assets"] = list(existing.values())
        os.makedirs(os.path.dirname(index_fp), exist_ok=True)
        with open(index_fp, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=1)
        log("索引已更新: %s（%d 个资产）" % (index_fp,
                                        len(idx["assets"])))

    log("完成: 入库 %d / 跳过 %d" % (ok, skip))


if __name__ == "__main__":
    main()
