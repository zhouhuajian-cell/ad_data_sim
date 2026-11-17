# traffic_sign_template.py — 用「标准模板 + 贴图」生成路标，并摆进指定场景
#
# 干什么（对应"上传路标图案 → 贴到标准牌面 → 放进不同场景"）：
#   1) 标准模板：牌面 = Cube 压扁成薄板，立柱 = 细长方块。
#      不用外部 3D 模型，全用 UE 基础几何体，任何工程都能跑。
#   2) 贴图 = 从 /Game/Assets/Materials/M_Img_<name> 取（由 import_texture.py 生成）。
#   3) 生成一个 StaticMeshActor 组合（牌面 + 立柱），打上类别 stencil（traffic_sign=4），
#      位置/朝向/尺寸由参数控制。
#
# 用法（服务器）：
#   xvfb-run -a UnrealEditor-Cmd <uproject> -ExecutePythonScript=<本文件> -NullRHI -unattended
#   环境变量：
#     AD_SIGN_MAP     目标关卡（默认 /Game/Maps/codex_validation）
#     AD_SIGN_MAT     材质资产路径（默认取 AssetInbox 里第一个已导入的）
#     AD_SIGN_POS     摆放位置 "x,y,z"（默认 2000,300,150）
#     AD_SIGN_YAW     朝向角（默认 180，正对相机）
#     AD_SIGN_SCALE   牌面缩放（默认 1.0）
# 产物：关卡里多出 "TrafficSign_<材质名>" 的 Actor，并保存关卡。
import json
import os
import sys
import traceback

import unreal

report = {"steps": []}


def rec(name, ok, value=""):
    report["steps"].append({"step": name, "ok": ok, "value": str(value)[:300]})
    print("[sign] %s %s %s" % ("OK " if ok else "ERR", name, value))


def pick_material():
    """选一个已导入的图片材质；优先用环境变量指定的。"""
    want = os.environ.get("AD_SIGN_MAT", "")
    if want and unreal.EditorAssetLibrary.does_asset_exist(want):
        return want
    assets = unreal.EditorAssetLibrary.list_assets("/Game/Assets/Materials", False, False)
    for a in assets:
        p = str(a).split(".")[0]
        if "M_Img_" in p:
            return p
    return None


try:
    MAP = os.environ.get("AD_SIGN_MAP", "/Game/Maps/codex_validation")
    POS = [float(x) for x in os.environ.get("AD_SIGN_POS", "2000,300,150").split(",")]
    YAW = float(os.environ.get("AD_SIGN_YAW", "180"))
    SCALE = float(os.environ.get("AD_SIGN_SCALE", "1.0"))

    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    les.load_level(MAP)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    rec("load_level", True, MAP)

    mat_path = pick_material()
    rec("pick_material", mat_path is not None, mat_path or "没有可用图片材质")
    mat = unreal.load_asset(mat_path) if mat_path else None

    cube = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    rec("load_cube", cube is not None)

    # 牌面：薄板。Cube 原始尺寸 100cm（1m 单位立方），scale 是倍数。
    #   尺寸取：厚 5cm、宽 70cm、高 70cm  → scale = (0.05, 0.70, 0.70)
    plate = eas.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(POS[0], POS[1], POS[2]),
                                       unreal.Rotator(0.0, YAW, 0.0))
    plate.set_actor_label("TrafficSign_Plate")
    plate.set_actor_scale3d(unreal.Vector(0.05 * SCALE, 0.7 * SCALE, 0.7 * SCALE))
    pm = plate.get_component_by_class(unreal.StaticMeshComponent)
    pm.set_static_mesh(cube)
    if mat:
        pm.set_material(0, mat)
    # 语义通道靠 stencil：traffic_sign = 4
    try:
        pm.set_editor_property("render_custom_depth", True)
        pm.set_editor_property("custom_depth_stencil_value", 4)
    except Exception as e:
        rec("plate_stencil", False, str(e)[:100])
    rec("plate", True, "位置 %s 朝向 %.0f°" % (POS, YAW))

    # 立柱：细长方体，顶端对齐牌面下沿、底端落到地面。
    #   ⚠️ 单位坑：UE 世界里位置是**厘米**，而 set_actor_scale3d 是**倍数**。
    #      之前把 POS[2]（厘米）当米用，算出 219 米高的柱子（实测 bug）。
    #   这里统一按厘米算：牌面半高 = 35cm*SCALE；立柱高 H_cm = POS[2] - 牌面半高；
    #      立柱中心 z_cm = H_cm/2；立柱 scale_z = H_cm / 100（因为 Cube 高 100cm）。
    plate_half_cm = 35.0 * SCALE                       # 牌面半高（厘米）
    post_h_cm = max(20.0, POS[2] - plate_half_cm)      # 立柱高度（厘米）
    post_z_cm = post_h_cm * 0.5                        # 立柱中心（厘米）
    post = eas.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(POS[0], POS[1], post_z_cm),
                                      unreal.Rotator(0.0, YAW, 0.0))
    post.set_actor_label("TrafficSign_Post")
    # 截面 6cm x 6cm，高度 post_h_cm
    post.set_actor_scale3d(unreal.Vector(0.06 * SCALE, 0.06 * SCALE, post_h_cm / 100.0))
    qm = post.get_component_by_class(unreal.StaticMeshComponent)
    qm.set_static_mesh(cube)
    try:
        qm.set_editor_property("render_custom_depth", True)
        qm.set_editor_property("custom_depth_stencil_value", 4)
    except Exception:
        pass
    rec("post", True, "立柱 高=%.2fm 中心z=%.2fm" % (post_h, post_z))

    les.save_current_level()
    rec("save", True, MAP)

    # 统计结果
    n_sign = sum(1 for a in eas.get_all_level_actors() if "TrafficSign" in a.get_actor_label())
    rec("total_sign_actors", True, "%d 个" % n_sign)
    report["material"] = mat_path
    report["map"] = MAP
    report["actor_count"] = n_sign

except Exception:
    report["fatal"] = traceback.format_exc()[:1500]

with open("/tmp/sign_template.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("DONE")
