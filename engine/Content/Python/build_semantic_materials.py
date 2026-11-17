# build_semantic_materials.py — 按编码表生成语义材质并落盘
#
# 干什么：为每个类别建一个 Unlit 材质，Emissive = ENCODE_INPUT[class]/255。
#   为什么不是 class/255：实测 Emissive 输出被 gamma+曝光抬高，低值区不可用
#   （1/255 实测输出 13，无法还原成 1）。编码表把类别铺到 32/64/96… 档位，间隔 32。
#   解码见 semantic_encode.py 的 decode_value。
#
# 怎么跑（服务器）：
#   xvfb-run -a UnrealEditor-Cmd <uproject> -ExecutePythonScript=<本文件> -NullRHI -unattended
# 产物：/Game/Validation/Semantic3/M_Sem3_<class>_<name>
import json
import os
import sys

sys.path.insert(0, "/data/avsynth/Content/Python")
import unreal
import semantic_encode as se

MAT_DIR = "/Game/Validation/Semantic3"

report = {}
tools = unreal.AssetToolsHelpers.get_asset_tools()
lib = unreal.MaterialEditingLibrary

for cls, name in se.SEMANTIC_CLASSES.items():
    gray = se.material_gray(cls)
    asset_name = "M_Sem3_%02d_%s" % (cls, name)
    path = MAT_DIR + "/" + asset_name

    existed = unreal.EditorAssetLibrary.does_asset_exist(path)
    if existed:
        mat = unreal.load_asset(path)
        report[cls] = "reuse gray=%.4f" % gray
        continue

    mat = tools.create_asset(asset_name, MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
    mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    const = lib.create_material_expression(mat, unreal.MaterialExpressionConstant, -300, 0)
    const.set_editor_property("r", gray)
    ok = lib.connect_material_property(const, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    lib.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    report[cls] = "created gray=%.4f connected=%s" % (gray, ok)

out = {"materials": report, "dir": MAT_DIR}
with open("/tmp/build_sem.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("DONE")
