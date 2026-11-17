# import_texture.py — 把 AssetInbox 里的图片导入成 UE Texture2D，并做成可用贴图材质
#
# 干什么（对应"上传路标图案 → 变成可用资产"）：
#   1) 扫描 /data/avsynth/AssetInbox/ 下的图片；
#   2) 用 AssetImportTask 导入到 /Game/Assets/Textures/（同名已存在则跳过，不覆盖）；
#   3) 为每张贴图生成一个 Unlit 材质 /Game/Assets/Materials/M_Img_<名>，
#      贴图接到 Emissive/BaseColor，便于直接贴到路标牌面上；
#   4) 结果写到 /tmp/import_tex.json，供面板展示"已导入/失败"状态。
#
# 为什么用 AssetImportTask：UnrealEditor-Cmd 无 GUI，只能走 Python 的资产导入 API。
#
# 用法：xvfb-run -a UnrealEditor-Cmd <uproject> -ExecutePythonScript=<本文件> -NullRHI -unattended
import json
import os
import traceback

import unreal

INBOX = "/data/avsynth/AssetInbox"
TEX_DIR = "/Game/Assets/Textures"
MAT_DIR = "/Game/Assets/Materials"
EXTS = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr")

report = {"imported": [], "skipped": [], "failed": [], "materials": []}


def rec(bucket, name, detail=""):
    report[bucket].append({"name": name, "detail": str(detail)[:200]})


try:
    os.makedirs(INBOX, exist_ok=True)
    files = sorted(f for f in os.listdir(INBOX)
                   if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(INBOX, f)))

    for fname in files:
        base = os.path.splitext(fname)[0]
        # ⚠️ 实测 AssetImportTask 的 destination_name 不一定生效（会退回用源文件名），
        #   所以资产名可能是 base 也可能是 T_<base>，两个都探一次。
        asset_path = None
        for cand in (TEX_DIR + "/" + base, TEX_DIR + "/T_" + base):
            if unreal.EditorAssetLibrary.does_asset_exist(cand) or os.path.exists(
                    "/data/avsynth/Content/Assets/Textures/" + cand.rsplit("/", 1)[-1] + ".uasset"):
                asset_path = cand
                break
        if asset_path is None:
            asset_path = TEX_DIR + "/" + base    # 还没导入过，用源文件名做目标

        if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
            rec("skipped", fname, "已存在: " + asset_path)
            continue

        src = os.path.join(INBOX, fname)
        try:
            task = unreal.AssetImportTask()
            task.set_editor_property("filename", src)
            task.set_editor_property("destination_path", TEX_DIR)
            task.set_editor_property("destination_name", base)
            task.set_editor_property("replace_existing", False)
            task.set_editor_property("automated", True)
            task.set_editor_property("save", True)
            unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

            # 注意：import_asset_tasks 是异步落盘的，刚导完立刻 load_asset 可能拿不到对象
            # （实测 asset 文件已生成但 load 返回 None）。这里以「资产文件是否存在」为准，
            # 再尝试加载一次拿真实对象。
            tex = unreal.load_asset(asset_path)
            if tex is None:
                exists = unreal.EditorAssetLibrary.does_asset_exist(asset_path)
                rec("imported" if exists else "failed", fname,
                    asset_path if exists else ("导入后资产不存在 " + asset_path))
                if not exists:
                    continue
                # 文件在但对象拿不到：材质那步会跳过，这里显式记一笔
                unreal.EditorAssetLibrary.save_directory(TEX_DIR, False, True, False)
                tex = unreal.load_asset(asset_path)
            if tex is None:
                rec("imported", fname, asset_path + "（贴图对象待下次会话可加载）")
                continue
            rec("imported", fname, asset_path)

            # 建一个把该贴图贴到 BaseColor 的 Unlit 材质，供路标牌面直接使用
            mat_path = MAT_DIR + "/M_Img_" + base
            if not unreal.EditorAssetLibrary.does_asset_exist(mat_path):
                tools = unreal.AssetToolsHelpers.get_asset_tools()
                lib = unreal.MaterialEditingLibrary
                mat = tools.create_asset("M_Img_" + base, MAT_DIR, unreal.Material,
                                         unreal.MaterialFactoryNew())
                mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
                mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
                samp = lib.create_material_expression(mat, unreal.MaterialExpressionTextureSample, -400, 0)
                samp.set_editor_property("texture", tex)
                lib.connect_material_property(samp, "RGB", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
                lib.recompile_material(mat)
                unreal.EditorAssetLibrary.save_loaded_asset(mat)
                rec("materials", fname, mat_path)
            else:
                rec("materials", fname, "已存在 " + mat_path)
        except Exception as e:
            rec("failed", fname, e)

except Exception:
    report["fatal"] = traceback.format_exc()[:1200]

# UE 的 Python 默认 locale 是 ascii，写中文必须显式指定 encoding，否则 UnicodeEncodeError`nwith open("/tmp/import_tex.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("DONE")
