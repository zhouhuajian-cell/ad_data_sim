#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modules/asset_library.py — 按国家分层的标牌素材库

设计目标（对应需求「素材库按国家分类，带国家代码」）：
  素材库目录结构：
    素材根/
      BR/speed_limit_60.png         巴西的限速60
      BR/prohibition_no_right.png
      RU/warning_pedestrian.png
      DE/...
      _common/...                   通用（无国家归属时使用）

  选素材时的优先级：
    1) 指定国家代码 → 该国家目录
    2) 找不到 → _common 目录
    3) 再找不到 → 报错（不静默用错国家的素材）

为什么按国家分：
  各国标牌在**同一语义类别下外观不同**（限速60 在中国是红圈白底，在美国是白底黑字方牌）。
  用错国家的素材去贴，等于给模型喂错误的视觉特征。
  所以国家代码必须显式参与选素材，而不是靠文件名猜。

来源说明：
  MTSD（Mapillary Traffic Sign Dataset）在网盘上已按国家拆分：
    /mnt/Data_Platform/zhj_datamining/Oversea_data_test/MTSD_Tsr/mtsd_by_country/<CC>_<中文名>/
  每国有 *.jpg（标牌裁切）+ manifest.jsonl；全量标注在 mtsd_fully/annotations/。
"""
import json
import os

# MTSD 标签的第一级 → 我们的类别名候选（第二级取自 MTSD 的中层名）
# 只做「大类」映射；具体子类由调用方指定，避免猜错
MTSD_TOP_MAP = {
    "speed-limit": "speed_limit",
    "regulatory": "prohibition",
    "warning": "warning",
    "information": "guide",
    "complementary": "guide",
    "other-sign": "guide",
}

# 国家代码 → 中文名（只列常见/本项目涉及的，其余按需扩展）
COUNTRY_NAMES = {
    "BR": "巴西", "AR": "阿根廷", "CL": "智利", "CO": "哥伦比亚", "PE": "秘鲁",
    "EC": "厄瓜多尔", "BO": "玻利维亚", "PY": "巴拉圭", "UY": "乌拉圭",
    "VE": "委内瑞拉", "MX": "墨西哥",
    "RU": "俄罗斯", "BY": "白俄罗斯", "KZ": "哈萨克斯坦", "UA": "乌克兰",
    "DE": "德国", "FR": "法国", "ES": "西班牙", "IT": "意大利", "GB": "英国",
    "CN": "中国", "US": "美国", "CA": "加拿大", "AU": "澳大利亚", "JP": "日本",
}

COMMON_DIR = "_common"


def country_dir(root, cc):
    """返回某国家在素材库里的目录（可能不存在）。

    素材库根下允许两种命名：直接用 <CC>，或用 MTSD 的 <CC>_<中文名>。
    """
    cc = str(cc or "").upper()
    if not cc:
        return None
    direct = os.path.join(root, cc)
    if os.path.isdir(direct):
        return direct
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if name.upper().startswith(cc + "_") or name.upper() == cc:
                p = os.path.join(root, name)
                if os.path.isdir(p):
                    return p
    return None


def list_countries(root):
    """列出素材库里有哪些国家：[{code, name, dir, count}]。"""
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p) or name == COMMON_DIR:
            continue
        code = name.split("_")[0].upper()
        if len(code) != 2 or not code.isalpha():
            continue
        n = sum(1 for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        out.append({"code": code, "name": COUNTRY_NAMES.get(code, name),
                    "dir": p, "count": n})
    return out


def find_asset(root, cc, class_name, allow_common=True):
    """按「国家 → 通用」顺序找某个类别的素材，返回路径或 None。

    class_name 用规范化名（speed_limit_60）；也接受同类别下的任意变体
    （如 speed_limit_60_v2.png，取第一个）。
    """
    def _search(d):
        if not d or not os.path.isdir(d):
            return None
        exact = os.path.join(d, class_name + ".png")
        if os.path.isfile(exact):
            return exact
        for ext in (".jpg", ".jpeg"):
            p = os.path.join(d, class_name + ext)
            if os.path.isfile(p):
                return p
        # 变体：class_name_v2.png / class_name_1.png
        for f in sorted(os.listdir(d)):
            stem = os.path.splitext(f)[0]
            if stem == class_name or stem.startswith(class_name + "_v") or \
                    stem.startswith(class_name + "_"):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    return os.path.join(d, f)
        return None

    hit = _search(country_dir(root, cc))
    if hit:
        return hit
    if allow_common:
        return _search(os.path.join(root, COMMON_DIR))
    return None


def summarize(root):
    """统计素材库覆盖情况（按国家、按类别），用于面板展示与自检。"""
    report = {"root": root, "countries": list_countries(root), "common": 0, "classes": {}}
    cd = os.path.join(root, COMMON_DIR)
    if os.path.isdir(cd):
        report["common"] = sum(1 for f in os.listdir(cd)
                               if f.lower().endswith((".png", ".jpg", ".jpeg")))
    # 汇总所有国家里出现过的类别
    for c in report["countries"]:
        d = c["dir"]
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            stem = os.path.splitext(f)[0]
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            if stem.endswith("_v2") or stem.endswith("_v3"):
                stem = stem[:-3]
            report["classes"].setdefault(stem, []).append(c["code"])
    report["class_count"] = len(report["classes"])
    return report


def mtsd_label_to_class(label):
    """把 MTSD 的层级标签转成我们的大类名。

    'warning--pedestrians-crossing--g4' → ('warning', 'pedestrians-crossing')
    无法识别时返回 (None, None)。
    """
    parts = str(label or "").split("--")
    if not parts:
        return None, None
    top = MTSD_TOP_MAP.get(parts[0].strip().lower())
    mid = parts[1].strip() if len(parts) > 1 else ""
    return top, mid


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    s = summarize(root)
    print("素材库: %s" % root)
    print("国家数: %d，通用素材: %d，类别数: %d"
          % (len(s["countries"]), s["common"], s["class_count"]))
    for c in s["countries"][:15]:
        print("  %s %-10s %d 张" % (c["code"], c["name"], c["count"]))
    print()
    print("标签解析自检：")
    for lab in ["warning--pedestrians-crossing--g4", "speed-limit--60--g1", "regulatory--no-entry--g1"]:
        print("  %-40s → %s" % (lab, mtsd_label_to_class(lab)))
