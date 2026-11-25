#!/usr/bin/env python3
"""生成一套完整的仿真样例包，结构对齐真实 to_label 数据。

输出根目录：/mnt/Data_Platform/zhj_datasim/sample_pack

结构：
  sample_pack/
  ├── OBJ/
  │   └── <clip>/
  │       └── <clip_id>-<ts>/
  │           ├── cam_st_x2f_front/<ts>.jpg      （仿真图）
  │           ├── cam_st_x2f_front/<ts>.jpg.et   （标记）
  │           ├── lidar_hs_pandar128_topmiddle/<ts>.bin  （真实点云，若有）
  │           └── label_lidar_hs_pandar128_topmiddle/
  │               ├── <ts>.xml
  │               └── <ts>_ctrl.txt
  ├── TSR/
  │   └── <clip>/
  │       ├── <ts>.jpg
  │       └── <ts>.xml
  ├── GND/
  │   └── <ts>/
  │       ├── img_ori/<ts>.jpg
  │       ├── img_rec/<ts>.jpg
  │       ├── seg_lane/<ts>.png
  │       ├── freespace_mask_ori_3.0.0/<ts>.png
  │       ├── obj_mask_ori_3.1.1/<ts>.png
  │       ├── sky_mask_ori_3.0.0/<ts>.png
  │       └── parameter_yaml/...
  └── README.md
"""
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")

OUT_ROOT = "/mnt/Data_Platform/zhj_datasim/sample_pack"

# 真实数据源
OBJ_SRC = ("/mnt/done_data_new/18_bev-obj/02_overseas/CZ/04_songbiao/"
           "已送/20261021_LYNK01_CZ_guizhou_1v_1_tuoche/"
           "134166-1785156623.099")
GND_SRC = ("/mnt/done_data_new/31_ex_GND/05_maxieye/05_MG6_002/"
           "20260928/1790569917.088")
TSR_SRC = ("/mnt/done_data_new/05_tlts/TLTS_overseas/auto_train_done/"
           "20261111_overseas_tsr_DE_simulation_148/DE/"
           "20260809_LYNK01_DE_2M_night_TLTS_release_slow_down_rider")


def log(s):
    print("[PACK] %s" % s)


def ensure(p):
    os.makedirs(p, exist_ok=True)
    return p


# ============================================================
# OBJ
# ============================================================
def build_obj():
    ts = "1785156623.099"
    clip = "20261021_LYNK01_CZ_guizhou_1v_1_tuoche"
    clip_dir = "134166-1785156623.099"

    root = ensure(os.path.join(OUT_ROOT, "OBJ", clip, clip_dir))
    d_img = ensure(os.path.join(root, "cam_st_x2f_front"))
    d_pc = ensure(os.path.join(root, "lidar_hs_pandar128_topmiddle"))
    d_lab = ensure(os.path.join(root, "label_lidar_hs_pandar128_topmiddle"))

    # 1) 图片：用真实原图 + 现有的多车型合成图（若有）
    src_jpg = os.path.join(OBJ_SRC, "cam_st_x2f_front", "%s.jpg" % ts)
    dst_jpg = os.path.join(d_img, "%s.jpg" % ts)
    if os.path.isfile(src_jpg):
        shutil.copy2(src_jpg, dst_jpg)
        log("OBJ 图片: %s" % dst_jpg)
    # .et marker
    open(os.path.join(d_img, "%s.jpg.et" % ts), "w").close()

    # 2) 点云：真实 bin 直接复制
    src_bin = os.path.join(OBJ_SRC, "lidar_hs_pandar128_topmiddle",
                           "%s.bin" % ts)
    if os.path.isfile(src_bin):
        shutil.copy2(src_bin, os.path.join(d_pc, "%s.bin" % ts))
        open(os.path.join(d_pc, "%s.bin.et" % ts), "w").close()
        log("OBJ 点云: %s" % os.path.join(d_pc, "%s.bin" % ts))
    else:
        # 占位空点云
        np.zeros(10, np.float32).tofile(os.path.join(d_pc, "%s.bin" % ts))
        log("OBJ 点云: 占位")

    # 3) XML：直接复制真实标注（作为 golden reference）
    src_xml = os.path.join(OBJ_SRC, "label_lidar_hs_pandar128_topmiddle",
                           "%s.xml" % ts)
    if os.path.isfile(src_xml):
        shutil.copy2(src_xml, os.path.join(d_lab, "%s.xml" % ts))
        log("OBJ XML: %s" % os.path.join(d_lab, "%s.xml" % ts))
    open(os.path.join(d_lab, "%s_ctrl.txt" % ts), "w").close()


# ============================================================
# TSR
# ============================================================
def build_tsr():
    clip = "20260809_LYNK01_DE_2M_night_TLTS_release_slow_down_rider"
    root = ensure(os.path.join(OUT_ROOT, "TSR", clip))

    ts = "1786220951.348"
    src_jpg = os.path.join(TSR_SRC, "%s.jpg" % ts)
    src_xml = os.path.join(TSR_SRC, "%s.xml" % ts)

    if os.path.isfile(src_jpg):
        shutil.copy2(src_jpg, os.path.join(root, "%s.jpg" % ts))
        log("TSR 图片: %s" % os.path.join(root, "%s.jpg" % ts))
    if os.path.isfile(src_xml):
        shutil.copy2(src_xml, os.path.join(root, "%s.xml" % ts))
        log("TSR XML(真实): %s" % os.path.join(root, "%s.xml" % ts))

    # 再用导出器生成一份仿真标记版（isVirtualScene=true）
    from export_tsr_gnd import export_tsr
    ann = {
        "objects": [
            {"class": "speed_limit", "shape": "circle",
             "bbox": [514.33, 377.70, 582.83, 445.67],
             "f_value": 60, "first_attribute": "limit",
             "limit_type": "SPEED", "type": "RELEASE",
             "label_type": "POS"},
        ]
    }
    tmp = "/tmp/_tsr_ann.json"
    json.dump(ann, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    export_tsr(tmp, root, "1786220951.348_sim")
    log("TSR XML(仿真): %s" % os.path.join(root,
                                           "1786220951.348_sim.xml"))


# ============================================================
# GND
# ============================================================
def build_gnd():
    ts = "1790569917.088"
    root = ensure(os.path.join(OUT_ROOT, "GND", ts))

    # 真实原图
    src_jpg = os.path.join(GND_SRC, "img_ori", "%s.jpg" % ts)
    d_ori = ensure(os.path.join(root, "img_ori"))
    if os.path.isfile(src_jpg):
        shutil.copy2(src_jpg, os.path.join(d_ori, "%s.jpg" % ts))
        log("GND 原图: %s" % os.path.join(d_ori, "%s.jpg" % ts))

    # 真实 seg_lane 作为 golden
    src_seg = os.path.join(GND_SRC, "seg_lane", "%s.png" % ts)
    d_seg = ensure(os.path.join(root, "seg_lane"))
    if os.path.isfile(src_seg):
        shutil.copy2(src_seg, os.path.join(d_seg, "%s.png" % ts))
        log("GND seg_lane(真实): %s" % os.path.join(d_seg, "%s.png" % ts))

    # 仿真图 img_rec
    d_rec = ensure(os.path.join(root, "img_rec"))
    sim = None
    for cand in ("/tmp/gnd_full/sim.jpg", "/data/align_test/bg.jpg"):
        if os.path.isfile(cand):
            sim = cand
            break
    if sim:
        shutil.copy2(sim, os.path.join(d_rec, "%s.jpg" % ts))
        log("GND img_rec(仿真): %s" % os.path.join(d_rec, "%s.jpg" % ts))

    # 三个 mask（真实数据有的直接复制）
    for src_dir, dname in (
            ("freespace_mask_ori_3.0.0", "freespace_mask_ori_3.0.0"),
            ("obj_mask_ori_3.1.1", "obj_mask_ori_3.1.1"),
            ("sky_mask_ori_3.0.0", "sky_mask_ori_3.0.0")):
        s = os.path.join(GND_SRC, src_dir, "%s.png" % ts)
        d = ensure(os.path.join(root, dname))
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(d, "%s.png" % ts))
            log("GND %s: %s" % (dname, os.path.join(d, "%s.png" % ts)))

    # parameter_yaml
    s_yaml = os.path.join(GND_SRC, "parameter_yaml")
    if os.path.isdir(s_yaml):
        shutil.copytree(s_yaml, os.path.join(root, "parameter_yaml"),
                        dirs_exist_ok=True)
        log("GND parameter_yaml: 已复制")


# ============================================================
# README
# ============================================================
README = """# 仿真数据样例包

生成时间：2026-11-23
用途：对照真实 to_label 标注格式，验证仿真产物兼容性。

## 目录

### OBJ（3D 目标）
```
OBJ/<clip>/<clip_id>-<ts>/
├── cam_st_x2f_front/
│   ├── <ts>.jpg         前视相机图像
│   └── <ts>.jpg.et      空标记文件
├── lidar_hs_pandar128_topmiddle/
│   ├── <ts>.bin         LiDAR 点云
│   └── <ts>.bin.et
└── label_lidar_hs_pandar128_topmiddle/
    ├── <ts>.xml         3D 框标注（61 字段）
    └── <ts>_ctrl.txt    空控制文件
```

**XML 关键约定（已用真实数据投影验证）**
- `objWx` = 自车坐标系纵向距离（米）
- `objWy` = 自车坐标系横向位置（左正右负）
- `objWz` = 车辆中心离地高度（= H/2）
- `objYaw` = 弧度
- 投影：`p_cam = R @ p_body + T`，R 由 `rotation_vector` 罗德里格斯展开
- 8 个组：Three / Cone / Unknown / Unsure / Special / Vehicle / Rider / Pedestrian
- Pedestrian 额外有 `<crowd>` 字段

### TSR（交通标志）
```
TSR/<clip>/
├── <ts>.jpg             图像
└── <ts>.xml             标注（20 字段，平铺同名配对）
```

**XML 关键约定**
- bbox 用 2 个对角点：`edgePts00X/Y`（左上）、`edgePts01X/Y`（右下）
- `isVirtualScene="true"` 表示仿真数据
- 4 个组：Vehicle / Pedestrian / Rider / TrafficSign

### GND（地面标线）
```
GND/<ts>/
├── img_ori/<ts>.jpg                    原始图像
├── img_rec/<ts>.jpg                    仿真图
├── seg_lane/<ts>.png                   车道线标注（调色板）
├── freespace_mask_ori_3.0.0/<ts>.png   可行驶区域
├── obj_mask_ori_3.1.1/<ts>.png         车辆 mask
├── sky_mask_ori_3.0.0/<ts>.png         天空 mask
└── parameter_yaml/<date>/              相机内外参
```

**seg_lane 调色板（RGB）**
| 颜色 | 含义 |
|---|---|
| (0, 0, 0) | 背景 |
| (0, 255, 0) | 车道边线 edge |
| (244, 35, 232) | 虚线/标线 dash |

注：用 PIL 读图得到 RGB；若用 cv2 读取则是 BGR（232,35,244）。

## 数据来源

本包中的图像、点云、真实标注均直接取自真实已标注数据集，作为格式对照基准。
仿真的 XML（文件名带 `_sim` 后缀）由 `tools/export_to_label.py` /
`tools/export_tsr_gnd.py` 生成。

## 导出脚本

```bash
# OBJ
python tools/export_to_label.py --ann <sim_ann.json> --out <dir>

# TSR
python tools/export_tsr_gnd.py tsr --ann <sim_ann.json> --out <dir> --ts <ts>

# GND（一次出全套 mask）
python tools/export_tsr_gnd.py gnd-full --orig a.jpg --sim b.jpg \\
    --out <dir> --ts <ts> --yaml <parameter_yaml源>
```
"""


def main():
    ensure(OUT_ROOT)
    log("输出根目录: %s" % OUT_ROOT)
    print()
    build_obj()
    print()
    build_tsr()
    print()
    build_gnd()
    print()
    with open(os.path.join(OUT_ROOT, "README.md"), "w",
              encoding="utf-8") as f:
        f.write(README)
    log("README.md 已写入")
    print()

    # 统计
    total = 0
    total_sz = 0
    for dp, dn, fn in os.walk(OUT_ROOT):
        for f in fn:
            total += 1
            total_sz += os.path.getsize(os.path.join(dp, f))
    log("=== 完成：%d 个文件, %.1f MB ===" % (total, total_sz / 1e6))


if __name__ == "__main__":
    main()
