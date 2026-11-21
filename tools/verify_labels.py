"""把 done_data 的 3D 标注投影到真实图上，验证坐标与标定。

产出：/tmp/label_overlay.jpg
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from done_data_labels import (object_corners, parse_xml,
                              project_body_to_pixel)

B = ("/mnt/done_data_new/18_bev-obj/00_M57/00_daizhongshen_changchun/"
     "已终审/hs5001/20261120_HS5001_changchun_1v_closecar/"
     "133007-1788053647.136")
LAB = os.path.join(B, "label_lidar_hs_pandar128_topmiddle",
                   "1788053647.136.xml")
IMG = os.path.join(B, "cam_st_x2f_front", "1788053647.136.jpg")

print("标注:", os.path.basename(LAB))
print("图像:", os.path.basename(IMG))

d = parse_xml(LAB)
cam = d["cameras"]["cam_st_x2f_front"]
im = cv2.imread(IMG)
if im is None:
    raise SystemExit("读图失败: " + IMG)
h, w = im.shape[:2]
print("图尺寸: %dx%d" % (w, h))

vis = im.copy()

COLORS = {
    "car": (0, 200, 255),
    "pedestrian": (255, 120, 0),
    "rider": (200, 0, 255),
    "three": (0, 255, 120),
}

n_drawn = 0
n_skip = 0
for o in d["objects"]:
    if o["delete"]:
        continue
    c8 = object_corners(o)
    uv, z, ok = project_body_to_pixel(c8, cam)
    if not ok.any():
        n_skip += 1
        continue

    col = COLORS.get(o["kind"], (200, 200, 200))

    # 只画可见点的连线（12 条棱）
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        if ok[a] and ok[b]:
            p1 = (int(uv[a][0]), int(uv[a][1]))
            p2 = (int(uv[b][0]), int(uv[b][1]))
            if (-5000 < p1[0] < w + 5000 and -5000 < p1[1] < h + 5000 and
                    -5000 < p2[0] < w + 5000 and -5000 < p2[1] < h + 5000):
                cv2.line(vis, p1, p2, col, 1)

    # 底面
    if ok[[0, 1, 2, 3]].all():
        pts = np.array([[int(uv[i][0]), int(uv[i][1])] for i in (0, 1, 2, 3)],
                       np.int32)
        cv2.polylines(vis, [pts], True, col, 2)

    # 标签
    if ok[4]:
        cx, cy = int(uv[4][0]), int(uv[4][1])
        if 0 <= cx < w and 0 <= cy < h:
            txt = "%s %.0fm" % (o["kind"][:5], o["Wx"])
            cv2.putText(vis, txt, (cx - 20, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    n_drawn += 1

print("画出: %d 个，跳过(全在相机后): %d 个" % (n_drawn, n_skip))

# 只存中间区域方便看
scale = 1600.0 / w
out = cv2.resize(vis, (1600, int(h * scale)), interpolation=cv2.INTER_AREA)
cv2.imwrite("/tmp/label_overlay.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("产出: /tmp/label_overlay.jpg %dx%d" % out.shape[1::-1])

# 统计
from collections import Counter
c = Counter(o["kind"] for o in d["objects"] if not o["delete"])
print("\n未删除物体:", dict(c))
front = [o for o in d["objects"] if not o["delete"] and o["Wx"] > 0]
print("车前方(Wx>0):", len(front))
