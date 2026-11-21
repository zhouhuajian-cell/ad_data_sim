"""把投影网格画在真实底图上，验证标定对不对。"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from maxieye_calib import (body_to_pixel, find_dataset, ground_homography,
                           list_frames, load_camera)

DS = find_dataset("141705")
CAM_DIR = "cam_st_x2f_front"
CAM = load_camera(DS, CAM_DIR)

frames = list_frames(DS, CAM_DIR)
print("数据集:", DS)
print("帧数:", len(frames))

# 取中间一帧
p = frames[len(frames) // 2]
im = cv2.imread(p)
if im is None:
    raise SystemExit("读图失败: " + p)
h, w = im.shape[:2]
print("图:", os.path.basename(p), "%dx%d" % (w, h))
print("标定期望: %dx%d" % (CAM["width"], CAM["height"]))

vis = im.copy()

# 1) 画纵向距离线（每 5m）
for x in range(5, 51, 5):
    pts = []
    for y in np.arange(-8.0, 8.01, 0.5):
        uv, z, ok = body_to_pixel([[x, y, 0.0]], CAM)
        if ok[0] and np.isfinite(uv[0]).all():
            pts.append(uv[0])
    if len(pts) > 1:
        pts = np.array(pts, np.int32)
        cv2.polylines(vis, [pts], False, (0, 220, 255), 2)
        # 标注
        uv, z, ok = body_to_pixel([[x, 0.0, 0.0]], CAM)
        if ok[0]:
            cv2.putText(vis, "%dm" % x, (int(uv[0][0]) - 30, int(uv[0][1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

# 2) 画横向线（每 1m，-6..6，从 5m 到 40m）
for y in np.arange(-6.0, 6.01, 1.0):
    pts = []
    for x in np.arange(5.0, 40.01, 0.5):
        uv, z, ok = body_to_pixel([[x, y, 0.0]], CAM)
        if ok[0] and np.isfinite(uv[0]).all():
            pts.append(uv[0])
    if len(pts) > 1:
        pts = np.array(pts, np.int32)
        col = (80, 255, 120) if abs(y) < 0.1 else (60, 180, 240)
        cv2.polylines(vis, [pts], False, col, 1 if abs(y) > 0.1 else 2)

# 3) 画一个 3.5×0.15 的车道线测试（x 从 6 到 30，y=-1.75）
quad_pts = []
for x in [6.0, 30.0]:
    for y in [-1.825, -1.675]:
        uv, z, ok = body_to_pixel([[x, y, 0.0]], CAM)
        if ok[0]:
            quad_pts.append(uv[0])
if len(quad_pts) == 4:
    q = np.array([quad_pts[0], quad_pts[2], quad_pts[3], quad_pts[1]], np.int32)
    cv2.fillConvexPoly(vis, q, (255, 255, 255))

# 4) 画锥桶位置（70cm 高，在 10m 和 15m）
for x, y in [(10, -1.0), (15, 0.5), (20, -0.5)]:
    uv, z, ok = body_to_pixel([[x, y, 0.0]], CAM)
    if ok[0]:
        c = (int(uv[0][0]), int(uv[0][1]))
        cv2.circle(vis, c, 8, (0, 0, 255), -1)
        cv2.putText(vis, "%.0fm" % x, (c[0] + 12, c[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 缩小输出
scale = 1600.0 / w
out = cv2.resize(vis, (1600, int(h * scale)), interpolation=cv2.INTER_AREA)
cv2.imwrite("/tmp/calib_overlay.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("产出: /tmp/calib_overlay.jpg  %dx%d" % out.shape[1::-1])

# 自洽性：像素→地面→像素
H = ground_homography(CAM)
print("\n地面单应 H 行列式: %.4g" % float(np.linalg.det(H)))
for (u, v) in [(1456, 866), (1455, 736), (1790, 866), (1121, 867)]:
    p3 = np.linalg.inv(H) @ np.array([u, v, 1.0])
    x, y = p3[0] / p3[2], p3[1] / p3[2]
    uv, z, ok = body_to_pixel([[x, y, 0.0]], CAM)
    err = np.hypot(uv[0][0] - u, uv[0][1] - v) if ok[0] else -1
    print("  (%.0f,%.0f) → 地面(%.2f,%.2f) → 回投误差 %.4f px"
          % (u, v, x, y, err))
