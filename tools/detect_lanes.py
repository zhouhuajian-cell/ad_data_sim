#!/usr/bin/env python3
"""车道线检测：传统 CV（比 YOLO 更适合细长线）。

方法：
  1) 逆透视变换（IPM）把路面变鸟瞰
  2) 在 IPM 空间做颜色阈值 + 滑窗拟合（经典 pipeline）
  3) 反变换回原图

这样得到的车道线在真实世界坐标里是准的，替换时也能对齐。
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from real_calib import body_to_pixel_real, load_cam_real


def build_ipm(cam, img_shape, x_range=(6.0, 40.0), y_range=(-8.0, 8.0),
              out_size=(400, 800)):
    """构造 IPM 单应：body 地面 (x 前, y 左) → 鸟瞰图。

    返回 (M, Minv, out_w, out_h)
    """
    h, w = img_shape
    x0, x1 = x_range
    y0, y1 = y_range
    ow, oh = out_size

    # 世界四角 → 像素
    world_pts = np.array([
        [x0, y0, 0.0], [x0, y1, 0.0], [x1, y1, 0.0], [x1, y0, 0.0]
    ])
    uv, z, ok = body_to_pixel_real(cam, world_pts)
    if not ok.all():
        return None, None, ow, oh
    if not np.isfinite(uv).all():
        return None, None, ow, oh

    src = uv.astype(np.float32)
    # 鸟瞰：x(前) → 竖直向下，y(左) → 水平
    dst = np.array([[ow * 0.5, oh], [ow * 0.5, 0],
                    [ow - 1, 0], [ow - 1, oh]], np.float32)
    # 用世界 → 鸟瞰的对应关系
    # 更直观：x0(近) 在底部，x1(远) 在顶部；y0(右) 在左，y1(左) 在右
    dst = np.array([
        [0, oh - 1],          # (x0, y0) 近右下
        [ow - 1, oh - 1],     # (x0, y1) 近左下
        [ow - 1, 0],          # (x1, y1) 远左下
        [0, 0],               # (x1, y0) 远右下
    ], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = np.linalg.inv(M)
    return M, Minv, ow, oh


def detect_lanes_ipm(bg, cam, debug_dir=None):
    """在 IPM 空间检测车道线，返回世界坐标的线列表。"""
    h, w = bg.shape[:2]
    M, Minv, ow, oh = build_ipm(cam, (h, w))
    if M is None:
        return []

    warp = cv2.warpPerspective(bg, M, (ow, oh))

    # 白/黄线阈值
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]

    white = ((v > 150) & (s < 60)).astype(np.uint8) * 255
    yellow = ((v > 120) & (s > 60) & (s < 200) &
              (hsv[..., 0] > 15) & (hsv[..., 0] < 40)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(white, yellow)

    # 形态学：强化纵向连续
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "warp.jpg"), warp)
        cv2.imwrite(os.path.join(debug_dir, "mask.png"), mask)

    # 在 IPM 里每行找线段中心
    # x(前) 对应图像 y；y(左) 对应图像 x
    x_lo, x_hi = 6.0, 40.0
    y_lo, y_hi = -8.0, 8.0

    lanes = []
    # 从远到近逐行扫描（顶部=远）
    for row in range(0, oh, 4):
        band = mask[max(0, row - 2):row + 3]
        if band.size == 0:
            continue
        col = band.any(axis=0).astype(np.uint8)
        if col.sum() < 3:
            continue

        # 分段（连续白区）
        segs = []
        st = None
        for i in range(len(col)):
            if col[i] and st is None:
                st = i
            elif not col[i] and st is not None:
                if i - st >= 2:
                    segs.append((st, i - 1))
                st = None
        if st is not None and len(col) - st >= 2:
            segs.append((st, len(col) - 1))

        # 转回世界坐标
        fx = row / max(oh - 1, 1)
        x_m = x_hi - (x_hi - x_lo) * fx       # 顶=远, 底=近
        for (c0, c1) in segs:
            fy0 = c0 / max(ow - 1, 1)
            fy1 = c1 / max(ow - 1, 1)
            y0 = y_hi - (y_hi - y_lo) * fy0
            y1 = y_hi - (y_hi - y_lo) * fy1
            lanes.append({
                "x_m": round(float(x_m), 2),
                "y0_m": round(float(y0), 2),
                "y1_m": round(float(y1), 2),
                "y_center": round(float((y0 + y1) / 2), 2),
                "width_m": round(float(abs(y1 - y0)), 3),
            })

    return lanes


def cluster_lanes(lanes, x_tol=1.5):
    """按 x(距离) 聚类成几条连续的线。"""
    if not lanes:
        return []
    lanes = sorted(lanes, key=lambda l: l["x_m"])
    groups = []
    cur = [lanes[0]]
    for l in lanes[1:]:
        if abs(l["x_m"] - cur[-1]["x_m"]) <= x_tol:
            cur.append(l)
        else:
            groups.append(cur)
            cur = [l]
    groups.append(cur)

    out = []
    for g in groups:
        if len(g) < 3:
            continue
        ys = [item["y_center"] for item in g]
        xs = [item["x_m"] for item in g]
        # 线性拟合 y = a*x + b
        if len(set(xs)) < 2:
            continue
        a, b = np.polyfit(xs, ys, 1)
        out.append({
            "y_at_near": round(float(a * min(xs) + b), 2),
            "y_at_far": round(float(a * max(xs) + b), 2),
            "slope": round(float(a), 4),
            "x_min": round(float(min(xs)), 1),
            "x_max": round(float(max(xs)), 1),
            "n_pts": len(g),
            "width_m": round(float(np.mean([i["width_m"] for i in g])), 3),
        })
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--stim", default="1788137262.664")
    a = ap.parse_args()

    REAL = "/data/test_data/real_clips"
    img_p = os.path.join(REAL, a.batch, a.clip, "cam_st_x2f_front",
                         a.stim + ".jpg")
    bg = cv2.imread(img_p)
    if bg is None:
        raise SystemExit("读图失败")

    cam = load_cam_real(a.batch, a.clip)
    if cam is None:
        raise SystemExit("无标定")

    print("图:", bg.shape, "标定 fx=%.1f" % cam["fx"])
    lanes = detect_lanes_ipm(bg, cam, "/tmp/lane_test")
    print("原始检测点:", len(lanes))

    clustered = cluster_lanes(lanes)
    print("\n聚类后的车道线 %d 条:" % len(clustered))
    for i, c in enumerate(clustered):
        print("  #%d  近端 y=%+.2fm  远端 y=%+.2fm  斜率=%.3f  宽=%.2fm  %d点"
              % (i, c["y_at_near"], c["y_at_far"], c["slope"],
                 c["width_m"], c["n_pts"]))

    # 可视化
    vis = bg.copy()
    colors = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255)]
    for i, c in enumerate(clustered):
        pts = []
        for x in np.arange(c["x_min"], c["x_max"], 0.5):
            y = c["slope"] * x + (c["y_at_near"] - c["slope"] * c["x_min"])
            uv, z, ok = body_to_pixel_real(cam, [[x, y, 0.0]])
            if ok[0] and np.isfinite(uv[0]).all():
                pts.append(uv[0])
        if len(pts) > 1:
            pts = np.array(pts, np.int32)
            cv2.polylines(vis, [pts], False, colors[i % 4], 4)
            cv2.putText(vis, "L%d" % i, tuple(pts[len(pts) // 2]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors[i % 4], 2)

    cv2.imwrite("/tmp/lane_test/detected.jpg", vis,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("\n产出: /tmp/lane_test/detected.jpg")

    with open("/tmp/lane_test/lanes.json", "w", encoding="utf-8") as f:
        json.dump(clustered, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
