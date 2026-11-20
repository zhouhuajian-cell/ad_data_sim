import glob
import json
import os

import cv2
import numpy as np


def estimate_camera(img_path):
    """从车道线估计相机参数。

    思路：
    1) Canny + HoughLinesP 找长直线
    2) 按斜率分成左右两组，各自拟合，求交点 = 灭点
    3) 灭点纵向位置 → 俯仰角
    4) 假设车道宽度 3.5m、车辆高度 1.35m → 反推焦距
    """
    im = cv2.imread(img_path)
    if im is None:
        return None
    h, w = im.shape[:2]

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # 只看下半部分（道路区）
    roi = edges.copy()
    roi[:int(h * 0.35)] = 0
    roi[int(h * 0.92):] = 0        # 去掉车头遮挡

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 80,
                            minLineLength=int(h * 0.18), maxLineGap=30)
    if lines is None:
        return {"ok": False, "reason": "no_lines"}

    left, right = [], []
    for ln in lines:
        a = np.asarray(ln).reshape(-1)
        if a.size < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in a[:4]]
        if y2 == y1:
            continue
        slope = (y2 - y1) / float(x2 - x1) if x2 != x1 else 999
        if abs(slope) < 0.15:
            continue
        seg = (x1, y1, x2, y2)
        # 车道线：向下发散。y 轴向下，斜率正 = 从左下到右下
        if slope > 0.3 and x1 < w * 0.55:
            right.append(seg)
        elif slope < -0.3 and x1 > w * 0.45:
            left.append(seg)

    if len(left) < 2 or len(right) < 2:
        return {"ok": False,
                "reason": "not_enough_lines",
                "left": len(left), "right": len(right)}

    def fit(lines_):
        pts = []
        for x1, y1, x2, y2 in lines_:
            pts.append((x1, y1))
            pts.append((x2, y2))
        pts = np.array(pts, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts.reshape(-1, 1, 2),
                                     cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0)

    lvx, lvy, lx0, ly0 = fit(left)
    rvx, rvy, rx0, ry0 = fit(right)

    # 交点
    det = lvx * rvy - lvy * rvx
    if abs(det) < 1e-6:
        return {"ok": False, "reason": "parallel"}
    t = ((rx0 - lx0) * rvy - (ry0 - ly0) * rvx) / det
    vpx = lx0 + lvx * t
    vpy = ly0 + lvy * t

    if not (0 < vpx < w and -h < vpy < h * 0.9):
        return {"ok": False, "reason": "vp_out_of_range",
                "vp": [round(vpx, 1), round(vpy, 1)]}

    # 俯仰角：灭点相对主点的偏移
    cy0 = h * 0.5
    pitch_deg = float(np.degrees(np.arctan((cy0 - vpy) / (h * 0.9))))

    # 焦距：假设远端车道宽 3.5m，在两条线之间取 y=0.55h 处测量像素宽
    probe_y = int(h * 0.55)

    def x_at(vx, vy, x0, y0, y):
        if abs(vy) < 1e-9:
            return None
        return x0 + vx * (y - y0) / vy

    xl = x_at(lvx, lvy, lx0, ly0, probe_y)
    xr = x_at(rvx, rvy, rx0, ry0, probe_y)
    if xl is None or xr is None or xr <= xl:
        return {"ok": False, "reason": "lane_measure_failed"}

    lane_px = xr - xl
    dist_m = 12.0                      # 假设测量处在车前方 12m
    cam_h_m = 1.35
    lane_w_m = 3.5
    # 相似三角形：lane_px / f = lane_w_m / dist_m
    fx = lane_px * dist_m / lane_w_m
    fy = fx                           # 假设方形像素

    return {
        "ok": True,
        "width": w, "height": h,
        "cx": round(vpx, 1), "cy": round(vpy, 1),
        "vanish_point": [round(vpx, 1), round(vpy, 1)],
        "fx": round(fx, 1), "fy": round(fy, 1),
        "fov_deg": round(float(np.degrees(2 * np.arctan(w / (2 * fx)))), 2),
        "pitch_deg": round(pitch_deg, 2),
        "cam_height_m": cam_h_m,
        "lane_px_at_probe": round(lane_px, 1),
        "probe_y": probe_y,
        "lines_left": len(left), "lines_right": len(right),
    }


if __name__ == "__main__":
    fs = sorted(glob.glob("/data/render_hybrid/bgs/*.jpg"))
    out = {}
    for f in fs:
        name = os.path.basename(f)
        r = estimate_camera(f)
        out[name] = r
        if r and r.get("ok"):
            print("%-14s fx=%7.1f fov=%6.1f° pitch=%6.2f° vp=(%.0f,%.0f)"
                  % (name, r["fx"], r["fov_deg"], r["pitch_deg"],
                     r["vanish_point"][0], r["vanish_point"][1]))
        else:
            print("%-14s 失败: %s" % (name, r.get("reason") if r else "read_fail"))

    p = "/data/render_hybrid/bgs/_camera_est.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已保存:", p)
