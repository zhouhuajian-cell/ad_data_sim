"""像素坐标 → UE 世界坐标（厘米）。

用底图的相机估计参数做反投影：
  已知像素 (u,v) + 地面 z=0 → 求世界 (X 前, Y 左)
"""
import json
import os

import numpy as np

CAM_EST = "/data/render_hybrid/bgs/_camera_est.json"
CALIB_DIR = "/data/render_hybrid/bgs"


def load_cam(bg_name):
    """读相机参数：优先人工标定，其次自动估计。"""
    stem = os.path.splitext(bg_name)[0]
    cp = os.path.join(CALIB_DIR, "_calib_" + stem + ".json")
    if os.path.isfile(cp):
        with open(cp, encoding="utf-8") as f:
            d = json.load(f)
        return {"mode": "homography", "H": np.array(d["H"], float).reshape(3, 3)}
    if os.path.isfile(CAM_EST):
        with open(CAM_EST, encoding="utf-8") as f:
            est = json.load(f)
        e = est.get(bg_name) or {}
        if e.get("ok"):
            return {"mode": "camera", "fx": e["fx"], "fy": e["fy"],
                    "cx": e.get("cx"), "cy": e.get("cy")}
    return None


def pixel_to_world(cam, u, v, cam_h_m=1.35, img_w=1920, img_h=1080):
    """像素 → 世界地面坐标（米，X 前 / Y 左）。失败返回 None。"""
    if cam is None:
        return None

    if cam["mode"] == "homography":
        # H 是 world(Y,Z) -> pixel，求逆
        try:
            Hi = np.linalg.inv(cam["H"])
        except Exception:
            return None
        p = np.array([u, v, 1.0])
        w = Hi @ p
        if abs(w[2]) < 1e-9:
            return None
        Y = w[0] / w[2]      # 横向（左正）
        Z = w[1] / w[2]      # 纵向（前正）
        if Z <= 0.5:
            return None
        return {"x_fwd_m": float(Z), "y_left_m": float(Y)}

    # 相机模式
    fx, fy = cam["fx"], cam["fy"]
    cx = cam.get("cx") or img_w * 0.5
    cy = cam.get("cy") or img_h * 0.55

    # 反投影：zc = fy*cam_h/(v-cy); Y_left = (u-cx)*zc/fx; X_fwd = zc
    if abs(v - cy) < 1e-6:
        return None
    zc = fy * cam_h_m / (v - cy)
    if zc <= 0.5:
        return None
    X = zc
    Y = (u - cx) * zc / fx
    return {"x_fwd_m": float(X), "y_left_m": float(Y)}


def pixel_to_ue_cm(cam, u, v, **kw):
    """像素 → UE 世界厘米坐标（X 前, Y 左, Z=0）。"""
    w = pixel_to_world(cam, u, v, **kw)
    if w is None:
        return None
    return {
        "x_cm": round(w["x_fwd_m"] * 100.0, 1),
        "y_cm": round(w["y_left_m"] * 100.0, 1),
        "z_cm": 0.0,
        "dist_m": round(w["x_fwd_m"], 2),
    }


if __name__ == "__main__":
    import sys
    bg = sys.argv[1] if len(sys.argv) > 1 else "cand_04.jpg"
    cam = load_cam(bg)
    print("bg:", bg, "cam:", cam["mode"] if cam else "NONE")
    for (u, v) in [(960, 800), (700, 900), (1200, 750), (960, 700)]:
        r = pixel_to_ue_cm(cam, u, v)
        print("  pixel(%d,%d) -> %s" % (u, v, r))
