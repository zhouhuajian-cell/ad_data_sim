"""从人工标定 H 反解 UE 相机参数（稳健版）。

思路：UE 侧调 `fov` 和 `pitch`，让「世界点 → 像素」与人工标定 H 一致。
不直接代数反解（标定点分布不对称会放大误差），而是在参数网格上
最小化「参考世界点的像素误差」。
"""
import json
import os

import numpy as np

REF_POINTS = [  # (Y_left_m, Z_fwd_m) 覆盖画面主要区域
    (-3.0, 8.0), (0.0, 8.0), (3.0, 8.0),
    (-2.0, 12.0), (0.0, 12.0), (2.0, 12.0),
    (-1.0, 16.0), (0.0, 16.0), (1.0, 16.0),
    (-0.8, 20.0), (0.0, 20.0), (0.8, 20.0),
]


def ue_project(x_cm, y_cm, z_cm, cam):
    """UE 相机模型：位于 (0,0,cz) 朝 UE +X，返回像素 (u,v)。

    UE: X 前、Y 左、Z 上。相机 pitch 为俯角（负向下看）。
    """
    fx = fy = cam["fy"]
    cx_px = cam["cx"]
    cy_px = cam["cy"]
    h = cam["cam_height_m"] * 100.0

    # 世界点相对相机
    dx = x_cm
    dy = y_cm
    dz = z_cm - h          # 地面点 = -h

    pr = np.radians(cam["pitch_deg"])
    cp, sp = np.cos(pr), np.sin(pr)

    # 绕 Y 轴旋转（UE pitch）
    # 前向向量在 (x, z) 平面
    fwd_x = cp
    fwd_z = sp
    # 右向量 = Y 轴方向（UE 里 Y 指向左，所以屏幕右 = -Y）
    zc = dx * fwd_x + dz * fwd_z        # 沿光轴
    if zc <= 1e-6:
        return None
    # 垂直方向（camera down）
    yc = -dy                            # 屏幕右 = -Y
    # 水平方向（camera up/down）
    xc = dx * (-fwd_z) + dz * fwd_x     # 与 fwd 正交

    u = cx_px + fx * yc / zc
    v = cy_px + fy * xc / zc
    return (u, v)


def model_error(cam, H, refs, img_w=1920, img_h=1080):
    """模型像素 与 人工标定 H 像素 的平均误差。"""
    errs = []
    for (Y, Z) in refs:
        gt = cv_H_project(H, Y, Z)
        if gt is None:
            continue
        pr = ue_project(Z * 100.0, Y * 100.0, 0.0, cam)
        if pr is None:
            errs.append(9999.0)
            continue
        errs.append(float(np.hypot(pr[0] - gt[0], pr[1] - gt[1])))
    return float(np.mean(errs)) if errs else 9999.0


def cv_H_project(H, Y, Z):
    """人工标定 H 是 (Y, Z) -> 像素。"""
    p = H @ np.array([Y, Z, 1.0])
    if abs(p[2]) < 1e-12:
        return None
    return (p[0] / p[2], p[1] / p[2])


def fit_ue_cam(H, img_w=1920, img_h=1080, cam_height_m=1.35):
    """网格搜索 fov / cy / pitch / cx，最小化参考点误差。"""
    H = np.asarray(H, float).reshape(3, 3)

    best = None
    # cy 由地平线给出（H 的 v_inf）
    if abs(H[2, 1]) > 1e-12:
        v_inf = float(H[1, 1] / H[2, 1])
    else:
        v_inf = img_h * 0.5

    for fov in np.arange(40.0, 120.0, 2.0):
        fy = (img_w / 2.0) / np.tan(np.radians(fov) / 2.0)
        for pitch in np.arange(-20.0, 4.0, 0.5):
            for cx_off in (-40, -20, 0, 20, 40):
                cam = {
                    "fx": fy, "fy": fy,
                    "cx": img_w / 2.0 + cx_off,
                    "cy": v_inf,
                    "pitch_deg": float(pitch),
                    "fov_deg": float(fov),
                    "cam_height_m": cam_height_m,
                }
                e = model_error(cam, H, REF_POINTS, img_w, img_h)
                if best is None or e < best[0]:
                    best = (e, cam)

    err, cam = best
    cam = dict(cam)
    cam["fit_error_px"] = round(float(err), 2)
    cam["source"] = "fitted_from_manual_calib"
    return cam


def load_ue_cam(bg_name, img_w=1920, img_h=1080):
    """给 UE 的相机参数：优先从人工标定拟合，回退自动估计。"""
    stem = os.path.splitext(bg_name)[0]

    cp = "/data/render_hybrid/bgs/_calib_" + stem + ".json"
    if os.path.isfile(cp):
        try:
            with open(cp, encoding="utf-8") as f:
                d = json.load(f)
            return fit_ue_cam(np.array(d["H"], float).reshape(3, 3),
                              img_w, img_h,
                              float(d.get("cam_height_m") or 1.35))
        except Exception as e:
            print("[cam] 拟合失败: %s" % e)

    ep = "/data/render_hybrid/bgs/_camera_est.json"
    if os.path.isfile(ep):
        try:
            with open(ep, encoding="utf-8") as f:
                all_est = json.load(f)
            e = all_est.get(bg_name) or {}
            if e.get("ok"):
                return {
                    "fx": float(e["fx"]), "fy": float(e["fy"]),
                    "cx": float(e.get("cx") or img_w * 0.5),
                    "cy": float(e.get("cy") or img_h * 0.55),
                    "pitch_deg": float(e.get("pitch_deg") or 0.0),
                    "fov_deg": float(e.get("fov_deg") or 70.0),
                    "cam_height_m": float(e.get("cam_height_m") or 1.35),
                    "source": "auto_estimate",
                }
        except Exception:
            pass

    return {"fx": 800.0, "fy": 800.0, "cx": img_w * 0.5, "cy": img_h * 0.55,
            "pitch_deg": 0.0, "fov_deg": 70.0, "cam_height_m": 1.35,
            "source": "default"}


if __name__ == "__main__":
    import sys
    bg = sys.argv[1] if len(sys.argv) > 1 else "cand_04.jpg"
    c = load_ue_cam(bg)
    print("bg:", bg)
    for k, v in c.items():
        print("  %-22s %s" % (k, v))
