"""从人工标定的单应反解相机内参/外参。

给 UE 用：fov / pitch / 相机高度。
与 pixel2world 的 H 保持一致，保证「点击位置 == 渲染位置」。
"""
import json

import numpy as np


def cam_from_homography(H, img_w=1920, img_h=1080, cam_height_m=1.35):
    """从地面单应 H（世界 (Y左,Z前) -> 像素）反解 pinhole 参数。

    H 把地面点 (Y, Z, 1) 投到 (u, v, 1)。
    地平线 = H 的 (h10, h11, h12) 行。地面 Z 轴的单位方向在像素上的投影
    决定俯角；Y 轴单位决定焦距。

    返回 dict(fx, fy, cx, cy, pitch_deg, fov_deg, cam_height_m, source)
    """
    H = np.asarray(H, float).reshape(3, 3)

    # 地平线：H 作用在 (Y, Z) = (t, inf) 的极限 -> 行 h1 / h2
    # 这里用更稳的办法：地平线是 h20*Y + h21*Z + h22 = 0 的像
    # 像素 v_inf = h11 / h21
    h = H
    if abs(h[2, 1]) < 1e-12:
        v_inf = img_h * 0.5
    else:
        v_inf = h[1, 1] / h[2, 1]

    # Y 方向（横向）单位 1 米在 Z=0 处的像素位移
    # 世界点 (0, 0) 和 (1, 0)
    p0 = h @ np.array([0.0, 0.0, 1.0])
    p1 = h @ np.array([1.0, 0.0, 1.0])
    if abs(p0[2]) < 1e-12 or abs(p1[2]) < 1e-12:
        raise ValueError("退化单应")
    p0 = p0[:2] / p0[2]
    p1 = p1[:2] / p1[2]

    # 世界点 (0, 0) 和 (0, 1) 的像素
    q1 = h @ np.array([0.0, 1.0, 1.0])
    q1 = q1[:2] / q1[2]

    # 相机在 (0, cam_h) 高，看向 Z 轴正向
    # 近似：fd = 1 米长度的像素尺度 / 距离
    # 用近处参考点估计焦距：取 Z=8m 处的两横向点
    Zref = 8.0
    a = h @ np.array([-1.0, Zref, 1.0])
    b = h @ np.array([1.0, Zref, 1.0])
    a = a[:2] / a[2]
    b = b[:2] / b[2]
    px_per_m_at_zref = abs(b[0] - a[0]) / 2.0
    fx = px_per_m_at_zref * Zref

    cy = float(v_inf)
    cx = float((p0[0] + p1[0]) / 2.0)

    # 俯角：地面点 Z 增大时 v 的变化率
    z_near, z_far = 6.0, 20.0
    vn = (h @ np.array([0.0, z_near, 1.0]))
    vf = (h @ np.array([0.0, z_far, 1.0]))
    vn = vn[1] / vn[2]
    vf = vf[1] / vf[2]
    # v = cy + fy * cam_h / Z * cos(pitch)
    # 粗略：由两点解 fy*cos(pitch) 与 pitch
    fy_cos = (vn - vf) * (z_near * z_far) / (z_far - z_near) / cam_height_m
    pitch_rad = np.arccos(np.clip(fy_cos / max(fx, 1e-6), -1, 1))
    pitch_deg = float(np.degrees(pitch_rad))

    fov_deg = float(2.0 * np.degrees(np.arctan(img_w / (2.0 * fx))))

    return {
        "fx": float(fx), "fy": float(fx),
        "cx": cx, "cy": cy,
        "pitch_deg": pitch_deg,
        "fov_deg": fov_deg,
        "cam_height_m": float(cam_height_m),
        "v_inf": float(v_inf),
        "source": "manual_calib",
    }


def load_ue_cam(bg_name, img_w=1920, img_h=1080):
    """给 UE 用的相机参数：优先人工标定，回退自动估计。"""
    import os
    stem = os.path.splitext(bg_name)[0]

    cp = "/data/render_hybrid/bgs/_calib_" + stem + ".json"
    if os.path.isfile(cp):
        try:
            with open(cp, encoding="utf-8") as f:
                d = json.load(f)
            out = cam_from_homography(
                np.array(d["H"], float).reshape(3, 3),
                img_w, img_h,
                float(d.get("cam_height_m") or 1.35))
            out["calib_path"] = cp
            return out
        except Exception as e:
            print("[cam] 人工标定反解失败: %s" % e)

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
        print("  %-14s %s" % (k, v))
