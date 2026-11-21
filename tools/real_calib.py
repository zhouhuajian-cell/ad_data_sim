"""扩展 pixel2world：支持 XML 真实标定（done_data 数据源）。

调用方式：
  load_cam_real(batch, clip)  → 从 XML 读 K/R/t
  pixel_to_ground_real(cam, u, v)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")

REAL_ROOT = "/data/test_data/real_clips"


def load_cam_real(batch, clip, xml_name=None):
    """从真实数据的 XML 读相机标定。

    返回 {"mode": "real_xml", "K": ..., "R": ..., "t": ..., "fx",...}
    """
    from label_gate import load_frame_labels

    # CAM-MULTI-ROOT: 支持 real_clips 和 real_clips_extra
    cp = None
    for root in ("/data/test_data/real_clips",
                 "/data/test_data/real_clips_extra"):
        p = os.path.join(root, batch, clip)
        if os.path.isdir(p):
            cp = p
            break
    if cp is None:
        return None

    lab = load_frame_labels(cp, xml_path=xml_name)
    if lab is None:
        return None
    cam = lab.get("camera")
    if not cam:
        return None

    return {
        "mode": "real_xml",
        "fx": float(cam["fx"]), "fy": float(cam["fy"]),
        "cx": float(cam["cx"]), "cy": float(cam["cy"]),
        "K": np.array(cam["K"], float).tolist(),
        "R": np.array(cam["R"], float).tolist(),
        "t": np.array(cam["t"], float).tolist(),
        "source": "xml:" + os.path.basename(clip),
        "camera_name": cam.get("name"),
    }


def body_to_pixel_real(cam, pts_body):
    """body 坐标 → 像素。p_cam = R @ p_body + t"""
    P = np.asarray(pts_body, float).reshape(-1, 3)
    R = np.array(cam["R"], float)
    t = np.array(cam["t"], float)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    Pc = P @ R.T + t
    z = Pc[:, 2]
    ok = z > 1e-6
    uv = np.full((len(P), 2), np.nan)
    if ok.any():
        uv[ok, 0] = cx + fx * Pc[ok, 0] / z[ok]
        uv[ok, 1] = cy + fy * Pc[ok, 1] / z[ok]
    return uv, z, ok


def pixel_to_body_real(cam, u, v):
    """像素 → body 地面点 (x 前, y 左)。

    先求地面单应再反解。
    """
    K = np.array(cam["K"], float)
    R = np.array(cam["R"], float)
    t = np.array(cam["t"], float)

    # 地面 z=0：p_cam = R[:, :2] @ [x,y] + t  （因为 R @ [x,y,0] = R[:, :2] @ [x,y]）
    A = R[:, :2]
    H = K @ np.column_stack([A, t])

    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    p = Hi @ np.array([u, v, 1.0])
    if abs(p[2]) < 1e-12:
        return None
    x = p[0] / p[2]
    y = p[1] / p[2]
    if x <= 0.5:
        return None
    return (float(x), float(y))


def horizon_v_real(cam):
    """地平线像素纵坐标。"""
    K = np.array(cam["K"], float)
    R = np.array(cam["R"], float)
    t = np.array(cam["t"], float)
    # 地面 (x,y) → 像素, v 在 x→∞ 时的极限
    # H 的 v 行: (K[1] @ [A | t])  →  v = (k11*(..)+k12*(..)+k13)/(k21*...)
    A = R[:, :2]
    H = K @ np.column_stack([A, t])
    # (x, y) → ∞ 沿任一方向；v 的极限 = H[1,0]/H[2,0] 或 H[1,1]/H[2,1]
    v1 = H[1, 0] / H[2, 0] if abs(H[2, 0]) > 1e-12 else None
    v2 = H[1, 1] / H[2, 1] if abs(H[2, 1]) > 1e-12 else None
    cands = [v for v in (v1, v2) if v is not None and 0 < v < 4320]
    if not cands:
        return None
    return float(sum(cands) / len(cands))


def pixel_to_ue_cm_real(cam, u, v, guard_px=25.0, max_dist_m=80.0):
    """像素 → 世界坐标（米），带地平线守卫。"""
    vh = horizon_v_real(cam)
    if vh is not None and v < (vh + guard_px):
        return None

    r = pixel_to_body_real(cam, u, v)
    if r is None:
        return None
    x, y = r
    if abs(x) > max_dist_m:
        return None
    return {"x_fwd_m": x, "y_left_m": y, "dist_m": x}


def cam_signature(cam):
    """用于校验：同一 clip 的标定是否一致。"""
    return "%s|%.2f|%.2f|%.2f|%.2f" % (
        cam.get("camera_name"), cam["fx"], cam["fy"], cam["cx"], cam["cy"])


if __name__ == "__main__":
    import glob

    clips = sorted(glob.glob(os.path.join(REAL_ROOT, "*", "*")))
    clips = [c for c in clips if os.path.isdir(os.path.join(c, "cam_st_x2f_front"))]
    if not clips:
        raise SystemExit("无本地数据")

    c0 = clips[0]
    parts = c0.replace(REAL_ROOT + "/", "").split("/")
    batch, clip = parts[0], parts[1]
    print("测试:", batch, "/", clip)

    cam = load_cam_real(batch, clip)
    if not cam:
        raise SystemExit("读标定失败")
    print("fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
          % (cam["fx"], cam["fy"], cam["cx"], cam["cy"]))
    print("地平线 v =", horizon_v_real(cam))

    print("\n=== 像素 → 地面 → 像素 ===")
    for (u, v) in [(960, 900), (960, 800), (600, 950), (1300, 850)]:
        r = pixel_to_ue_cm_real(cam, u, v)
        if r is None:
            print("  (%4d,%3d) 拒绝" % (u, v))
            continue
        uv, z, ok = body_to_pixel_real(cam, [[r["x_fwd_m"], r["y_left_m"], 0.0]])
        err = np.hypot(uv[0][0] - u, uv[0][1] - v) if ok[0] else -1
        print("  (%4d,%3d) → 前%.1fm 左%+.2fm → 回投 %.1f,%.1f (误差 %.3f)"
              % (u, v, r["x_fwd_m"], r["y_left_m"],
                 uv[0][0], uv[0][1], err))
