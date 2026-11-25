#!/usr/bin/env python3
"""GND 真值标定解析（camera-1.yaml + camera-1_to_body-1.yaml）。"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, "/opt/ue5_synth/tools")
from data_guard import guard_all_mnt_ops

guard_all_mnt_ops()

CALIB_DIRS = [
    "/data/test_data/gnd_train/{batch}/{clip}/parameter_yaml",
    "/data/test_data/gnd_clips/{batch}/{clip}/parameter_yaml",
    "/mnt/done_data_new/31_ex_GND/05_maxieye/{batch}/{clip}/parameter_yaml",
]


def find_calib(batch, clip):
    """找标定目录。"""
    for pat in CALIB_DIRS:
        root = pat.format(batch=batch, clip=clip)
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if os.path.isdir(os.path.join(p, "intrinsic")):
                return p
    return None


def load_gnd_calib(batch, clip):
    """读 GND 标定。"""
    cdir = find_calib(batch, clip)
    if not cdir:
        return None

    # 内参
    ip = os.path.join(cdir, "intrinsic", "camera-1.yaml")
    if not os.path.isfile(ip):
        return None
    with open(ip, encoding="utf-8") as f:
        ic = yaml.safe_load(f)

    K = np.array(ic["intrinsic"], float)
    dist = np.array(ic.get("distortion") or [], float).ravel()

    # 外参必须明确存在，不能以单位矩阵冒充有效标定。
    import glob
    exts = glob.glob(os.path.join(cdir, 'extrinsic', 'camera-1_to_body-*.yaml'))
    if len(exts) != 1:
        raise ValueError('缺少或存在多个 camera-to-body 外参: %s' % cdir)
    with open(exts[0], encoding='utf-8') as f:
        ec = yaml.safe_load(f)
    T = np.array(ec['transformation'], float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('camera-to-body 外参无效')
    R_cb, t_cb = T[:3, :3], T[:3, 3]
    if not np.allclose(R_cb.T @ R_cb, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R_cb), 1.0, atol=1e-3):
        raise ValueError('camera-to-body 旋转矩阵无效')

    return {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "K": K.tolist(),
        "distortion": dist.tolist(),
        "width": int(ic.get("width") or 1920),
        "height": int(ic.get("height") or 1080),
        "reproj_err": float((ic.get("meta") or {}).get("reproj_err") or 0),
        "vehicle": (ic.get("meta") or {}).get("vehicle"),
        "R_cam2body": R_cb.tolist(),
        "t_cam2body": t_cb.tolist(),
        "calib_dir": cdir,
    }


def body_to_pixel_gnd(calib, pts_body):
    """body 系点 (x前, y左, z上) → 像素。

    相机朝向：camera-1_to_body-1 显示相机 x 朝车体 -y（左转 90°）
    实际用 R_cam2body 的逆变换。
    """
    K = np.array(calib["K"], float)
    R = np.array(calib["R_cam2body"], float)
    t = np.array(calib["t_cam2body"], float)

    P = np.asarray(pts_body, float).reshape(-1, 3)
    # body → camera:  p_cam = R^T @ (p_body - t)
    # 因为时 T 是 camera→body，所以 body→camera = T^-1
    Pc = (R.T @ (P - t).T).T

    # 相机系 → 像素（假设相机 z 朝前）
    # Maxieye 约定：p_cam = [x右, y下, z前]
    z = Pc[:, 2]
    u = K[0, 0] * Pc[:, 0] / np.where(np.abs(z) < 1e-6, 1e-6, z) + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / np.where(np.abs(z) < 1e-6, 1e-6, z) + K[1, 2]
    ok = z > 1e-3
    return np.stack([u, v], 1), z, ok


def ground_to_pixel(calib, x_m, y_m, z_m=0.0):
    """地面点 → 像素（用真值外参）。返回 (u, v) 或 None。"""
    uv, z, ok = body_to_pixel_gnd(calib, [[x_m, y_m, z_m]])
    if not bool(ok[0]):
        return None
    u, v = float(uv[0][0]), float(uv[0][1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return (u, v)


if __name__ == "__main__":
    batch = "02_HS5-001_20260629"
    clip = "1782705797.755"

    c = load_gnd_calib(batch, clip)
    if c is None:
        print("本地无标定，试网盘")
        c = load_gnd_calib("02_HS5-001", "20260629/1782705797.755")

    if c is None:
        raise SystemExit("找不到标定")

    print("=== GND 真值标定 ===")
    print("  来源:      ", c["calib_dir"])
    print("  车辆:      ", c["vehicle"])
    print("  fx=%.3f  fy=%.3f  cx=%.3f  cy=%.3f"
          % (c["fx"], c["fy"], c["cx"], c["cy"]))
    print("  分辨率:    %dx%d" % (c["width"], c["height"]))
    print("  重投影误差: %.3f px" % c["reproj_err"])
    print("  畸变:      ", [round(v, 5) for v in c["distortion"]])
    print("\n  相机在车体的位置: "
          + str([round(v, 3) for v in c["t_cam2body"]]))

    print("\n=== 地面点投影测试 ===")
    print("  %-16s %s" % ("世界坐标(m)", "像素"))
    for x, y in [(5, 0), (10, 0), (20, 0), (10, 3.5), (10, -3.5),
                 (30, 5), (50, 0)]:
        uv = ground_to_pixel(c, x, y)
        if uv is None:
            print("  (%5.1f,%5.1f)  不可投影" % (x, y))
        else:
            print("  (%5.1f,%5.1f)  ->  (%7.1f, %6.1f)" % (x, y, uv[0], uv[1]))
