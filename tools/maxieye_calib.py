"""读 Maxieye 结构化数据的相机标定。

数据布局：
  <dataset>/parameter_yaml/<date>/intrinsic/camera-<n>.yaml
  <dataset>/parameter_yaml/<date>/extrinsic/camera-<n>_to_body-2.yaml
  <dataset>/img_rectified/cam_st_x2f_front/*.jpg   (2912×1184，已去畸变)
  <dataset>/img_rectified/cam_st_x8d_front/*.jpg   (7200×2400)

坐标系约定（Maxieye）：
  body：x 前、y 左、z 上
  camera_to_body 的 transformation 把「相机坐标点」变到「body 坐标点」
  camera 系（pinhole）：x 右、y 下、z 前

因此 pixon 投影：
  点 p_body → p_cam = R^T @ (p_body - t) → u = cx + fx*X/Z, v = cy + fy*Y/Z
"""
import glob
import json
import os

import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


DATA_ROOT = ("/mnt/Data_Platform/files/MaxieyeStructuredData/"
             "GS4-002_2M_8M_GND/GS4-002")

# 相机目录 ↔ 标定编号
CAMERA_MAP = {
    "cam_st_x2f_front": "camera-1",
    "cam_st_x8d_front": "camera-7",
}


def _read_yaml(path):
    if not HAS_YAML:
        raise ImportError("需要 pyyaml: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_dataset(name_hint="141705"):
    """找数据集目录。"""
    for p in sorted(glob.glob(os.path.join(DATA_ROOT, "*"))):
        if os.path.basename(p).startswith(name_hint):
            return p
    return None


def latest_param_dir(dataset):
    """parameter_yaml 下最新日期目录。"""
    pd = os.path.join(dataset, "parameter_yaml")
    if not os.path.isdir(pd):
        return None
    ds = sorted([d for d in os.listdir(pd)
                 if os.path.isdir(os.path.join(pd, d))])
    if not ds:
        return None
    return os.path.join(pd, ds[-1])


def load_camera(dataset, cam_dir="cam_st_x2f_front"):
    """读一个相机的内参 + 外参。

    返回 dict:
      fx, fy, cx, cy     —— 用去畸变后的内参（配 img_rectified）
      width, height      —— 去畸变图尺寸
      R, t               —— camera→body（4×4 齐次里的 R, t）
      distortion         —— 原始畸变（配原图时用）
      K_raw, W_raw, H_raw—— 去畸变前
    """
    cid = CAMERA_MAP.get(cam_dir)
    if cid is None:
        raise ValueError("未知相机目录 %s（已知 %s）"
                         % (cam_dir, list(CAMERA_MAP)))

    pdir = latest_param_dir(dataset)
    if pdir is None:
        raise FileNotFoundError("找不到 parameter_yaml")

    ip = os.path.join(pdir, "intrinsic", cid + ".yaml")
    ep = os.path.join(pdir, "extrinsic", "body-2_to_" + cid + ".yaml")
    if not os.path.isfile(ep):
        ep = os.path.join(pdir, "extrinsic", cid + "_to_body-2.yaml")

    inn = _read_yaml(ip)
    ext = _read_yaml(ep)

    # 内参：去畸变后的
    Kd = np.array(inn.get("undistort_intrinsic")
                  or inn["intrinsic"], float)
    W = int(inn.get("undistort_width") or inn["width"])
    H = int(inn.get("undistort_height") or inn["height"])

    Kr = np.array(inn["intrinsic"], float)
    Wr = int(inn["width"])
    Hr = int(inn["height"])
    dist = np.array(inn.get("distortion") or [], float).ravel()

    # 外参
    T = np.array(ext["transformation"], float).reshape(4, 4)
    fname = os.path.basename(ep)
    if fname.startswith("body-2_to_"):
        # body→camera，取逆
        T = np.linalg.inv(T)

    R = T[:3, :3]
    t = T[:3, 3]

    return {
        "camera_id": cid,
        "cam_dir": cam_dir,
        "fx": float(Kd[0, 0]), "fy": float(Kd[1, 1]),
        "cx": float(Kd[0, 2]), "cy": float(Kd[1, 2]),
        "K": Kd.tolist(),
        "width": W, "height": H,
        "K_raw": Kr.tolist(), "raw_width": Wr, "raw_height": Hr,
        "distortion": dist.tolist(),
        "R": R.tolist(), "t": t.tolist(),
        "T": T.tolist(),
        "source": ip + " | " + ep,
        "reproj_err": (inn.get("meta") or {}).get("reproj_err"),
    }


def body_to_pixel(pts_body, cam):
    """body 坐标点 (N,3) → 像素 (N,2)、相机系深度 (N,)。"""
    P = np.asarray(pts_body, float).reshape(-1, 3)
    R = np.array(cam["R"], float)
    t = np.array(cam["t"], float)
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]

    Pc = (P - t) @ R            # body→camera: p_cam = R^T (p_body - t)
    z = Pc[:, 2]
    ok = z > 1e-6
    uv = np.full((len(P), 2), np.nan)
    uv[ok, 0] = cx + fx * Pc[ok, 0] / z[ok]
    uv[ok, 1] = cy + fy * Pc[ok, 1] / z[ok]
    return uv, z, ok


def ground_homography(cam):
    """body 地面 (x 前, y 左, z=0) → 像素的单应。

    返回 H (3×3)，作用在 (x, y, 1)。
    """
    K = np.array(cam["K"], float)
    R = np.array(cam["R"], float)
    t = np.array(cam["t"], float)
    # p_cam = R^T (p_body - t)；地面 z=0
    # p_cam = R^T [x, y, 0] - R^T t
    #       = R^T[:, :2] [x,y] + (-R^T t)
    Rt = R.T
    A = Rt[:, :2]
    b = -Rt @ t
    H = K @ np.column_stack([A, b])
    return H


def pixel_to_ground(cam, u, v):
    """像素 → body 地面点 (x 前, y 左, z=0)。"""
    H = ground_homography(cam)
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


def frame_time(cam_dir, dataset):
    """取一张代表性帧的时间戳（用于对齐）。"""
    d = os.path.join(dataset, "img_rectified", cam_dir)
    if not os.path.isdir(d):
        return None
    fs = sorted(glob.glob(os.path.join(d, "*.jpg")))
    if not fs:
        return None
    return os.path.splitext(os.path.basename(fs[0]))[0]


def list_frames(dataset, cam_dir):
    d = os.path.join(dataset, "img_rectified", cam_dir)
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "*.jpg")))


if __name__ == "__main__":
    import sys
    hint = sys.argv[1] if len(sys.argv) > 1 else "141705"
    ds = find_dataset(hint)
    if not ds:
        raise SystemExit("找不到数据集 " + hint)
    print("数据集:", ds)

    for cd in ("cam_st_x2f_front", "cam_st_x8d_front"):
        print("\n=== %s ===" % cd)
        try:
            c = load_camera(ds, cd)
        except Exception as e:
            print("  失败:", e)
            continue
        print("  相机编号: %s" % c["camera_id"])
        print("  fx=%.2f fy=%.2f cx=%.2f cy=%.2f  %dx%d"
              % (c["fx"], c["fy"], c["cx"], c["cy"],
                 c["width"], c["height"]))
        print("  reproj_err = %s px" % c["reproj_err"])
        print("  t (相机在 body 的位置) = [%.3f, %.3f, %.3f]"
              % tuple(c["t"]))
        n = len(list_frames(ds, cd))
        print("  帧数: %d" % n)

        # 投影测试
        for (x, y) in [(10, 0), (20, 0), (10, -2), (10, 2), (30, 0)]:
            uv, z, ok = body_to_pixel([[x, y, 0.0]], c)
            if ok[0]:
                print("    地面 (%2d, %+d) → 像素 (%.0f, %.0f)  z=%.1fm"
                      % (x, y, uv[0][0], uv[0][1], z[0]))
            else:
                print("    地面 (%2d, %+d) → 不可见" % (x, y))
