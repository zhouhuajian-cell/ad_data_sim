"""OpenCV ↔ Blender 相机参数转换 + 3D 框投影 + 深度遮挡裁剪。

坐标系约定：
  世界系 (自动驾驶/Blender 一致): +X 前, +Y 左, +Z 上, 地面 Z=0
  OpenCV 相机系: +X 右, +Y 下, +Z 前
  Blender 相机系: +X 右, +Y 上, -Z 前
  转换: P_cv = diag(1, -1, -1) @ P_bl
"""
import math

import numpy as np


# ============================================================
# 内参
# ============================================================
def apply_opencv_intrinsics_to_blender(scene, cam_data, K, img_w, img_h,
                                       sensor_width_mm=36.0):
    """把 OpenCV 内参 K 精确同步到 Blender Camera。"""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    scene.render.resolution_x = int(img_w)
    scene.render.resolution_y = int(img_h)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy if fy else 1.0

    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = sensor_width_mm
    cam_data.lens = fx * sensor_width_mm / float(img_w)

    # Blender shift 语义（已用渲染 ground truth 实测验证）：
    #   实际渲染主点  cx_render = W/2 - shift_x * W
    # 要与 OpenCV cx 对齐：
    #   cx = W/2 - shift_x*W  =>  shift_x = (W/2 - cx)/W
    cam_data.shift_x = (img_w / 2.0 - cx) / float(img_w)
    # 垂直同理：cy_render = H/2 - shift_y * H  =>  shift_y = (H/2 - cy)/H
    cam_data.shift_y = (img_h / 2.0 - cy) / float(img_h)
    return {"lens_mm": cam_data.lens,
            "shift_x": cam_data.shift_x,
            "shift_y": cam_data.shift_y}


# ============================================================
# 外参
# ============================================================
def opencv_extrinsics_to_blender_matrix_world(T_world2cam_cv):
    """World->Cam(OpenCV) 转 Blender camera matrix_world (Cam_BL -> World)。"""
    T = np.asarray(T_world2cam_cv, dtype=np.float64)
    T_cam2world_cv = np.linalg.inv(T)
    cv_to_bl = np.diag([1.0, -1.0, -1.0, 1.0])
    return T_cam2world_cv @ cv_to_bl


def build_camera_from_ground_pose(cam_height_m=1.5, pitch_deg=0.0,
                                  roll_deg=0.0, yaw_deg=0.0,
                                  cam_xyz=None):
    """只有高度/姿态角时，构造 World->OpenCV Camera 外参。

    pitch_deg > 0 表示向下俯视。
    """
    R_base = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    y = math.radians(yaw_deg)

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(p), -math.sin(p)],
                   [0, math.sin(p), math.cos(p)]], dtype=np.float64)
    Ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]], dtype=np.float64)
    Rz = np.array([[math.cos(r), -math.sin(r), 0],
                   [math.sin(r), math.cos(r), 0],
                   [0, 0, 1]], dtype=np.float64)

    R_w2c = Rz @ Rx @ Ry @ R_base
    cam_pos = np.array(cam_xyz if cam_xyz is not None
                       else [0.0, 0.0, cam_height_m], dtype=np.float64)
    t_w2c = -R_w2c @ cam_pos

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_w2c
    T[:3, 3] = t_w2c
    return T


def set_blender_camera(scene, cam_obj, K, T_world2cam_cv, img_w, img_h):
    """一步设置 Blender 相机内外参。"""
    from mathutils import Matrix
    info = apply_opencv_intrinsics_to_blender(
        scene, cam_obj.data, K, img_w, img_h)
    M_bl = opencv_extrinsics_to_blender_matrix_world(T_world2cam_cv)
    cam_obj.matrix_world = Matrix(M_bl.tolist())
    return info


# ============================================================
# 3D 框投影
# ============================================================
def project_3d_box_opencv(K, T_world2cam_cv, center_xyz, size_lwh, yaw_rad):
    """世界系 3D 框投影到像素坐标 (8,2)。

    center_xyz: 车辆【底面中心】(x, y, 0)
    size_lwh:   (l, w, h)
    车头朝 +X（局部），yaw 绕 Z 轴。
    """
    l, w, h = size_lwh
    cx, cy, cz = center_xyz

    x_c = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_c = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_c = [0.0, 0.0, 0.0, 0.0, h, h, h, h]
    local = np.vstack([x_c, y_c, z_c])            # (3,8)

    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    corners_w = R_yaw @ local + np.array([[cx], [cy], [cz]])   # (3,8)

    corners_h = np.vstack([corners_w, np.ones((1, 8))])
    corners_c = (np.asarray(T_world2cam_cv) @ corners_h)[:3, :]  # (3,8)

    uv_h = np.asarray(K) @ corners_c
    uv = (uv_h[:2, :] / uv_h[2:3, :]).T
    return uv                                     # (8,2) 像素


BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7),
             (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]


# ============================================================
# 深度遮挡裁剪
# ============================================================
def read_exr_depth(exr_path):
    """读 EXR 深度（用 imageio 或 OpenCV），返回 (H,W) float32 米。"""
    try:
        import cv2
        d = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR)
        if d is not None:
            if d.ndim == 3:
                d = d[:, :, 0]
            return d.astype(np.float32)
    except Exception:
        pass
    try:
        import imageio.v3 as iio
        d = iio.imread(exr_path)
        if d.ndim == 3:
            d = d[:, :, 0]
        return d.astype(np.float32)
    except Exception:
        return None


def occlude_by_real_depth(fg_rgba, synth_depth, real_depth,
                          tolerance_m=0.5, real_valid_mask=None):
    """用真实场景深度裁掉被真实物体挡住的部分。

    fg_rgba:          (H,W,4) 合成前景（合成车辆）
    synth_depth:      (H,W)   合成车辆深度（米，无车处为 0 或 inf）
    real_depth:       (H,W)   真实场景深度（米，0/inf 表示无数据）
    tolerance_m:      容差（大于该差距认定为被遮挡）
    real_valid_mask:  (H,W) bool，标记真实深度可信区域（无 LiDAR 时可传 None，
                      此时退化为只用合成深度做接地判断）

    返回: (occluded_rgba, occluded_mask)
    """
    h, w = fg_rgba.shape[:2]
    alpha = fg_rgba[:, :, 3] > 0
    sd = np.asarray(synth_depth, dtype=np.float32)
    rd = np.asarray(real_depth, dtype=np.float32)

    # 合成深度有效范围
    sd_valid = (sd > 1e-6) & np.isfinite(sd)
    if real_valid_mask is not None:
        rd_valid = real_valid_mask
    else:
        rd_valid = (rd > 1e-6) & np.isfinite(rd)

    # 真实物体比合成车辆更近 -> 被遮挡
    occluded = np.zeros((h, w), dtype=bool)
    both = alpha & sd_valid & rd_valid
    occluded[both] = rd[both] < (sd[both] - tolerance_m)

    out = fg_rgba.copy()
    out[occluded, 3] = 0

    occluded_mask = (occluded & alpha).astype(np.uint8) * 255
    return out, occluded_mask


def composite_with_shadow(fg_rgba, bg_bgr, shadow_gain=1.0):
    """把 RGBA 前景叠到背景上（含阴影）。

    前景的阴影已经烘焙在 RGB 里（Shadow Catcher），alpha 只标车辆本体，
    所以需要额外处理“只有阴影没有车”的区域。
    """
    import cv2
    h, w = bg_bgr.shape[:2]
    if fg_rgba.shape[:2] != (h, w):
        fg_rgba = cv2.resize(fg_rgba, (w, h), interpolation=cv2.INTER_AREA)

    a = fg_rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = fg_rgba[:, :, :3].astype(np.float32)
    bgf = bg_bgr.astype(np.float32)

    # 车辆本体：正常 alpha 混合
    out = fg * a + bgf * (1.0 - a)

    # 阴影区域：alpha=0 但前景 RGB 非黑（Shadow Catcher 输出了阴影色）
    shadow_zone = (a[:, :, 0] < 0.02)
    fg_lum = fg.mean(axis=2)
    bg_lum = bgf.mean(axis=2)
    # 阴影处的 fg 是暗的（被遮光），用乘法叠加
    darken = np.clip(fg_lum / np.maximum(bg_lum, 1.0), 0.0, 1.0)
    shadow_mask = shadow_zone & (darken < 0.98)
    for c in range(3):
        ch = out[:, :, c]
        ch[shadow_mask] = bgf[:, :, c][shadow_mask] * (darken[shadow_mask] ** shadow_gain)
        out[:, :, c] = ch

    return np.clip(out, 0, 255).astype(np.uint8)


def verify_box_alignment(fg_rgba, corners_px, tol_px=2.0):
    """验证 3D 框投影与渲染车身边缘是否吻合。

    做法：在 alpha>0 的车辆掩码上，对每条框边做采样，
    统计框边点到最近车辆前景像素的距离。
    """
    import cv2
    alpha = (fg_rgba[:, :, 3] > 127).astype(np.uint8)
    if alpha.sum() == 0:
        return {"error": "empty foreground"}

    # 车辆前景的距离变换
    inv = (1 - alpha).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    errors = []
    for i, j in BOX_EDGES:
        p1 = np.array(corners_px[i])
        p2 = np.array(corners_px[j])
        n = max(2, int(np.linalg.norm(p2 - p1)))
        for t in np.linspace(0, 1, n):
            pt = p1 + t * (p2 - p1)
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= y < dist.shape[0] and 0 <= x < dist.shape[1]:
                errors.append(dist[y, x])

    errors = np.array(errors)
    return {
        "mean_err_px": float(errors.mean()),
        "max_err_px": float(errors.max()),
        "p90_err_px": float(np.percentile(errors, 90)),
        "within_tol_ratio": float((errors <= tol_px).mean()),
        "tol_px": tol_px,
        "pass": bool(errors.mean() <= tol_px),
    }


# ============================================================
# ROI 局部渲染 + Cycles 加速
# ============================================================
def set_blender_render_roi(scene, uv_corners, img_w, img_h,
                           shadow_pad_ratio=0.65, min_pad_px=80):
    """按 3D 框的 2D 投影自动开局部 ROI 渲染。

    关键：use_border=True 但 use_crop_to_border=False
      -> Cycles 只算 ROI 窗口，但输出仍是完整 WxH 画布
    """
    uv = np.asarray(uv_corners)
    u_min, u_max = float(uv[:, 0].min()), float(uv[:, 0].max())
    v_min, v_max = float(uv[:, 1].min()), float(uv[:, 1].max())
    bw = max(u_max - u_min, 1.0)
    bh = max(v_max - v_min, 1.0)

    pad_x = max(bw * shadow_pad_ratio, float(min_pad_px))
    pad_top = max(bh * 0.20, 30.0)
    pad_bottom = max(bh * 0.50, float(min_pad_px))

    x1 = max(0.0, u_min - pad_x)
    x2 = min(float(img_w), u_max + pad_x)
    y1 = max(0.0, v_min - pad_top)
    y2 = min(float(img_h), v_max + pad_bottom)

    scene.render.use_border = True
    scene.render.use_crop_to_border = False
    scene.render.border_min_x = x1 / img_w
    scene.render.border_max_x = x2 / img_w
    # OpenCV Y 向下 -> Blender 归一化 Y 向上
    scene.render.border_min_y = 1.0 - (y2 / img_h)
    scene.render.border_max_y = 1.0 - (y1 / img_h)

    return {"border": [round(scene.render.border_min_x, 4),
                       round(scene.render.border_max_x, 4),
                       round(scene.render.border_min_y, 4),
                       round(scene.render.border_max_y, 4)],
            "roi_px": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "area_ratio": round((x2 - x1) * (y2 - y1) / (img_w * img_h), 4)}


def optimize_cycles_for_batch(scene, samples=24):
    """批量生产用的 Cycles 加速配置（配合 OPTIX Denoiser）。"""
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.transparent_max_bounces = 4
    return {"samples": samples, "adaptive": True,
            "max_bounces": 4, "denoise": True}


def load_blender_depth_exr_as_npy(exr_path, npy_path):
    """在 Blender 内调用：读 EXR -> flipud -> 0 值转 inf -> 存 npy。

    ROI 模式下 ROI 外深度为 0.0，需转成 inf 避免误判为超近障碍。
    """
    import bpy
    img = bpy.data.images.load(exr_path)
    w, h = img.size
    px = np.array(list(img.pixels), dtype=np.float32).reshape(h, w, 4)
    z = np.flipud(px[:, :, 0]).copy()
    z[z <= 0.0] = np.inf
    np.save(npy_path, z)
    bpy.data.images.remove(img)
    valid = z[np.isfinite(z)]
    return {"shape": list(z.shape),
            "valid_px": int(valid.size),
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None}


# ============================================================
# 真车实例深度估计（射线与地面 Z=0 求交，共用同一套外参）
# ============================================================
def estimate_instance_depth_from_ground(mask_bool, K, T_world2cam_cv,
                                        ground_z=0.0):
    """用射线-地面求交估计实例每个像素的地面距离（米）。

    严格用相机内外参，不硬编码 cy。
    对 mask 内每个像素 (u,v)：
      ray_cam = K^-1 @ [u,v,1]
      ray_world = R^T @ ray_cam   (R = T[:3,:3])
      t = (ground_z - C_z) / ray_world_z
      P_world = C + t * ray_world
      dist = ||P_world - C||
    """
    import numpy as np

    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T_world2cam_cv, dtype=np.float64)
    Kinv = np.linalg.inv(K)

    R_w2c = T[:3, :3]
    t_w2c = T[:3, 3]
    R_c2w = R_w2c.T
    C = -R_c2w @ t_w2c          # 相机世界坐标

    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None, None

    uv1 = np.stack([xs, ys, np.ones_like(xs)], axis=0)   # (3,N)
    ray_c = Kinv @ uv1                                   # 相机系方向
    ray_w = R_c2w @ ray_c                                # 世界系方向

    denom = ray_w[2, :]
    ok = np.abs(denom) > 1e-9
    out = np.full(len(ys), np.nan, dtype=np.float32)
    if not ok.any():
        return out, np.stack([xs, ys], axis=1)

    t = np.full(len(ys), np.nan, dtype=np.float64)
    t[ok] = (ground_z - C[2]) / denom[ok]
    good = ok & (t > 0.5) & (t < 300.0)

    P = np.stack([C[0] + t * ray_w[0, :],
                  C[1] + t * ray_w[1, :],
                  C[2] + t * ray_w[2, :]], axis=1)
    dist = np.linalg.norm(P - C, axis=1)

    out[good] = dist[good].astype(np.float32)
    return out, np.stack([xs, ys], axis=1)


def instance_distance_by_percentile(mask_bool, K, T_world2cam_cv,
                                    percentile=95.0):
    """取实例内距离的 P95（最远处）稳住脚点噪声。"""
    import numpy as np
    dists, px = estimate_instance_depth_from_ground(mask_bool, K,
                                                    T_world2cam_cv)
    if dists is None:
        return None
    v = dists[np.isfinite(dists)]
    if v.size == 0:
        return None
    return {"dist_m": float(np.percentile(v, percentile)),
            "dist_median_m": float(np.median(v)),
            "n_px": int(v.size)}


def compute_horizon_v(K, T_world2cam_cv, ground_z=0.0, img_h=1080):
    """计算当前外参下 Z=0 平面的地平线像素纵坐标（远处极限）。"""
    import numpy as np
    T = np.asarray(T_world2cam_cv, dtype=np.float64)
    R_w2c = T[:3, :3]
    # 相机前向轴（世界系）= R^T @ [0,0,1]
    fwd_w = R_w2c.T @ np.array([0.0, 0.0, 1.0])
    # 前向若朝上，则无地平线
    if fwd_w[2] >= -1e-6:
        return None
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    C = -R_w2c.T @ T[:3, 3]
    far = C + fwd_w * 10000.0
    far[2] = ground_z
    d = far - C
    ray_c = R_w2c @ d
    if abs(ray_c[2]) < 1e-9:
        return None
    uv = np.asarray(K) @ ray_c
    v = uv[1] / uv[2]
    return float(v)


def estimate_pitch_from_horizon_v(horizon_v, K, img_h):
    """由地平线像素位置反推俯仰角（度，正=下俯）。"""
    import math
    cy = float(K[1, 2])
    fy = float(K[1, 1])
    # horizon_v < cy 表示地平线在中心上方 -> 相机下俯
    tan_pitch = (cy - horizon_v) / fy
    return math.degrees(math.atan(tan_pitch))
