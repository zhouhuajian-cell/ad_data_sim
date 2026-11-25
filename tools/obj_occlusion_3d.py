#!/usr/bin/env python3
"""Conservative LiDAR/semantic-mask occlusion for synthetic OBJ renders.

All arrays must be in the same undistorted pinhole image space.  Unknown depth
inside a real foreground mask is treated as occluded (fail closed).
"""
import cv2
import numpy as np


def project_lidar(xyz, K, R, t, image_shape):
    """Project synchronized LiDAR XYZ; keep the nearest sample per pixel."""
    pts = np.asarray(xyz, np.float64)
    k = np.asarray(K, np.float64).reshape(3, 3)
    r = np.asarray(R, np.float64).reshape(3, 3)
    tv = np.asarray(t, np.float64).reshape(3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("xyz must have shape Nx3")
    h, w = image_shape[:2]
    pc = pts @ r.T + tv
    z = pc[:, 2]
    valid = np.isfinite(pc).all(axis=1) & (z > 0.05)
    u = np.zeros(len(pts), np.int32)
    v = np.zeros(len(pts), np.int32)
    u[valid] = np.rint(k[0, 0] * pc[valid, 0] / z[valid] +
                       k[0, 2]).astype(np.int32)
    v[valid] = np.rint(k[1, 1] * pc[valid, 1] / z[valid] +
                       k[1, 2]).astype(np.int32)
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[valid], v[valid], z[valid].astype(np.float32)
    depth = np.full((h, w), np.nan, np.float32)
    if len(z):
        order = np.argsort(z)[::-1]  # far first, nearest assignment wins
        depth[v[order], u[order]] = z[order]
    return depth


def foreground_depth(xyz, K, R, t, foreground_mask, max_fill_px=6.0):
    """Build sparse real-foreground depth and fill only locally inside mask."""
    mask = np.asarray(foreground_mask).astype(bool)
    sparse = project_lidar(xyz, K, R, t, mask.shape)
    sparse[~mask] = np.nan
    known = np.isfinite(sparse)
    out = sparse.copy()
    if not known.any():
        return out, {"samples": 0, "filled": 0}

    from scipy.spatial import cKDTree
    # 不跨不相连的前景区域传播深度：相邻车辆/行人的稀疏点不能互相填充。
    n, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    for component in range(1, n):
        region = components == component
        ky, kx = np.where(known & region)
        if not len(kx):
            continue
        hy, hx = np.where(region & ~known)
        if not len(hx):
            continue
        tree = cKDTree(np.column_stack((kx, ky)))
        distance, index = tree.query(np.column_stack((hx, hy)), k=1)
        fill = distance <= float(max_fill_px)
        out[hy[fill], hx[fill]] = sparse[ky[index[fill]], kx[index[fill]]]
    return out, {"samples": int(known.sum()),
                 "filled": int(np.isfinite(out).sum())}


def radial_to_camera_z(radial_depth, K):
    """Convert Blender Z-pass ray distance to OpenCV optical-axis depth."""
    d = np.asarray(radial_depth, np.float32)
    k = np.asarray(K, np.float64).reshape(3, 3)
    h, w = d.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    ray_norm = np.sqrt(1.0 + ((xx - k[0, 2]) / k[0, 0]) ** 2 +
                       ((yy - k[1, 2]) / k[1, 1]) ** 2)
    return d / ray_norm


def composite(bg_bgr, fg_rgba, synth_camera_z, real_depth,
              foreground_mask, tolerance_m=0.35, harmonizer=None):
    """Composite with real foreground occlusion and return diagnostics.

    Pixels in a real foreground mask are visible only when both depths are
    known and the synthetic surface is clearly closer.  This prevents a
    relaxed 2D placement gate from painting through real vehicles.
    """
    bg = np.asarray(bg_bgr)
    fg = np.asarray(fg_rgba)
    h, w = bg.shape[:2]
    if fg.shape[:2] != (h, w):
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    if fg.ndim != 3 or fg.shape[2] != 4:
        raise ValueError("fg_rgba must have four channels")
    sd = np.asarray(synth_camera_z, np.float32)
    rd = np.asarray(real_depth, np.float32)
    if sd.shape != (h, w):
        sd = cv2.resize(sd, (w, h), interpolation=cv2.INTER_NEAREST)
    if rd.shape != (h, w):
        raise ValueError("real_depth shape mismatch")

    mask = np.asarray(foreground_mask).astype(bool)
    if mask.shape != (h, w):
        raise ValueError("foreground_mask shape mismatch")
    alpha = fg[:, :, 3].astype(np.float32) / 255.0
    synth_z = sd
    # Blender 无命中背景可能是极大有限值而非 inf；必须设物理上限，
    # 否则透明背景/Shadow Catcher 会被误认成车体并撑大 BBox。
    sd_ok = np.isfinite(synth_z) & (synth_z > 0.05) & (synth_z < 200.0)
    rd_ok = np.isfinite(rd) & (rd > 0.05)
    permitted = (~mask) | (sd_ok & rd_ok &
                           (synth_z < rd - float(tolerance_m)))
    quality = {'applied':False}
    if harmonizer is not None:
        fg, quality = harmonizer.apply(bg, fg, (fg[:,:,3] > 8) & sd_ok & permitted, mask)
    alpha *= permitted.astype(np.float32)
    a = alpha[:, :, None]
    out = fg[:, :, :3].astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    original = fg[:, :, 3] > 8
    # Shadow Catcher 也写 alpha，但没有合成车辆深度。BBox/可见率只能
    # 统计 depth 有效的车体像素，否则框会扩到整片投影阴影。
    body = original & sd_ok
    visible = body & permitted
    if visible.any():
        vy, vx = np.where(visible)
        visible_bbox = [int(vx.min()), int(vy.min()),
                        int(vx.max()) + 1, int(vy.max()) + 1]
    else:
        visible_bbox = None
    return np.clip(out, 0, 255).astype(np.uint8), {
        "alpha_pixels": int(body.sum()),
        "visible_pixels": int(visible.sum()),
        "visible_ratio": float(visible.sum() / max(body.sum(), 1)),
        "unknown_foreground_hidden": int((body & mask & ~rd_ok).sum()),
        "visible_bbox": visible_bbox,
        "harmonization": quality,
    }
