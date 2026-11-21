"""用 UE 深度图做精确遮挡（替代 HSV 猜测）。

输入：
  depth.png    R 通道 = 线性深度 / 200m（由 render_semantic_depth.py 产出）
  inst.png     R 通道 = 实例 ID / 255
输出：
  z_real      每像素真实深度（米），无效区为 nan
  visible     合成物体是否可见（z_synth < z_real）

相比 ground_masks.foreground_mask() 的 HSV 猜法：
  - 银色/白色车不再漏检
  - 遮挡率是真实数值，可以标注
"""
import cv2
import numpy as np

DEPTH_SCALE_M = 200.0     # render_semantic_depth.py 的归一化上限


def load_depth(path, scale_m=DEPTH_SCALE_M):
    """读深度图 → 每像素真实距离（米）。0 值视为无效。"""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)

    if im.ndim == 3:
        ch = im[..., 0]                   # 深度在 R 通道
    else:
        ch = im

    if ch.dtype == np.uint16:
        d = ch.astype(np.float32) / 65535.0
    else:
        d = ch.astype(np.float32) / 255.0

    z = d * scale_m
    z[z <= 1e-3] = np.nan                 # 0 = 无几何（天空/未渲染）
    return z


def load_instance(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    ch = im[..., 0] if im.ndim == 3 else im
    if ch.dtype == np.uint16:
        return (ch.astype(np.float32) / 65535.0 * 255.0).round().astype(np.int32)
    return ch.astype(np.int32)


def synth_depth_from_quad(quad, z_synth_m, shape):
    """把「物体在某个深度」这一事实铺成深度图。

    2.5D 精灵是平面的，所以整个物体用同一个 z。
    如果有多个物体，按 Z 从远到近铺。
    """
    h, w = shape
    zs = np.full((h, w), np.nan, np.float32)
    if quad is None:
        return zs
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, np.rint(np.asarray(quad)).astype(np.int32), 255)
    zs[m > 0] = float(z_synth_m)
    return zs


def depth_visibility(z_synth, z_real, margin_m=0.05):
    """返回 (visible, uncertain)。

    visible   合成物体确实在真实几何之前
    uncertain 真实深度无效（天空/未渲染区），无法判定 —— 不能当"可见"
    """
    if z_synth.shape != z_real.shape:
        raise ValueError("深度尺寸不一致 %s vs %s"
                         % (z_synth.shape, z_real.shape))

    has_synth = np.isfinite(z_synth)
    has_real = np.isfinite(z_real)

    visible = has_synth & has_real & (z_synth + margin_m < z_real)
    uncertain = has_synth & ~has_real
    return visible.astype(np.float32), uncertain.astype(np.float32)


def occlusion_stats(visible, uncertain, alpha=None):
    """统计遮挡率，用于写回标注。"""
    if alpha is None:
        total = float((visible > 0.5).sum() + (uncertain > 0.5).sum())
        vis = float((visible > 0.5).sum())
    else:
        a = np.clip(alpha, 0, 1)
        total = float(a.sum())
        vis = float((a * visible).sum())

    if total <= 1e-6:
        return {"occlusion_in_frame": None, "visible_ratio": None,
                "reason": "no_synthetic_pixels"}

    return {
        "occlusion_in_frame": round(1.0 - vis / total, 4),
        "visible_ratio": round(vis / total, 4),
        "uncertain_pixels": int((uncertain > 0.5).sum()),
    }


def compose_with_depth(bg_bgr, sprite_bgr, alpha, visible):
    """只在 visible 区域合成精灵。"""
    a = np.clip(alpha, 0, 1) * np.clip(visible, 0, 1)
    out = (bg_bgr.astype(np.float32) * (1 - a[..., None])
           + sprite_bgr.astype(np.float32) * a[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), a


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "/data/render/semantic/depth.png"
    try:
        z = load_depth(p)
        ok = np.isfinite(z)
        print("深度图:", p)
        print("  有效像素: %.1f%%" % (ok.mean() * 100))
        if ok.any():
            print("  距离范围: %.2f ~ %.2f m" % (np.nanmin(z), np.nanmax(z)))
            print("  中位数:   %.2f m" % np.nanmedian(z))
    except Exception as e:
        print("失败:", e)
