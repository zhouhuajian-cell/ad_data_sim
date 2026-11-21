"""GND V2：素材驱动 + 场景光照融合。

修两个问题：
  ① 不能指定素材 → 按 lane_type 从 /data/gnd_assets 取纹理
  ② 车道线太亮   → 按场景光照自适应亮度（不再硬编码 246）
"""
import glob
import json
import os

import cv2
import numpy as np

GND_ASSETS = "/data/gnd_assets"

# 素材类型 → 中文名
LANE_TYPES = {
    "lane_solid_white": "白色实线",
    "lane_dashed_white": "白色虚线",
    "lane_solid_yellow": "黄色实线",
    "road_edge_white": "路缘白线",
    "crosswalk_zebra": "斑马线",
}

_TEX_CACHE = {}


def load_texture(lane_type):
    """加载标线纹理。"""
    if lane_type in _TEX_CACHE:
        return _TEX_CACHE[lane_type]

    d = os.path.join(GND_ASSETS, lane_type)
    p = os.path.join(d, "tex.png")
    if not os.path.isfile(p):
        return None

    tex = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if tex is None:
        return None
    if tex.ndim == 2:
        tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    if tex.shape[2] < 4:
        # 无 alpha，用亮度当 alpha
        g = cv2.cvtColor(tex[..., :3], cv2.COLOR_BGR2GRAY)
        a = (g > 100).astype(np.uint8) * 255
        tex = np.dstack([tex[..., :3], a])

    _TEX_CACHE[lane_type] = tex
    return tex


def list_lane_types():
    """列出可用素材。"""
    out = []
    for k, name in LANE_TYPES.items():
        p = os.path.join(GND_ASSETS, k, "tex.png")
        if os.path.isfile(p):
            out.append({"key": k, "label": name, "path": p})
    return out


# ---------------- 场景光照融合 ----------------
def sample_scene_light(bg, seg, exclude_mask=None):
    """采样场景光照：路面亮度 + 色温。

    采样位置：车道线附近的路面（不是天空、不是车）。
    """
    h, w = bg.shape[:2]
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]

    # 有效采样区：下半部分 + 低饱和（路面）
    mask = np.zeros((h, w), np.uint8)
    mask[int(h * 0.45):, :] = 255
    # 排除车道线本身
    if seg is not None:
        for k in (1, 2, 3):
            mask[seg == k] = 0
    # 排除高饱和（车、树）
    mask[s > 90] = 0
    # 排除过暗（阴影）和过亮（天空）
    mask[v < 25] = 0
    mask[v > 245] = 0
    if exclude_mask is not None:
        mask[exclude_mask > 0] = 0

    if mask.sum() < 500:
        # 兜底：全图
        vals = v.reshape(-1)
        road_lum = float(np.median(vals))
        bgr_mean = bg.reshape(-1, 3).mean(axis=0)
    else:
        px = bg[mask > 0]
        road_lum = float(np.median(cv2.cvtColor(
            px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)[:, 2]))
        bgr_mean = px.mean(axis=0)

    # 色温比例（归一化到均值 1.0）
    m = max(float(bgr_mean.mean()), 1.0)
    color_gain = bgr_mean / m

    return {
        "road_lum": road_lum,
        "color_gain": color_gain.tolist(),
        "n_samples": int(mask.sum()),
    }


def adapt_line_color(tex_bgr, scene_light, reflectivity=1.95,
                     max_lum=232, min_lum=48):
    """按场景光照调整标线颜色。

    原理：真实标线的亮度 ≈ 路面亮度 × 反光率
      白线反光率约 2.0-2.5（比路面亮一倍多）
      黄线约 1.6-2.0

    关键：不硬编码纯白，而是跟着场景变。
    """
    road_lum = scene_light["road_lum"]
    gain = np.array(scene_light["color_gain"], np.float32)

    target_lum = float(np.clip(road_lum * reflectivity, min_lum, max_lum))

    # 纹理原本的亮度
    hsv = cv2.cvtColor(tex_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    cur_v = float(np.median(hsv[hsv[..., 2] > 40][:, 2])) \
        if (hsv[..., 2] > 40).any() else 200.0
    cur_v = max(cur_v, 1.0)

    # 亮度缩放
    k = target_lum / cur_v
    out = tex_bgr.astype(np.float32) * k

    # 色温
    out = out * gain[None, None, :]

    return np.clip(out, 0, 255).astype(np.uint8), target_lum


def apply_tex_alpha(tex, strength=1.0):
    """纹理 alpha × 强度。"""
    a = tex[..., 3].astype(np.float32) / 255.0
    return np.clip(a * strength, 0, 1)


# ---------------- 绘制 ----------------
def draw_lane_with_tex(bg, calib, cfg, seg=None, tex=None):
    """用纹理绘制车道线。"""
    from gnd_calib import ground_to_pixel

    lanes = int(cfg.get("lanes", 3))
    dash = cfg.get("dash", True)
    edge_w = float(cfg.get("edge", 0.20))
    line_w = float(cfg.get("width", 0.15))
    wear = float(cfg.get("wear", 0.25))
    opacity = float(cfg.get("opacity", 1.0))
    lane_type = cfg.get("lane_type", "lane_solid_white")
    edge_type = cfg.get("edge_type", "road_edge_white")
    center_type = cfg.get("center_type", "lane_dashed_white")
    reflectivity = float(cfg.get("reflectivity", 1.95))

    if tex is None:
        tex = load_texture(lane_type)
    if tex is None:
        # 兜底：无纹理用纯色
        tex = np.full((40, 40, 4), [200, 200, 200, 255], np.uint8)

    h, w = bg.shape[:2]

    # 场景光照
    light = sample_scene_light(bg, seg)
    tex_color, target_lum = adapt_line_color(tex[..., :3], light, reflectivity)

    # 累积 alpha 层
    alpha_acc = np.zeros((h, w), np.float32)

    lane_wid = 3.5
    total = lanes * lane_wid
    ys = [-total / 2 + i * lane_wid for i in range(lanes + 1)]
    DASH_ON, DASH_GAP = 6.0, 9.0

    for i, y_m in enumerate(ys):
        is_edge = (i == 0 or i == len(ys) - 1)
        wdt = edge_w if is_edge else line_w
        dashed = (not is_edge) and dash

        prev = None
        for x_m in np.arange(4.0, 55.0, 0.5):
            if dashed and (x_m % (DASH_ON + DASH_GAP)) > DASH_ON:
                prev = None
                continue
            uvl = ground_to_pixel(calib, x_m, y_m + wdt / 2)
            uvr = ground_to_pixel(calib, x_m, y_m - wdt / 2)
            if uvl is None or uvr is None:
                prev = None
                continue
            cur = (uvl, uvr, x_m)
            if prev is not None and abs(cur[2] - prev[2]) <= 1.0:
                q = np.array([prev[0], cur[0], cur[1], prev[1]], np.float32)
                if not np.all(np.isfinite(q)):
                    prev = cur
                    continue
                qi = np.rint(q).astype(np.int32)
                # 用纹理 alpha 填充（沿路面方向映射纹理 v 轴）
                tmp = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(tmp, qi, 255)
                if tex.shape[1] > 1:
                    # 简单：整块用纹理平均 alpha
                    tmp_a = apply_tex_alpha(tex).mean()
                else:
                    tmp_a = 1.0
                alpha_acc = np.maximum(alpha_acc, (tmp > 0).astype(np.float32) * tmp_a)
            prev = cur

    # 磨损
    if wear > 0.01:
        rng = np.random.default_rng(1234)
        low = rng.random((max(h // 8, 2), max(w // 8, 2))).astype(np.float32)
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
        low = cv2.GaussianBlur(low, (0, 0), max(min(h, w) * 0.02, 1.5))
        thr = 0.10 + 0.75 * wear
        wt = np.ones((h, w), np.float32)
        holes = low < thr
        wt[holes] = np.clip(low[holes] / max(thr, 1e-3), 0.0, 1.0)
        alpha_acc = alpha_acc * wt

    alpha_acc = np.clip(alpha_acc * opacity, 0, 1)

    # 合成（用适配后的颜色，不是纯白）
    out = (bg.astype(np.float32) * (1 - alpha_acc[..., None])
           + tex_color.astype(np.float32).mean(axis=(0, 1))[None, None, :]
           * alpha_acc[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, {
        "lane_type": lane_type,
        "target_lum": round(target_lum, 1),
        "road_lum": round(light["road_lum"], 1),
        "reflectivity": reflectivity,
        "color_gain": [round(v, 3) for v in light["color_gain"]],
        "n_samples": light["n_samples"],
    }


if __name__ == "__main__":
    print("GND 素材:")
    for x in list_lane_types():
        print("  %-22s %s" % (x["key"], x["label"]))
