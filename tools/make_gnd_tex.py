import glob
import os

import cv2
import numpy as np


def paint_texture(length_m, width_m, ppm=200, seed=0, wear=0.15,
                  rgb=(215, 213, 201), dashed=None, edge_wear=True):
    """生成一条标线的真实磨损纹理（RGBA）。

    length_m/width_m: 物理尺寸（米）
    ppm: 每米像素数
    wear: 0~1 磨损程度
    dashed: (dash_m, gap_m) 虚线
    """
    rng = np.random.default_rng(seed)
    h = max(4, int(round(length_m * ppm)))
    w = max(4, int(round(width_m * ppm)))
    if h * w > 40_000_000:
        raise ValueError("texture too large")

    # 1) 低频损坏场（成片的剥落）
    ch = max(3, h // 40)     # 沿长度方向更大块
    cw = max(3, w // 2)
    coarse = rng.random((ch, cw)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    field = np.clip(field, 0, 1)

    # 2) 高频颗粒（颜料不匀）
    fine = rng.random((h, w)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (3, 3), 0)
    # 表面纹理用更低频的场，避免"木纹"感
    surface = cv2.GaussianBlur(fine, (0, 0), max(2.0, w * 0.10))

    # 3) 磨损：低频场做大块剥落，高频只做轻微表面不匀
    density = 0.92 * field + 0.08 * surface
    # 白漆默认满覆盖；只有密度最低的零星区域剥落。
    # wear=0 → 几乎不掉漆；wear=1 → 掉漆面积约 40%
    # wear 单调：wear 越大阈值越高 → 越容易掉漆（修正原来的反向）
    thr = 0.10 + 0.75 * wear
    alpha = np.ones((h, w), np.float32)
    holes = density < thr
    alpha[holes] = np.clip(density[holes] / max(thr, 1e-3), 0.0, 1.0)
    # 剥落边缘圆润（真实掉漆是块状，不是针孔）
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.2, w * 0.02))
    alpha = np.clip(alpha, 0, 1)

    # 4) 边缘磨损（两侧剥落更明显）
    if edge_wear:
        dist = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
        edge = np.clip(dist / max(1.0, w * 0.12), 0, 1)
        edge = edge[None, :] ** 0.6
        alpha *= edge

    # 5) 虚线
    if dashed is not None:
        dash, gap = dashed
        s = (np.arange(h) + 0.5) / ppm
        mask = ((s % (dash + gap)) < dash).astype(np.float32)[:, None]
        alpha *= mask

    # 6) 颜色微变化
    var = 0.955 + 0.045 * surface
    rgba = np.empty((h, w, 4), np.uint8)
    for i in range(3):
        rgba[..., i] = np.clip(rgb[i] * var, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.rint(alpha * 255).astype(np.uint8)
    return rgba


def save_all(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    specs = [
        # 颜色用 BGR（cv2 写 PNG 期望 BGR）
        ("lane_solid_white",   20.0, 0.15, (201, 213, 215), None,        0.12),
        ("lane_dashed_white",  20.0, 0.15, (201, 213, 215), (4.0, 6.0),  0.16),
        ("lane_solid_yellow",  20.0, 0.15, (22, 132, 178),  None,        0.11),
        ("lane_double_yellow", 20.0, 0.15, (22, 132, 178),  None,        0.10),
        ("stop_line",          10.0, 0.40, (201, 213, 215), None,        0.28),
        ("crosswalk_zebra",    10.0, 0.45, (201, 213, 215), None,        0.35),
        ("road_edge_white",    20.0, 0.20, (201, 213, 215), None,        0.15),
    ]
    made = []
    for i, (name, L, W, rgb, dashed, wear) in enumerate(specs):
        tex = paint_texture(L, W, ppm=200, seed=1000 + i, wear=wear,
                            rgb=rgb, dashed=dashed)
        p = os.path.join(out_dir, name + "_worn.png")
        cv2.imwrite(p, tex)
        made.append((name, tex.shape, p))

    # 拼一张预览
    tiles = []
    for name, shape, p in made:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        # 铺到灰底上看效果
        canvas = np.full((im.shape[0], im.shape[1], 3), 90, np.uint8)
        a = im[..., 3:4].astype(np.float32) / 255.0
        canvas = (im[..., :3] * a + canvas * (1 - a)).astype(np.uint8)
        tiles.append(cv2.resize(canvas, (200, 600)))

    grid = np.hstack(tiles)
    cv2.imwrite(os.path.join(out_dir, "_preview.jpg"), grid,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    print("生成 %d 张贴图：" % len(made))
    for name, shape, p in made:
        print("  %-22s %s" % (name, shape))
    print("\n预览: %s/_preview.jpg" % out_dir)


if __name__ == "__main__":
    save_all("/data/avsynth/Content/Validation/GND/textures")
