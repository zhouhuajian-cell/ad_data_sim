"""OBJ 2.5D 精灵合成：用人工标定 H 把预渲染的 3D 物体贴到底图。

不走 UE 实时相机（UE 的 fov+pitch 复现不了人工标定 H）。
改为：
  1) UE 只渲染物体本体（透明背景，正交/固定视角，一次性资产）
  2) 用同一个 H 把物体按世界坐标缩放/投影到底图
  3) 保证「点击位置 == 最终落点」

这样 OBJ 和 GND 共享同一套几何，点哪里就放在哪里。
"""
import json
import os

import cv2
import numpy as np

try:
    from ground_masks import road_mask, foreground_mask, hood_mask
    HAS_MASKS = True
except ImportError:
    HAS_MASKS = False

SPRITE_DIR = "/data/render/obj_sprites"


def load_H(bg_name):
    stem = os.path.splitext(bg_name)[0]
    p = "/data/render_hybrid/bgs/_calib_" + stem + ".json"
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["H"], float).reshape(3, 3), float(d.get("cam_height_m") or 1.35)


def asset_size_m(kind):
    """各类物体的真实尺寸（宽, 高）米。"""
    return {
        "cone": (0.28, 0.70),
        "car": (1.80, 1.50),
        "suv": (1.90, 1.75),
        "truck": (2.50, 3.20),
        "pedestrian": (0.55, 1.70),
    }.get(kind, (0.5, 0.8))


def find_sprite(kind):
    """找预渲染的精灵图（RGBA）。"""
    for ext in (".png", ".jpg"):
        for name in ("obj_" + kind, kind, "SM_" + kind):
            p = os.path.join(SPRITE_DIR, name + ext)
            if os.path.isfile(p):
                return p
    return None


def render_sprite_fallback(kind, h=256):
    """没有预渲染精灵时，生成一个占位图（锥桶=橙色三角，车=灰块）。"""
    w = int(h * 0.62)
    im = np.zeros((h, w, 4), np.uint8)
    if kind == "cone":
        pts = np.array([[w // 2, 10], [w - 8, h - 6], [8, h - 6]], np.int32)
        cv2.fillPoly(im, [pts], (30, 130, 240, 255))
        cv2.rectangle(im, (4, h - 14), (w - 4, h - 4), (20, 20, 20, 255), -1)
    elif kind in ("car", "suv", "truck"):
        cv2.rectangle(im, (6, int(h * 0.30)), (w - 6, h - 8), (150, 150, 150, 255), -1)
        cv2.rectangle(im, (int(w * 0.18), int(h * 0.10)),
                      (int(w * 0.82), int(h * 0.36)), (110, 110, 110, 255), -1)
    else:
        cv2.rectangle(im, (int(w * 0.30), 20), (int(w * 0.70), int(h * 0.55)),
                      (90, 140, 200, 255), -1)
        cv2.rectangle(im, (int(w * 0.28), int(h * 0.55)), (int(w * 0.72), h - 8),
                      (60, 60, 160, 255), -1)
    return im


def project_point(H, Y, Z):
    p = H @ np.array([Y, Z, 1.0])
    if abs(p[2]) < 1e-12:
        return None
    return (p[0] / p[2], p[1] / p[2])


def project_footprint(H, Y, Z, w_m, h_m):
    """物体占地 → 像素四边形（底宽 + 顶宽）。"""
    # 底面四角
    foot = []
    for dy, dz in ((-w_m / 2, 0), (w_m / 2, 0)):
        p = project_point(H, Y + dy, Z + dz)
        if p is None:
            return None
        foot.append(p)

    # 顶部：用横向比例换算高度像素。
    # 关键：横向 1m 在 Z 处对应 px_per_m 像素；同一 Z 处垂直 1m
    # 对应的像素数与之同量级（针孔相机下近似相等）。
    # 之前用「往前挪 8% 距离」估算，得到的是纵向梯度，量纲不对，
    # 导致锥桶只有实际 1/4 大小。
    p_l = project_point(H, Y - 0.5, Z)
    p_r = project_point(H, Y + 0.5, Z)
    if p_l is None or p_r is None:
        return None
    px_per_m = abs(p_r[0] - p_l[0])     # 横向 1 米 = 多少像素
    top_dy = h_m * px_per_m

    tl = (foot[0][0], foot[0][1] - top_dy)
    tr = (foot[1][0], foot[1][1] - top_dy)
    br = foot[1]
    bl = foot[0]
    return np.array([tl, tr, br, bl], np.float32)


def composite(bg_bgr, H, placed, out_p, alpha_shadow=0.35):
    """把 placed 列表投影到底图。

    placed: [{"cls": "cone", "Y": 米, "Z": 米}, ...]
    """
    h, w = bg_bgr.shape[:2]
    out = bg_bgr.astype(np.float32)
    anns = []

    # 前景保护：真实车辆/行人/车头区域不允许被合成物体覆盖
    if HAS_MASKS:
        _road = road_mask(bg_bgr)
        _fg = foreground_mask(bg_bgr, _road)
        _hood = hood_mask(bg_bgr)
        protect = np.clip(np.maximum(_fg, _hood), 0, 1)
        print("前景保护: fg=%.4f hood=%.4f" % (_fg.mean(), _hood.mean()))
    else:
        protect = np.zeros((h, w), np.float32)

    # 按 Z 从远到近画，保证近处物体在上面
    items = []
    for p in placed:
        cls = str(p.get("cls") or "cone")
        Y = float(p.get("Y") or 0.0)
        Z = float(p.get("Z") or 8.0)
        items.append((Z, cls, Y))
    items.sort(key=lambda x: -x[0])

    for Z, cls, Y in items:
        w_m, h_m = asset_size_m(cls)
        quad = project_footprint(H, Y, Z, w_m, h_m)
        if quad is None:
            continue

        sp = find_sprite(cls)
        if sp and sp.endswith(".png"):
            sprite = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
            if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
                sprite = render_sprite_fallback(cls)
        else:
            sprite = render_sprite_fallback(cls)

        sh, sw = sprite.shape[:2]
        src = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32)
        Pm = cv2.getPerspectiveTransform(src, quad.astype(np.float32))

        a = sprite[..., 3].astype(np.float32) / 255.0
        rgb = sprite[..., :3].astype(np.float32)
        prem = rgb * a[..., None]

        a_w = cv2.warpPerspective(a, Pm, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        p_w = cv2.warpPerspective(prem, Pm, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        a_w = np.clip(a_w, 0, 1)
        straight = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
        straight[a_w <= 1e-4] = 0

        # 被真实前景挡住的部分不画
        keep = a_w * (1.0 - protect)
        blocked = float((a_w.sum() - keep.sum()) / max(a_w.sum(), 1e-6))
        a_w = keep

        # 接地阴影
        base = quad.astype(np.int32)
        sh_mask = np.zeros((h, w), np.float32)
        cx = int((base[2][0] + base[3][0]) / 2)
        cy = int((base[2][1] + base[3][1]) / 2)
        rw = max(int(abs(base[1][0] - base[0][0]) * 0.6), 6)
        rh = max(int(rw * 0.28), 3)
        cv2.ellipse(sh_mask, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)
        sh_mask = cv2.GaussianBlur(sh_mask, (15, 15), 0) * alpha_shadow
        out = out * (1 - sh_mask[..., None])

        # 合成
        out = out * (1 - a_w[..., None]) + straight * a_w[..., None]

        xs = quad[:, 0]
        ys = quad[:, 1]
        anns.append({
            "class": cls,
            "bbox": [float(xs.min()), float(ys.min()),
                     float(xs.max()), float(ys.max())],
            "world": {"Y_m": round(Y, 3), "Z_m": round(Z, 3)},
            "size_m": [w_m, h_m],
            "pixel_quad": quad.tolist(),
        })

    out8 = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(out_p, out8, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return anns


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--place", required=True,
                    help='JSON: [{"cls":"cone","Y":0,"Z":8},...]')
    a = ap.parse_args()

    bgp = "/data/render_hybrid/bgs/" + a.bg
    bg = cv2.imread(bgp)
    if bg is None:
        raise SystemExit("底图读取失败: " + bgp)

    r = load_H(a.bg)
    if r is None:
        raise SystemExit("无人工标定: " + a.bg)
    H, cam_h = r

    placed = json.loads(a.place)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    anns = composite(bg, H, placed, a.out)

    print("底图: %s  %dx%d" % (a.bg, bg.shape[1], bg.shape[0]))
    print("放置: %d 个" % len(placed))
    for an in anns:
        b = an["bbox"]
        print("  %-10s bbox=(%.0f,%.0f,%.0f,%.0f)  Z=%.1fm Y=%+.2fm"
              % (an["class"], b[0], b[1], b[2], b[3],
                 an["world"]["Z_m"], an["world"]["Y_m"]))
    print("产出:", a.out)

    ad = os.path.join(os.path.dirname(a.out), "annotations")
    os.makedirs(ad, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.out))[0]
    with open(os.path.join(ad, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump({"image": os.path.basename(a.out),
                   "source_image": a.bg,
                   "source_path": bgp,
                   "width": bg.shape[1], "height": bg.shape[0],
                   "module": "OBJ",
                   "objects": anns,
                   "calibration": "manual_calib",
                   "training_eligible": False,
                   "training_blockers": ["no_instance_depth_pass",
                                         "no_original_labels",
                                         "occlusion_is_2d_only"]},
                  f, ensure_ascii=False, indent=1)
    print("标注:", os.path.join(ad, stem + ".json"))
