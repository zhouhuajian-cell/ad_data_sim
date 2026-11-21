"""GND 2.5D 标线投影：把米制标线图案投影到真实底图。

不依赖 UE。平坦路面用单应变换即可。
输入：底图 + 相机参数/单应 + 标线世界坐标
输出：合成图 + 标线标注 + 覆盖率掩码

用法：
  python gnd_project.py --bg cand_04.jpg \
      --lines "lane_solid_white,-1.75,0,30,lane_dashed_white,1.75,0,30" \
      --out /tmp/gnd_out
"""
import argparse
import json
import os

import cv2
import numpy as np

PATTERN_DIR = "/data/render_hybrid/gnd_patterns"
CALIB_DIR = "/data/render_hybrid/bgs"
CAM_EST = "/data/render_hybrid/bgs/_camera_est.json"

# 标线类型 → (图案名, 物理宽度 m)
LINE_SPECS = {
    "lane_solid_white":   ("lane_solid_white_worn", 0.15),
    "lane_dashed_white":  ("lane_dashed_white_worn", 0.15),
    "lane_solid_yellow":  ("lane_solid_yellow_worn", 0.15),
    "lane_double_yellow": ("lane_double_yellow_worn", 0.15),
    "stop_line":          ("stop_line_worn", 0.40),
    "crosswalk_zebra":    ("crosswalk_zebra_worn", 0.45),
    "road_edge_white":    ("road_edge_white_worn", 0.20),
}


def load_camera(bg_name):
    """优先读人工标定的单应，其次读自动估计的相机参数。"""
    stem = os.path.splitext(bg_name)[0]
    calib_p = os.path.join(CALIB_DIR, "_calib_" + stem + ".json")
    if os.path.isfile(calib_p):
        with open(calib_p, encoding="utf-8") as f:
            d = json.load(f)
        return {"mode": "homography", "H": np.array(d["H"], float).reshape(3, 3),
                "source": "manual_calib", "lane_w": d.get("lane_width_m", 3.5)}

    if os.path.isfile(CAM_EST):
        with open(CAM_EST, encoding="utf-8") as f:
            est = json.load(f)
        e = est.get(bg_name) or {}
        if e.get("ok"):
            return {"mode": "camera", "fx": e["fx"], "fy": e["fy"],
                    "cx": e.get("cx"), "cy": e.get("cy"),
                    "vp": e.get("vanish_point"),
                    "source": "auto_estimate", "lane_w": 3.5}
    return None


def world_to_homography(mode):
    """世界地面 (X 前, Y 左, 单位米) → 像素单应的构造。

    相机模式：用针孔 + 地面平面
      u = fx * Y/(-X) ... 需要相机外参。这里用简化模型：
      相机在原点，朝 -X 看（车头方向），高 cam_h
    """
    return None


def build_ground_h(cam, img_w, img_h):
    """构造世界地面 → 像素的单应。

    做法：在相机坐标系里取 4 个地面点，投影到像素，求单应。
    简化假设：相机光轴水平朝前（-X），车辆在地面 z=0，相机高 h。
    """
    if cam["mode"] == "homography":
        return cam["H"]

    fx, fy = cam["fx"], cam["fy"]
    cx = cam.get("cx") or img_w * 0.5
    cy = cam.get("cy") or img_h * 0.55
    cam_h = 1.35

    # 世界点 (X 前, Y 左) 在 z=0 平面
    # 相机坐标：x_cam = Y, y_cam = -cam_h, z_cam = X
    def proj(X, Y):
        zc = X
        if zc <= 0.1:
            return None
        u = cx + fx * Y / zc
        v = cy + fy * cam_h / zc
        return (u, v)

    # 取 4 个世界点（近 6m，远 30m）
    pts_w = [(6.0, -3.5), (6.0, 3.5), (30.0, 3.5), (30.0, -3.5)]
    pts_px = []
    for X, Y in pts_w:
        p = proj(X, Y)
        if p is None:
            return None
        pts_px.append(p)

    src = np.array(pts_w, np.float32)
    dst = np.array(pts_px, np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def render_lines(bg, H, lines, out_dir, H_ORDER="xy"):
    """把标线投影到 bg 上。"""
    h, w = bg.shape[:2]
    canvas = bg.astype(np.float32)
    overlays = []          # (mask, line_id, class)

    for idx, spec in enumerate(lines):
        cls = spec["class"]
        y0 = spec["y"]              # 横向偏移（米，正=左）
        x0 = spec.get("x", 0.0)     # 起点（米）
        length = spec.get("length", 30.0)
        width = spec.get("width")

        name, def_w = LINE_SPECS.get(cls, (None, 0.15))
        pw = width or def_w

        pat_p = os.path.join(PATTERN_DIR, (name or cls) + ".png")
        if not os.path.isfile(pat_p):
            print("  跳过（无图案）: %s" % cls)
            continue

        tex = cv2.imread(pat_p, cv2.IMREAD_UNCHANGED)
        if tex is None or tex.ndim != 3 or tex.shape[2] != 4:
            print("  跳过（图案无效）: %s" % cls)
            continue

        # 世界四边形：沿纵向从 x0 到 x0+length，横向从 y0-pw/2 到 y0+pw/2
        # 注意：不同来源的 H 输入顺序不同
        #   manual_calib → (Y_left, Z_fwd)
        #   auto_estimate → (X_fwd, Y_left)
        if H_ORDER == "yz":
            quad = np.array([
                [y0 - pw / 2, x0],
                [y0 - pw / 2, x0 + length],
                [y0 + pw / 2, x0 + length],
                [y0 + pw / 2, x0],
            ], np.float32)
        else:
            quad = np.array([
                [x0,              y0 - pw / 2],
                [x0 + length,     y0 - pw / 2],
                [x0 + length,     y0 + pw / 2],
                [x0,              y0 + pw / 2],
            ], np.float32)

        # 世界 → 像素
        pts = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

        # 纹理角点（TL, TR, BR, BL）
        th, tw = tex.shape[:2]
        src4 = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)

        # 检查投影是否合法
        if not np.all(np.isfinite(pts)):
            print("  跳过（投影非法）: %s" % cls)
            continue
        if cv2.contourArea(pts.astype(np.float32)) < 20:
            print("  跳过（投影太小）: %s" % cls)
            continue

        # warpPerspective 无 INV 标志时：M 是「源 → 目标」
        # 实测：M = src4 → pts 且不用 WARP_INVERSE_MAP 才正确
        Pm = cv2.getPerspectiveTransform(src4, pts.astype(np.float32))

        # 预乘 alpha（避免黑边）
        alpha = tex[..., 3].astype(np.float32) / 255.0
        rgb = tex[..., :3].astype(np.float32) / 255.0
        prem = rgb * alpha[..., None]

        # 透视变换（M 是 src→dst，不加 INV 标志）
        a_w = cv2.warpPerspective(
            alpha, Pm, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        p_w = cv2.warpPerspective(
            prem, Pm, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        a_w = np.clip(a_w, 0, 1)
        straight = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
        straight[a_w <= 1e-4] = 0

        # 合成（sRGB 直接混合，简化处理）
        fg = (straight * 255.0)
        canvas = canvas * (1 - a_w[..., None]) + fg * a_w[..., None]

        overlays.append({
            "line_id": idx,
            "class": cls,
            "world_quad": quad.tolist(),
            "pixel_quad": pts.tolist(),
            "width_m": pw,
            "alpha_mean": float(a_w.mean()),
        })

    out = np.clip(canvas, 0, 255).astype(np.uint8)
    return out, overlays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", required=True)
    ap.add_argument("--lines", default="",
                    help="class,y,x,length[;class,y,x,length...]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bg_p = a.bg if os.path.isabs(a.bg) else os.path.join(CALIB_DIR, a.bg)
    bg = cv2.imread(bg_p)
    if bg is None:
        raise SystemExit("背景读取失败: " + bg_p)
    h, w = bg.shape[:2]
    print("底图: %s (%dx%d)" % (os.path.basename(bg_p), w, h))

    bg_name = os.path.basename(bg_p)
    cam = load_camera(bg_name)
    if not cam:
        raise SystemExit("无相机参数，请先在平台里做「地面标定」")
    print("相机来源: %s" % cam["source"])

    H = build_ground_h(cam, w, h)
    if H is None:
        raise SystemExit("单应构造失败")

    # 解析标线
    lines = []
    if a.lines:
        for part in a.lines.split(";"):
            f = part.split(",")
            if len(f) < 4:
                continue
            lines.append({
                "class": f[0],
                "y": float(f[1]),
                "x": float(f[2]),
                "length": float(f[3]),
            })
    else:
        # 默认：三车道标线
        # 近 4m → 远 20m：这个区间投影后有足够像素宽度
        lines = [
            {"class": "road_edge_white",   "y": -5.25, "x": 4, "length": 16},
            {"class": "lane_dashed_white", "y": -1.75, "x": 4, "length": 16},
            {"class": "lane_dashed_white", "y": 1.75,  "x": 4, "length": 16},
            {"class": "road_edge_white",   "y": 5.25,  "x": 4, "length": 16},
        ]

    print("标线: %d 条" % len(lines))
    H_ORDER = "yz" if cam["mode"] == "homography" else "xy"
    out_im, overlays = render_lines(bg, H, lines, a.out, H_ORDER)

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(bg_name)[0]
    out_p = os.path.join(a.out, stem + "_gnd_composed.jpg")
    cv2.imwrite(out_p, out_im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("产出: %s" % out_p)

    # 标注
    ann = {
        "image": os.path.basename(out_p),
        "source_image": bg_name,
        "source_path": bg_p,
        "width": w, "height": h,
        "module": "GND",
        "camera": cam["source"],
        "lines": overlays,
        "quality": {
            "policy": "gnd_projection_v1",
            "passed": False,
            "status": "unchecked",
            "semantic_status": "not_checked",
            "note": "投影几何已产出；像素融合与语义正确性未验证",
        },
        "training_eligible": False,
        "training_blockers": [
            "no_projection_verification",
            "no_topology_linkage",
            "no_bev_raster",
            "no_multi_camera_temporal",
        ],
    }
    ann_p = os.path.join(a.out, "annotations")
    os.makedirs(ann_p, exist_ok=True)
    with open(os.path.join(ann_p, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=1)
    print("标注: %s" % os.path.join(ann_p, stem + ".json"))

    for o in overlays:
        print("  [%d] %-22s w=%.2fm alpha=%.3f" %
              (o["line_id"], o["class"], o["width_m"], o["alpha_mean"]))


if __name__ == "__main__":
    main()
