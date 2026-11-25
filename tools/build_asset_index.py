import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np


def _build_shape_templates(size=64):
    """生成标准圆形、正三角形、倒三角形、矩形二值模板"""
    tpls = {}

    # 1. 圆形
    circle = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(circle, (size // 2, size // 2), size // 2 - 1, 1, -1)
    tpls["circle"] = circle

    # 2. 正三角形
    tri_up = np.zeros((size, size), dtype=np.uint8)
    pts_up = np.array([[size // 2, 1], [1, size - 2], [size - 2, size - 2]],
                      dtype=np.int32)
    cv2.fillPoly(tri_up, [pts_up], 1)
    tpls["triangle_up"] = tri_up

    # 3. 倒三角形（让行牌等）
    tri_down = np.zeros((size, size), dtype=np.uint8)
    pts_down = np.array([[1, 1], [size - 2, 1], [size // 2, size - 2]],
                        dtype=np.int32)
    cv2.fillPoly(tri_down, [pts_down], 1)
    tpls["triangle_down"] = tri_down

    # 4. 矩形
    rect = np.ones((size, size), dtype=np.uint8)
    tpls["rect"] = rect

    return tpls


TEMPLATES = _build_shape_templates(64)


def _classify_shape(alpha_mask):
    """基于前景 Alpha 轮廓与标准模板的 IoU 判定形状"""
    ys, xs = np.where(alpha_mask > 128)
    if len(xs) < 25:
        return "other"

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    w, h = x2 - x1, y2 - y1
    aspect = w / (h + 1e-5)

    # 明显长条形直接判矩形
    if aspect > 1.45 or aspect < 0.68:
        return "rect"

    crop = (alpha_mask[y1:y2, x1:x2] > 128).astype(np.uint8)
    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_NEAREST)

    best_shape, best_iou = "rect", -1.0
    for name, tpl in TEMPLATES.items():
        inter = np.logical_and(resized, tpl).sum()
        union = np.logical_or(resized, tpl).sum() + 1e-5
        iou = inter / union
        if iou > best_iou:
            best_iou = iou
            best_shape = name

    if best_shape.startswith("triangle"):
        return "triangle"
    return best_shape


def _classify_color(bgr, alpha_mask):
    """统计前景区域的 HSV 颜色占比，判定主色语义"""
    fg = alpha_mask > 128
    total = int(fg.sum())
    if total < 25:
        return "other"

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0][fg], hsv[:, :, 1][fg], hsv[:, :, 2][fg]

    blue_ratio = np.mean((H >= 95) & (H <= 130) & (S >= 60) & (V >= 40))
    red_ratio = np.mean(((H <= 12) | (H >= 168)) & (S >= 60) & (V >= 40))
    yellow_ratio = np.mean((H >= 15) & (H <= 38) & (S >= 60) & (V >= 60))

    # 蓝底指示牌
    if blue_ratio >= 0.25 and blue_ratio >= red_ratio and \
            blue_ratio >= yellow_ratio:
        return "blue"
    # 黄底警告牌
    if yellow_ratio >= 0.25 and yellow_ratio > blue_ratio:
        return "yellow"
    # 红圈白底 / 红底禁令牌（红圈通常占面积 12%~40%）
    if red_ratio >= 0.12:
        return "red_white"
    return "other"


def process_single_asset(args):
    file_path, root_dir = args
    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)

    rel_path = str(Path(file_path).relative_to(root_dir)).replace("\\", "/")
    # 默认取第一级子目录名作为国家/分组代码（如 VN/xxx.png -> VN）
    parts = rel_path.split("/")
    country = parts[0] if len(parts) > 1 else "DEFAULT"

    return {
        "path": rel_path,
        "country": country,
        "shape": _classify_shape(alpha),
        "color_type": _classify_color(bgr, alpha),
    }


def main():
    parser = argparse.ArgumentParser(description="批量构建 TSR 素材形状+颜色索引")
    parser.add_argument("--asset-dir", required=True, help="素材库根目录")
    parser.add_argument("--output", default="asset_index.json",
                        help="输出索引 JSON 路径")
    parser.add_argument("--workers", type=int,
                        default=max(1, os.cpu_count() - 2), help="并发进程数")
    args = parser.parse_args()

    root = Path(args.asset_dir)
    exts = {".png", ".webp", ".jpg", ".jpeg"}
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    print("扫描到 %d 张素材，启动 %d 进程建库..." % (len(files), args.workers))

    records = []
    stats = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_single_asset, (f, root)) for f in files]
        for idx, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res:
                records.append(res)
                key = "%s_%s" % (res["shape"], res["color_type"])
                stats[key] = stats.get(key, 0) + 1
            if idx % 5000 == 0 or idx == len(files):
                print("进度: %d/%d" % (idx, len(files)))

    out_data = {"total": len(records), "stats": stats, "assets": records}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print("索引构建完成 -> %s" % args.output)
    print("分布统计:", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
