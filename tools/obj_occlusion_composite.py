"""真实遮挡合成：Mask2Former 真实车辆 mask + 接地估距 + Blender 前景遮挡。

流程：
  1. Mask2Former 分割真实车辆（Vehicle 类）
  2. 每辆真车用 bbox 底部中心 + 相机内参接地估距 -> 世界坐标
  3. Blender 渲染的虚拟车（带深度 EXR）
  4. 逐像素比较：真实车比虚拟车近 -> 抹掉虚拟车该像素
  5. 输出遮挡裁剪后的合成图 + 覆盖率统计
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")

VEHICLE_IDS = [22, 52, 54, 55, 56, 57, 59, 60]   # Mapillary Vehicle 类


def get_real_vehicle_mask(img_path, clip_dir, stem):
    """用 Mask2Former 拿真实车辆像素 mask。"""
    from gnd_mask_4a import MapillaryMarkingExtractor
    img = cv2.imread(img_path)
    ext = MapillaryMarkingExtractor()
    # 复用 extractor 的模型，但取 Vehicle 类
    seg = ext._run_segmentation(img) if hasattr(ext, "_run_segmentation") else None
    if seg is None:
        # 退化：直接调 predictor
        m = ext.get_marking_mask(img, prob_thresh=0.15)
        return np.zeros_like(m), []
    veh = np.isin(seg, VEHICLE_IDS).astype(np.uint8) * 255
    return veh, []


def vehicles_from_semantic(img, ext, min_area=800):
    """Mask2Former 语义 -> 车辆连通域列表。"""
    proc = ext._seg_semantic(img) if hasattr(ext, "_seg_semantic") else None
    if proc is None:
        return []
    n, lab, st, _ = cv2.connectedComponentsWithStats(proc, 8)
    out = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = st[i, cv2.CC_STAT_LEFT]
        y = st[i, cv2.CC_STAT_TOP]
        w = st[i, cv2.CC_STAT_WIDTH]
        h = st[i, cv2.CC_STAT_HEIGHT]
        out.append({"bbox": [x, y, w, h], "area": int(area),
                    "bottom_center": [x + w // 2, y + h]})
    return out


def ground_distance_from_bottom(px, py, cam_h, fy, cy):
    """接地估距：已知相机高度 h、像素纵坐标 y、主点 cy、焦距 fy
    平地假设下，地面点距离 Z = h * fy / (y - cy)
    """
    dy = py - cy
    if dy <= 1e-3:
        return None
    return cam_h * fy / dy


def occlude_virtual_by_real(fg_rgba, synth_depth, real_veh_mask,
                            real_veh_depths, tolerance=0.8):
    """真实车更近 -> 抹掉虚拟车。

    real_veh_depths: (H,W) 真实车每个像素的距离（用接地估距填充）
    """
    h, w = fg_rgba.shape[:2]
    alpha = fg_rgba[:, :, 3] > 0
    sd = np.asarray(synth_depth, dtype=np.float32)
    rd = np.asarray(real_veh_depths, dtype=np.float32)

    sd_ok = np.isfinite(sd) & (sd > 0.01) & (sd < 1e5)
    rd_ok = real_veh_mask & np.isfinite(rd) & (rd > 0.01) & (rd < 200)

    occluded = np.zeros((h, w), dtype=bool)
    both = alpha & sd_ok & rd_ok
    occluded[both] = rd[both] < (sd[both] - tolerance)

    out = fg_rgba.copy()
    out[occluded, 3] = 0
    return out, occluded


def main():
    fg_path = "/tmp/align_p2/fg_rgba.png"
    depth_npy = "/tmp/align_p2/depth.npy"
    bg_path = "/data/align_test/bg.jpg"
    out_dir = "/tmp/occlusion_out"
    os.makedirs(out_dir, exist_ok=True)

    fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
    bg = cv2.imread(bg_path)
    depth = np.load(depth_npy)
    meta = json.load(open("/tmp/align_p2/meta.json"))
    K = meta["K"]

    print("=== 1. Mask2Former 真实车辆 ===")
    from gnd_mask_4a import MapillaryMarkingExtractor
    ext = MapillaryMarkingExtractor()
    veh_mask = ext.get_vehicle_mask(bg, prob_thresh=0.5) > 127
    print("  车辆像素: %d (%.2f%%)" % (veh_mask.sum(),
                                      veh_mask.mean() * 100))

    # 每个连通域接地估距
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        veh_mask.astype(np.uint8), 8)
    real_depth_map = np.full(bg.shape[:2], np.nan, np.float32)
    veh_list = []
    cam_h = meta["camera"]["height_m"]
    # 用 pitch=2 度重建外参（与本帧地平线一致）
    from cv_to_blender_cam import (build_camera_from_ground_pose,
                                   estimate_instance_depth_from_ground)
    Knp = np.array([[K["fx"], 0, K["cx"]],
                    [0, K["fy"], K["cy"]],
                    [0, 0, 1]], dtype=np.float64)
    T_w2c = build_camera_from_ground_pose(cam_height_m=cam_h, pitch_deg=2.0)

    real_depth_map = np.full(bg.shape[:2], np.nan, np.float32)
    veh_list = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if area < 500:
            continue
        x = st[i, cv2.CC_STAT_LEFT]; y = st[i, cv2.CC_STAT_TOP]
        w = st[i, cv2.CC_STAT_WIDTH]; hh = st[i, cv2.CC_STAT_HEIGHT]
        inst = (lab == i)
        # 接地估距：只取 mask 最底部 8% 的行（脚点区域），
        # 顶部像素贴近地平线，距离会发散到无穷
        ys_i, xs_i = np.where(inst)
        y_max = int(ys_i.max())
        y_lo = int(y_max - 3)
        foot = np.zeros_like(inst)
        sel = (ys_i >= y_lo) & (ys_i <= y_max)
        foot[ys_i[sel], xs_i[sel]] = True
        dists, _ = estimate_instance_depth_from_ground(foot, Knp, T_w2c)
        if dists is None:
            continue
        vv = dists[np.isfinite(dists) & (dists > 0)]
        if vv.size == 0:
            continue
        d_foot = float(np.median(vv))
        if d_foot < 2 or d_foot > 150:
            continue
        # 整实例距离图仍用全 mask（供逐像素遮挡比较）
        dists_full, _ = estimate_instance_depth_from_ground(inst, Knp, T_w2c)
        d_p95 = float(np.percentile(
            dists_full[np.isfinite(dists_full)], 50))
        d_med = d_foot
        print("    [DBG] bbox=%s y_max=%d y_lo=%d foot_px=%d d_foot=%.2f d_fullP50=%.2f" % (
            [int(x),int(y),int(w),int(hh)], y_max, y_lo, int(foot.sum()), d_foot, d_p95))
        veh_list.append({"bbox": [int(x), int(y), int(w), int(hh)],
                         "y_bottom": int(y_max),
                         "dist_m": round(d_foot, 2),
                         "dist_median_m": round(d_med, 2),
                         "area": int(area)})
        # dists 是 mask 内扁平数组，需按像素坐标回填
        # 真车作为刚性前景：整个实例统一用底部接地估距值
        # （逐像素距离在倾斜视角下会因车身上部接近地平线而发散到 100m+）
        real_depth_map[inst] = d_foot

    veh_list.sort(key=lambda v: v["dist_m"])
    print("  有效真实车辆: %d" % len(veh_list))
    for v in veh_list[:6]:
        print("    bbox=%s  dist=%.1fm" % (v["bbox"], v["dist_m"]))

    print()
    print("=== 2. 遮挡裁剪（诊断）===")
    _va = fg[:, :, 3] > 0
    _sd = np.asarray(depth, np.float32)
    _sd_ok = np.isfinite(_sd) & (_sd > 0.01) & (_sd < 1e5)
    _rd_ok = veh_mask & np.isfinite(real_depth_map) & (real_depth_map > 0.01)
    print("  虚拟车像素: %d" % _va.sum())
    print("  虚拟车深度有效: %d" % (_va & _sd_ok).sum())
    print("  真实车深度有效: %d" % _rd_ok.sum())
    print("  两者同时有效: %d" % (_va & _sd_ok & _rd_ok).sum())
    _both = _va & _sd_ok & _rd_ok
    if _both.any():
        _rv = real_depth_map[_both]; _sv = _sd[_both]
        print("    真车距离 %.2f~%.2f  中位%.2f" % (
            np.nanmin(_rv), np.nanmax(_rv), np.nanmedian(_rv)))
        print("    虚拟车深度 %.2f~%.2f 中位%.2f" % (
            _sv.min(), _sv.max(), np.median(_sv)))
        _closer = _rv < (_sv - 0.8)
        print("    真车更近的像素: %d (%.1f%%)" % (_closer.sum(),
              _closer.mean()*100))
    print()
    if not veh_list:
        print("  无真实车辆，跳过遮挡")
        comp = bg.copy()
        out_occl = fg
        occ_mask = np.zeros(bg.shape[:2], np.uint8)
    else:
        out_occl, occ = occlude_virtual_by_real(
            fg, depth, veh_mask, real_depth_map, tolerance=0.8)
        occ_mask = (occ.astype(np.uint8)) * 255
        vis_before = (fg[:, :, 3] > 0).sum()
        vis_after = (out_occl[:, :, 3] > 0).sum()
        print("  虚拟车像素: %d -> %d  (被遮挡 %.1f%%)" % (
            vis_before, vis_after,
            (vis_before - vis_after) / max(vis_before, 1) * 100))

        # 简单可靠的 alpha 合成（阴影已烘焙在前景 RGB 里）
        a = out_occl[:, :, 3:4].astype(np.float32) / 255.0
        fg_rgb = out_occl[:, :, :3].astype(np.float32)
        comp = (fg_rgb * a + bg.astype(np.float32) * (1 - a))
        comp = np.clip(comp, 0, 255).astype(np.uint8)

    # 合成
    cv2.imwrite(os.path.join(out_dir, "composite_occluded.jpg"), comp,
                [cv2.IMWRITE_JPEG_QUALITY, 94])
    # 遮挡前（未裁剪）对比
    a0 = fg[:, :, 3:4].astype(np.float32) / 255.0
    comp0 = np.clip(fg[:, :, :3].astype(np.float32) * a0
                    + bg.astype(np.float32) * (1 - a0), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "composite_before.jpg"), comp0,
                [cv2.IMWRITE_JPEG_QUALITY, 94])
    cv2.imwrite(os.path.join(out_dir, "occluded_mask.png"), occ_mask)

    # 可视化：真实车辆 mask 叠加
    vis = comp.copy()
    overlay = vis.copy()
    overlay[veh_mask] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
    cv2.imwrite(os.path.join(out_dir, "real_vehicle_mask.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    with open(os.path.join(out_dir, "occlusion_meta.json"), "w",
              encoding="utf-8") as _f:
        json.dump({"real_vehicles": veh_list,
                   "occluded_px": int((occ_mask > 0).sum()) if veh_list else 0},
                  _f, ensure_ascii=False, indent=2)
    print()
    print("输出:", out_dir)


if __name__ == "__main__":
    main()
