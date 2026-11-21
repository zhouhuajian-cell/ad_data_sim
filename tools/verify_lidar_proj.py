#!/usr/bin/env python3
"""点云 + 相机投影验证：确认点云能投到图像上对准物体。"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from label_gate import load_frame_labels
from real_calib import load_cam_real


def load_lidar(path):
    """读点云：每点 16 字节 (i, x, y, z) float32。"""
    raw = np.fromfile(path, dtype=np.float32)
    n = len(raw) // 4
    return raw[:n * 4].reshape(-1, 4)


def main():
    clip = ("/data/test_data/real_clips/"
            "20261120_HS5001_changchun_1v_extremely_closetruck/"
            "133913-1788137262.664")
    lidar_d = os.path.join(clip, "lidar_hs_pandar128_topmiddle")
    bin_f = sorted(glob.glob(os.path.join(lidar_d, "*.bin")))[0]
    stem = os.path.basename(bin_f).replace(".bin", "")

    pts = load_lidar(bin_f)
    print("点云: %d 点" % len(pts))
    print("  x: [%.2f, %.2f]" % (pts[:, 1].min(), pts[:, 1].max()))
    print("  y: [%.2f, %.2f]" % (pts[:, 2].min(), pts[:, 2].max()))
    print("  z: [%.2f, %.2f]" % (pts[:, 3].min(), pts[:, 3].max()))

    # 相机投影
    cam = load_cam_real(os.path.basename(os.path.dirname(clip)),
                        os.path.basename(clip))
    print("\n相机 fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
          % (cam["fx"], cam["fy"], cam["cx"], cam["cy"]))

    img_p = os.path.join(clip, "cam_st_x2f_front", stem + ".jpg")
    img = cv2.imread(img_p)
    if img is None:
        print("读图失败:", img_p)
        return
    print("图像:", img.shape)

    # 用 real_calib 投影
    from real_calib import body_to_pixel_real

    # 点云 xyz 可能是 lidar 系，先试直投
    xyz = pts[:, 1:4].astype(np.float64)
    # lidar → body：假设 lidar 在车顶，需要外参
    # 先看标定文件有没有 lidar_to_camera
    print("\n=== 尝试直接投影（假设同系）===")
    uv, zc, ok = body_to_pixel_real(cam, xyz[:50000])
    valid = ok & np.isfinite(uv).all(axis=1)
    u = uv[valid, 0]
    v = uv[valid, 1]
    print("可投影点: %d / %d" % (valid.sum(), len(xyz)))
    if valid.sum() > 100:
        in_img = (u > 0) & (u < img.shape[1]) & (v > 0) & (v < img.shape[0])
        print("落在图像内: %d" % in_img.sum())
        print("  u: [%.0f, %.0f]" % (u[in_img].min() if in_img.any() else 0,
                                     u[in_img].max() if in_img.any() else 0))
        print("  v: [%.0f, %.0f]" % (v[in_img].min() if in_img.any() else 0,
                                     v[in_img].max() if in_img.any() else 0))

        # 可视化
        vis = img.copy()
        for uu, vv in zip(u[in_img][:20000], v[in_img][:20000]):
            cv2.circle(vis, (int(uu), int(vv)), 2, (0, 255, 0), -1)
        out = "/tmp/lidar_proj.jpg"
        cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("\n可视化:", out)

    # 对照标注中的卡车
    lab = load_frame_labels(clip, xml_path=os.path.join(
        clip, "label_lidar_hs_pandar128_topmiddle", stem + ".xml"))
    if lab:
        print("\n标注物体（前 3）:")
        cnt = 0
        for o in (lab.get("objects") or []):
            if o.get("delete") or not o.get("in_front"):
                continue
            print("  %-8s Wx=%.1f Wy=%.1f L=%.1f W=%.1f H=%.1f bbox=%s"
                  % (o.get("kind"), o.get("Wx", 0), o.get("Wy", 0),
                     o.get("L", 0), o.get("W", 0), o.get("H", 0),
                     [round(x) for x in (o.get("bbox2d") or [])]))
            cnt += 1
            if cnt >= 3:
                break


if __name__ == "__main__":
    main()
