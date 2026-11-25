#!/usr/bin/env python3
"""验证真实 OBJ 标注里 objWx/Wy/Wz 的语义：
用 XML 里的内参 + 3D 框，投影到图像，看是否贴合真实车辆。

用法:
    python verify_obj_boxes.py <xml路径> [输出图]
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def rodrigues(rv):
    """罗德里格斯旋转向量 -> 旋转矩阵。"""
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def box_corners(x, y, z, L, W, H, yaw):
    """按 (x,y,z)=中心 生成 8 个角点。"""
    half = np.array([[L / 2, W / 2, H / 2]]) * np.array(
        [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
         [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    pts = (R @ half.T).T
    pts += np.array([x, y, z])
    return pts


EDGES = [(0, 1), (0, 2), (1, 3), (2, 3),
         (4, 5), (4, 6), (5, 7), (6, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def main():
    xml_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/verify_boxes.jpg"

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ---------- 内参 ----------
    cam = root.find(".//cam_st_x2f_front")
    if cam is None:
        print("no cam_st_x2f_front")
        return
    iv = [float(cam.find("intrinsic_matrix%d" % i).text) for i in range(9)]
    K = np.array(iv, dtype=np.float64).reshape(3, 3)
    rv = [float(cam.find("rotation_vector%d" % i).text) for i in range(3)]
    tv = [float(cam.find("translation_vector%d" % i).text) for i in range(3)]
    R = rodrigues(rv)
    T = np.array(tv, dtype=np.float64).reshape(3)

    print("K:\n", np.round(K, 2))
    print("R:\n", np.round(R, 4))
    print("T:", np.round(T, 3))

    # ---------- 图像 ----------
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    base = os.path.dirname(os.path.dirname(xml_path))   # 上一层 clip
    cands = [
        os.path.join(base, "cam_st_x2f_front", stem + ".jpg"),
        os.path.join(os.path.dirname(xml_path), stem + ".jpg"),
        os.path.splitext(xml_path)[0] + ".jpg",
    ]
    img_path = None
    for c in cands:
        if os.path.isfile(c):
            img_path = c
            break
    img = cv2.imread(img_path) if img_path else None
    if img is None:
        print("no image")
        return
    h, w = img.shape[:2]
    print("image: %s  %dx%d" % (os.path.basename(img_path), w, h))

    # ---------- 标注 ----------
    vis = img.copy()
    n_drawn = 0
    for grp in root.findall(".//LabelInfo/*"):
        gname = grp.tag
        if gname in ("Vehicle", "Pedestrian", "Rider", "Three", "Cone"):
            for obj in grp.findall("Object"):
                def gv(tag, dv=0.0):
                    e = obj.find(tag)
                    return float(e.text) if e is not None and e.text else dv
                x = gv("objWx"); y = gv("objWy"); z = gv("objWz")
                L = gv("objLength"); Wd = gv("objWidth"); H = gv("objHeight")
                yaw = gv("objYaw")
                corners = box_corners(x, y, z, L, Wd, H, yaw)

                # 世界(自车) -> 相机
                # 尝试 3 种解释：
                #  A) objW* 是自车坐标，需经 R,T 变换
                #  B) objW* 已是相机坐标
                #  C) objWz 是离地高度（中心），世界 z 需减相机高度
                for mode in ("A", "B"):
                    if mode == "A":
                        pc = (R @ corners.T).T + T
                    else:
                        pc = corners
                    if (pc[:, 2] <= 0.5).any():
                        continue
                    uv = (K @ pc.T).T
                    uv = uv[:, :2] / uv[:, 2:3]
                    ok = ((uv[:, 0] > -50) & (uv[:, 0] < w + 50) &
                          (uv[:, 1] > -50) & (uv[:, 1] < h + 50))
                    if ok.sum() < 4:
                        continue
                    color = (0, 255, 0) if mode == "A" else (0, 128, 255)
                    for i, j in EDGES:
                        cv2.line(vis,
                                 (int(uv[i][0]), int(uv[i][1])),
                                 (int(uv[j][0]), int(uv[j][1])),
                                 color, 2)
                    for p in uv:
                        cv2.circle(vis, (int(p[0]), int(p[1])), 3,
                                   (0, 0, 255), -1)
                    lbl = "%s %s %s" % (obj.get("kind", gname)[:6], mode,
                                        "z=%.1f" % z)
                    cv2.putText(vis, lbl, (int(uv[0][0]), int(uv[0][1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    n_drawn += 1
                    print("  %s %s mode=%s  x=%.1f y=%.1f z=%.1f "
                          "L=%.1f W=%.1f H=%.1f  uv0=(%.0f,%.0f)" % (
                              gname, obj.get("kind", ""), mode,
                              x, y, z, L, Wd, H, uv[0][0], uv[0][1]))
                    break

    print("\n画出 %d 个框" % n_drawn)
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("输出:", out_path)


if __name__ == "__main__":
    main()
