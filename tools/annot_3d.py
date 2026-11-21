#!/usr/bin/env python3
"""仿真输出：带完整 3D 标注（真实物体真值 + 合成物体生成）。

输出格式（BEV 训练用）：
{
  "frame": "1788137262.664",
  "timestamp": 1788137262.664,
  "image": "frames/xxx_sim.jpg",
  "source_image": "frames/xxx_orig.jpg",
  "width": 1920, "height": 1080,
  "camera": {"name": "cam_st_x2f_front", "fx": ..., "fy": ..., "cx": ..., "cy": ...},
  "objects": [
    {
      "id": 1,
      "source": "real",                    // real | synthetic
      "class": "truck",
      "bbox2d": [821, 277, 1137, 672],     // 像素
      "box3d": {                           // 车体系，米
        "x": 20.6, "y": 0.0, "z": 1.15,    // 中心
        "L": 9.4, "W": 2.4, "H": 2.3,      // 尺寸
        "yaw": 3.14,                       // 朝向（弧度）
        "roll": 0.0, "pitch": 0.0
      },
      "depth_m": 20.6,                     // 相机系纵深
      "visible": true,
      "occlusion": null                    // 有深度时方能算
    }
  ],
  "meta": {
    "detector": "lane_obj_v1",
    "removed": [...],                      // 被抹除的物体
    "placed": [...],                       // 新放置的物体
    "module": "obj",
    "training_eligible": false,
    "training_blockers": [...]
  }
}
"""
import json
import os


def real_object_3d(o, idx):
    """done_data XML 物体 → 标准 3D 标注。"""
    return {
        "id": idx,
        "source": "real",
        "class": o.get("kind") or "unknown",
        "bbox2d": [round(float(v), 1) for v in (o.get("bbox2d") or [])],
        "box3d": {
            "x": round(float(o.get("Wx") or 0), 3),
            "y": round(float(o.get("Wy") or 0), 3),
            "z": round(float(o.get("Wz") or 0), 3),
            "L": round(float(o.get("L") or 0), 3),
            "W": round(float(o.get("W") or 0), 3),
            "H": round(float(o.get("H") or 0), 3),
            "yaw": round(float(o.get("yaw") or 0), 5),
            "roll": round(float(o.get("roll") or 0), 5),
            "pitch": round(float(o.get("pitch") or 0), 5),
        },
        "depth_m": round(float(o.get("depth_m") or 0), 2)
        if o.get("depth_m") else None,
        "visible": True,
        "occlusion": None,
        "tracking_id": o.get("tracking_id"),
        "uuid": o.get("uuid"),
    }


def synth_object_3d(cls, x, y, w_m, h_m, idx, sprite_meta=None):
    """合成物体 → 标准 3D 标注。

    假设：物体站在地面，z = h/2，朝向 0（正对车头）
    """
    return {
        "id": idx,
        "source": "synthetic",
        "class": cls,
        "bbox2d": None,           # 渲染后填
        "box3d": {
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "z": round(float(h_m) / 2.0, 3),
            "L": round(float(w_m), 3),
            "W": round(float(w_m), 3),
            "H": round(float(h_m), 3),
            "yaw": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
        },
        "depth_m": round(float(x), 2),
        "visible": True,
        "occlusion": None,
        "sprite": (sprite_meta or {}).get("source", "unknown"),
    }


def build_frame_annotation(stem, batch, clip, cam, img_w, img_h,
                           real_objects, synth_objects, removed,
                           module="obj", extra=None):
    """构造完整帧标注。"""
    objects = []
    idx = 1

    # 真实物体（读 XML 真值）
    for o in real_objects:
        if o.get("delete") or not o.get("in_front"):
            continue
        if not o.get("bbox2d"):
            continue
        objects.append(real_object_3d(o, idx))
        idx += 1

    # 合成物体
    synth_ids = []
    for s in synth_objects:
        rec = synth_object_3d(
            s["cls"], s["x"], s["y"],
            s.get("w_m", 1.0), s.get("h_m", 1.5),
            idx, s.get("meta"))
        if s.get("bbox2d"):
            rec["bbox2d"] = [round(float(v), 1) for v in s["bbox2d"]]
        objects.append(rec)
        synth_ids.append(idx)
        idx += 1

    ann = {
        "frame": stem,
        "timestamp": _to_ts(stem),
        "batch": batch,
        "clip": clip,
        "image": "frames/%s_sim.jpg" % stem,
        "source_image": "frames/%s_orig.jpg" % stem,
        "width": img_w,
        "height": img_h,
        "module": module,
        "camera": {
            "name": cam.get("camera_name", "cam_st_x2f_front"),
            "fx": round(float(cam["fx"]), 3),
            "fy": round(float(cam["fy"]), 3),
            "cx": round(float(cam["cx"]), 3),
            "cy": round(float(cam["cy"]), 3),
            "R": cam.get("R"),
            "t": cam.get("t"),
        },
        "objects": objects,
        "meta": {
            "n_real": len([o for o in objects if o["source"] == "real"]),
            "n_synth": len(synth_ids),
            "removed": removed,
            "placed": synth_ids,
            "training_eligible": False,
            "training_blockers": [
                "appearance_unverified",
                "occlusion_unverified",
                "single_camera",
            ],
        },
    }
    if extra:
        ann["meta"].update(extra)
    return ann


def _to_ts(stem):
    try:
        return float(stem.split(".")[0])
    except Exception:
        return None


def write_frame(out_dir, ann):
    """写单帧标注。"""
    ad = os.path.join(out_dir, "annotations")
    os.makedirs(ad, exist_ok=True)
    p = os.path.join(ad, ann["frame"] + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=1)
    return p


def write_clip(out_dir, frames, batch, clip, module, cfg):
    """写整个 clip 的标注（训练用）。"""
    manifest = {
        "version": "bev_sim_v1",
        "batch": batch,
        "clip": clip,
        "module": module,
        "config": cfg,
        "n_frames": len(frames),
        "frames": [f["frame"] for f in frames],
        "classes": sorted(set(
            o["class"] for f in frames for o in f["objects"])),
        "coordinate_system": {
            "frame": "body",
            "x": "forward (m)",
            "y": "left (m)",
            "z": "up (m)",
            "yaw": "rad, 0 = facing +x",
            "box3d": "center + LWH + yaw",
        },
        "camera": frames[0]["camera"] if frames else None,
        "training_eligible": False,
        "training_blockers": [
            "appearance_unverified",
            "occlusion_unverified",
            "single_camera",
        ],
    }
    p = os.path.join(out_dir, "clip_manifest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return p




# CORNERS-3D: 3D 框 8 角点 → 像素
def box_corners_pixel(box3d, cam):
    """3D 框 8 角点投影到像素。

    box3d: {x,y,z,L,W,H,yaw}
    返回 8 个 (u,v)，顺序：
      0-3: 底面 (近左, 近右, 远右, 远左)
      4-7: 顶面 (近左, 近右, 远右, 远左)
    """
    import math
    import numpy as np

    x, y, z = float(box3d.get("x") or 0), float(box3d.get("y") or 0), \
        float(box3d.get("z") or 0)
    L = float(box3d.get("L") or 1.0)
    W = float(box3d.get("W") or 1.0)
    H = float(box3d.get("H") or 1.0)
    yaw = float(box3d.get("yaw") or 0.0)

    # 框体 8 角（车体坐标系局部）
    hl, hw, hh = L / 2.0, W / 2.0, H / 2.0
    local = np.array([
        [-hl, -hw, -hh], [hl, -hw, -hh], [hl, hw, -hh], [-hl, hw, -hh],
        [-hl, -hw, hh], [hl, -hw, hh], [hl, hw, hh], [-hl, hw, hh],
    ], float)

    c, s_ = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]], float)
    pts = (Rz @ local.T).T + np.array([x, y, z], float)

    from real_calib import body_to_pixel_real
    uv, zc, ok = body_to_pixel_real(cam, pts)
    # CORNER-CLAMP: 过滤掉画面外的/异常的角点
    W = float(cam.get("img_w") or 1920)
    H = float(cam.get("img_h") or 1080)
    LIM = 3.0   # 允许越界 3 倍画面
    out = []
    for i in range(8):
        okk = False
        if ok[i] and np.isfinite(uv[i]).all():
            u, v = float(uv[i][0]), float(uv[i][1])
            if (-W * LIM) <= u <= (W * (1 + LIM)) and \
               (-H * LIM) <= v <= (H * (1 + LIM)):
                out.append([u, v])
                okk = True
        if not okk:
            out.append(None)
    return out


if __name__ == "__main__":
    print("仿真标注格式说明")
    print("=" * 60)
    print(__doc__)
