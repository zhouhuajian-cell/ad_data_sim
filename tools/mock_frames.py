# -*- coding: utf-8 -*-
# mock_frames.py — 合成假帧（不依赖 UE，用于把 post/ 与 dashboard 链路先跑通）
# 干什么：按真值 JSON 契约（附录 B）生成 raw 四通道 + meta + tasks.jsonl + done 标记，模拟引擎输出。
# 怎么跑：python tools/mock_frames.py --out _dryrun/data --frames 12 [--bit 8|16] [--width 320 --height 180]
# 成功判据：raw 下每帧 *_rgb/_depth/_sem.png 齐全，meta/<frame_id>.json 字段齐全且框在画面内；
#          同参数重跑逐字节一致（固定种子）。
# ⚠️ 这是给自检/界面调试用的假数据，图像内容与真实渲染无关；不要拿它训练，也不要当成阶段三验收。
import argparse
import json
import os
import random
import struct
import zlib

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config")
CLASS_STENCIL = {"road": 1, "vehicle": 2, "pedestrian": 3,
                 "traffic_sign": 4, "cone": 5, "vegetation": 6, "sky": 255}


def write_png16_rgb(path, arr):
    """自己写 16bit RGB PNG（滤波全 0）：不依赖 OpenCV，保证生成端可复现、可被 cv2/Pillow 复核。"""
    h, w, _ = arr.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)                                  # filter type 0（None）
        raw += arr[y].astype(">u2").tobytes()
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)   # 16bit, color type 2 = RGB
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def make_scene(w, h, rng, bit):
    """合成一张道路场景：上半天空、下半路面，叠几块类别色块。返回 (rgb 图, 深度, 语义, 目标框)。"""
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]

    # 线性 RGB（0~1）：天空偏蓝、路面偏灰，色块按类别上色
    img = np.zeros((h, w, 3), dtype=np.float32)
    sky = yy < 0.45
    img[..., 0] = np.where(sky, 0.35 + 0.25 * (1 - yy), 0.28 + 0.10 * yy)
    img[..., 1] = np.where(sky, 0.45 + 0.25 * (1 - yy), 0.28 + 0.10 * yy)
    img[..., 2] = np.where(sky, 0.75 + 0.20 * (1 - yy), 0.30 + 0.10 * yy)
    depth = np.full((h, w), 200.0, dtype=np.float32)          # 天空=远平面 200m
    sem = np.full((h, w), CLASS_STENCIL["sky"], dtype=np.uint8)
    road = (yy >= 0.45)[:, 0]                              # 只与行号有关，取一维行掩码
    depth = np.where(road[:, None], 6.0 + 60.0 * (yy - 0.45) / 0.55, depth)
    sem[road, :] = CLASS_STENCIL["road"]

    specs = [
        ("traffic_sign", (0.58, 0.20, 0.76, 0.52), (0.85, 0.15, 0.15), 22.0),
        ("vehicle", (0.10, 0.55, 0.40, 0.88), (0.20, 0.35, 0.85), 12.0),
        ("pedestrian", (0.80, 0.46, 0.90, 0.80), (0.95, 0.75, 0.10), 9.0),
        ("cone", (0.45, 0.78, 0.50, 0.91), (0.95, 0.40, 0.05), 7.0),
    ]
    objects = []
    for i, (cls, rel, color, dist) in enumerate(specs):
        dx, dy, dw, dh = [rng.uniform(-0.012, 0.012) for _ in range(4)]
        x1 = int(max(0.0, min(1.0, rel[0] + dx)) * w)
        y1 = int(max(0.0, min(1.0, rel[1] + dy)) * h)
        x2 = int(max(0.0, min(1.0, rel[2] + dw)) * w)
        y2 = int(max(0.0, min(1.0, rel[3] + dh)) * h)
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        img[y1:y2, x1:x2] = color
        depth[y1:y2, x1:x2] = dist
        sem[y1:y2, x1:x2] = CLASS_STENCIL[cls]
        objects.append({
            "class": cls,
            "instance_id": i + 1,
            "bbox_3d": [round(dist, 3), round((x1 + x2) / 2.0 / w * 6 - 3, 3),
                        round((1 - (y1 + y2) / 2.0 / h) * 3, 3),
                        round((x2 - x1) / w * 4, 3), round((y2 - y1) / h * 3, 3),
                        round((x2 - x1) / w * 4, 3), 0.0],
            "bbox_2d": [float(x1), float(y1), float(x2), float(y2)],
            "occlusion": 0.15 if cls == "pedestrian" else 0.0,
            "distance": round(dist, 3),
            "truncated": False,
        })
    img = np.clip(img, 0, 1)
    if bit == 16:
        rgb = (img * 65535.0 + 0.5).astype(np.uint16)     # 线性 16bit，不做色调映射
    else:
        rgb = (img * 255.0 + 0.5).astype(np.uint8)
    depth16 = np.clip(depth / 200.0 * 65535.0, 0, 65535).astype(np.uint16)
    return rgb, depth16, sem, objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "_dryrun", "data"))
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--bit", type=int, choices=[8, 16], default=8)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    if a.frames <= 0:
        raise SystemExit("--frames 必须为正")
    with open(os.path.join(CONFIG, "lights.json"), encoding="utf-8") as f:
        lights = json.load(f)
    with open(os.path.join(CONFIG, "cameras.json"), encoding="utf-8") as f:
        cameras = json.load(f)
    with open(os.path.join(CONFIG, "scenes.json"), encoding="utf-8") as f:
        scenes = json.load(f)
    raw_dir = os.path.join(a.out, "raw")
    meta_dir = os.path.join(a.out, "meta")
    queue_dir = os.path.join(a.out, "queue")
    done_dir = os.path.join(queue_dir, "done")
    for d in (raw_dir, meta_dir, done_dir):
        os.makedirs(d, exist_ok=True)

    tasks = []
    for i in range(a.frames):
        tid = "av_%06d" % i
        rng = random.Random((a.seed << 8) ^ i)
        rgb, depth16, sem, objects = make_scene(a.width, a.height, rng, a.bit)
        lighting = lights[i % len(lights)]
        camera = cameras[i % len(cameras)]
        speed = float(camera.get("speed_mps") or 0.0)
        stem = tid
        if a.bit == 16:
            write_png16_rgb(os.path.join(raw_dir, "%s_rgb.png" % stem), rgb)
        else:
            Image.fromarray(rgb).save(os.path.join(raw_dir, "%s_rgb.png" % stem))
        Image.fromarray(depth16).save(os.path.join(raw_dir, "%s_depth.png" % stem))
        Image.fromarray(sem).save(os.path.join(raw_dir, "%s_sem.png" % stem))
        meta = {
            "frame_id": stem,
            "timestamp": round(i * (1.0 / 30.0), 4),
            "width": a.width,
            "height": a.height,
            "seed": a.seed,
            "speed": speed,
            "source": "mock_frames",
            "camera": {
                "name": camera.get("name"),
                "location": camera.get("location"),
                "rotation": camera.get("rotation"),
                "focal_length": camera.get("focal_length"),
                "film_back_width": camera.get("film_back_width"),
                "film_back_height": camera.get("film_back_height"),
                "aperture": camera.get("aperture"),
                "shutter_deg": camera.get("shutter_deg"),
            },
            "lighting": lighting,
            "objects": objects,
            "image_files": {
                "rgb": "%s_rgb.png" % stem,
                "depth": "%s_depth.png" % stem,
                "semantic": "%s_sem.png" % stem,
                "bit_depth": a.bit,
            },
        }
        with open(os.path.join(meta_dir, "%s.json" % stem), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        with open(os.path.join(done_dir, "%s.done" % tid), "w") as f:
            f.write("ok")
        tasks.append({
            "id": tid, "frame_id": stem, "seed": a.seed,
            "map": scenes[i % len(scenes)].get("map"),
            "lighting": lighting,
            "targets": [{"class": o["class"], "actor_class": "/Game/Mock/BP_Mock.BP_Mock_C",
                         "location": o["bbox_3d"][:3], "rotation": [0, 0, 0]} for o in objects],
            "camera": camera,
        })
    with open(os.path.join(queue_dir, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for row in tasks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("mock: %d 帧（%dbit，%dx%d）→ %s" % (a.frames, a.bit, a.width, a.height, a.out))


if __name__ == "__main__":
    main()
