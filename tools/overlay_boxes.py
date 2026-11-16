# -*- coding: utf-8 -*-
# overlay_boxes.py — 真值抽检图（把 bbox_2d 画到成品图上，出拼图供人工目检）
# 干什么：读逐帧真值 + 8bit 成品图，按类别画框与标签，输出逐帧图与一张 contact sheet；
#         用于表 11「真值质量：二十帧人工抽检」与框/像素对齐目检。
# 怎么跑：python tools/overlay_boxes.py --meta /data/render/meta --images /data/render/final \
#              --out /data/render/overlay --limit 20 --cols 4
# 成功判据：out 下 contact_sheet.jpg 与 limit 张逐帧 jpg；框位置与目标边缘偏差可直接目检。
# ⚠️ 只用于人工抽检，不参与训练；框坐标按真值 width/height 换算到实际图像尺寸。
import argparse
import json
import os
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:                      # noqa: N816
    sys.exit("抽检图需要 Pillow，当前 Python（%s）没有装：\n"
             "  Ubuntu：sudo apt install python3-pil\n  Windows：pip install pillow\n" % sys.executable)

# 类别 → 框颜色（RGB）；与语义模板值无关，只为目检好区分
CLASS_COLORS = {
    "road": (120, 120, 120),
    "vehicle": (0, 160, 255),
    "pedestrian": (255, 210, 0),
    "traffic_sign": (255, 60, 60),
    "cone": (255, 140, 0),
    "vegetation": (60, 200, 90),
}
DEFAULT_COLOR = (255, 255, 255)


def load_frames(meta_dir):
    frames = []
    for fn in sorted(os.listdir(meta_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(meta_dir, fn), encoding="utf-8") as f:
                fr = json.load(f)
        except Exception:
            continue
        if isinstance(fr, dict) and fr.get("frame_id"):
            frames.append(fr)
    return frames


def draw_boxes(fr, image_path, out_path):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    mw = float(fr.get("width") or im.width)
    mh = float(fr.get("height") or im.height)
    sx, sy = im.width / mw, im.height / mh          # 真值像素坐标 → 实际图像坐标
    for ob in fr.get("objects") or []:
        b = ob.get("bbox_2d")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        cls = ob.get("class")
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = "%s %.0fm" % (cls, float(ob.get("distance") or 0))
        draw.text((x1 + 3, y1 + 2), label, fill=color)
    im.save(out_path, quality=90)
    return im


def contact_sheet(images, cols, out_path, thumb_w=480):
    if not images:
        return None
    thumb_h = int(round(images[0].height * thumb_w / float(images[0].width)))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w + (cols - 1) * 6,
                              rows * thumb_h + (rows - 1) * 6), (18, 22, 28))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im.resize((thumb_w, thumb_h), Image.LANCZOS),
                    (c * (thumb_w + 6), r * (thumb_h + 6)))
    sheet.save(out_path, quality=88)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20, help="最多抽检帧数（表 11 建议 20）")
    ap.add_argument("--cols", type=int, default=4, help="拼图列数")
    a = ap.parse_args()
    if not os.path.isdir(a.meta):
        raise SystemExit("meta 目录不存在: %s" % a.meta)
    os.makedirs(a.out, exist_ok=True)
    frames = load_frames(a.meta)
    if a.limit > 0:
        frames = frames[:a.limit]
    drawn = []
    skipped = 0
    for fr in frames:
        name = "%s.png" % fr["frame_id"]
        src = os.path.join(a.images, name)
        if not os.path.exists(src):
            skipped += 1
            continue
        out = os.path.join(a.out, "%s.jpg" % fr["frame_id"])
        try:
            drawn.append(draw_boxes(fr, src, out))
        except Exception as e:
            skipped += 1
            print("WARN 跳过 %s: %s" % (name, e), file=sys.stderr)
    if not drawn:
        raise SystemExit("没有可抽检的帧（成品图缺失？）")
    sheet = contact_sheet(drawn, max(1, a.cols), os.path.join(a.out, "contact_sheet.jpg"))
    print("抽检图 %d 张 / 跳过 %d → %s（拼图 %s）" % (
        len(drawn), skipped, a.out, os.path.basename(sheet)))


if __name__ == "__main__":
    main()
