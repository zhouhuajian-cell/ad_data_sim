# -*- coding: utf-8 -*-
# voc_export.py — 逐帧真值 JSON → PASCAL VOC XML（方案模块 5：训练用 VOC 标签）
# 干什么：读 meta/<frame_id>.json，为每帧写一份 VOC 标注 Annotations/<frame_id>.xml：
#   标准 VOC 结构（annotation / folder / filename / path / source / size{w,h,depth} / segmented / object*），
#   每个 object 的 <bndbox> 之后追加自定义扩展节点 distance / occlusion / visible_ratio / light_level /
#   weather / wear_level / difficult_level（Pascal VOC / mmdetection / ultralytics / labelImg 解析器都只读
#   自己认识的节点，未知节点被忽略 —— 加了不破坏兼容，这是方案要求的「带扩展节点的 VOC」）。
# 怎么跑：python post/voc_export.py --meta /data/render/meta --images /data/render/final \
#              --out /data/render/voc [--limit N] [--no-normalize] [--clear]
#   产物：<out>/Annotations/<frame_id>.xml          每帧一份
#         <out>/ImageSets/Main/{train}.txt          全量帧名清单（训练脚本吃这个）
# 成功判据：
#   * XML 份数 == 可用帧数 == ImageSets/Main/train.txt 行数；每份能被 xml.etree 正常解析；
#   * 每个 <object> 的 bndbox 都是整数、1 <= xmin < xmax <= width、1 <= ymin < ymax <= height；
#   * 图像缺失的帧跳过并 WARN（给了 --images 时）；退化框（<1px）跳过并计数；
#   * 退出码 0 = 全部写成；空目录/没有帧 → 退出码 2 且不产出半套 XML。
# 坐标口径（必须写清楚，否则训练框会整体偏 1 像素）：
#   真值 bbox_2d 是 [xmin,ymin,xmax,ymax]，浮点像素、左上原点、**右开**（x2 那一列像素不算进框、也不画出来）。
#   VOC 是 1-based、**含端点**的整数像素：xmin = floor(xmin_true) + 1，xmax = ceil(xmax_true)。
#   例：xmax_true = 120.0 → 框覆盖第 0..119 列 → VOC xmax = 120（不越界、不多盖一列）。
#   裁剪：先按 width/height 夹紧（浮点），再转整数；任一边不足 1px 的退化框直接跳过。
# ⚠️ 不做归一：字段归一/难例判定全在 common/truth_schema.py，本脚本只负责「取字段 + 写 XML + 判窗口」。
#    加 --no-normalize 时直接读原始字段（旧字段 class/bbox_2d/occlusion），用于「归一前后对照」。
# 兼容：Python 3.8；只依赖标准库。
import argparse
import json
import os
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common import truth_schema as ts  # noqa: E402  字段表/归一/难例判定的唯一口径

VOC_FOLDER = "JPEGImages"       # VOC 约定：图片放 JPEGImages/（PNG 也照样放里面，VOC 解析器不看扩展名）
VOC_DEPTH = 3                   # RGB 三通道；深度/语义图不参与 VOC 标注
FALLBACK_SIZE = (1920, 1080)    # 与 pack_dataset.FALLBACK_SIZE 一致：帧里没有 width/height 时的兜底


def load_frames(meta_dir):
    """读逐帧真值；跳过没有 frame_id 的 JSON（params_log / settings 之类）。"""
    frames = []
    skipped = 0
    for fn in sorted(os.listdir(meta_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(meta_dir, fn), "r", encoding="utf-8") as f:
                fr = json.load(f)
        except Exception:
            skipped += 1
            continue
        if not isinstance(fr, dict) or not fr.get("frame_id"):
            skipped += 1
            continue
        frames.append(fr)
    if skipped:
        print("WARN 跳过 %d 个非帧 JSON（无 frame_id 或解析失败）" % skipped, file=sys.stderr)
    return frames


def frame_size(fr):
    """帧尺寸 → (int w, int h)，兜底 1920x1080（与 pack_dataset 同口径）。"""
    try:
        w = int(fr.get("width") or FALLBACK_SIZE[0])
        h = int(fr.get("height") or FALLBACK_SIZE[1])
    except (TypeError, ValueError):
        return FALLBACK_SIZE
    return max(1, w), max(1, h)


def image_name(fr):
    """成品图文件名：与 meta 同主文件名（postprocess 的命名契约）。"""
    return "%s.png" % fr["frame_id"]


def image_candidates(fr):
    """图表候选名（用于 --images 核对），按优先级排在前面的是后处理成品：
      1) <frame_id>.png                  —— post/postprocess.py 的成品（首选）
      2) image_files.rgb 的文件名         —— batch_capture 写的原始 16bit RGB（没跑后处理时）
    Windows 不区分大小写、Ubuntu 区分，所以是「存在性核对」而不是精确匹配一条路径。
    """
    names = ["%s.png" % fr["frame_id"]]
    rgb = (fr.get("image_files") or {}).get("rgb") if isinstance(fr.get("image_files"), dict) else None
    if rgb and rgb not in names:
        names.append(str(rgb))
    return names


def find_image(image_dir, fr):
    """在图片目录里找到该帧的图：先找成品、再找原始 RGB；找不到返回 None。

    目录不存在时直接返回 None（调用方给 WARN）——不在这里抛异常，单帧缺图不该打断整批导出。
    """
    for name in image_candidates(fr):
        if os.path.exists(os.path.join(image_dir, name)):
            return name
    return None


def to_voc_box(bbox, width, height):
    """真值浮点框 [x1,y1,x2,y2]（右开）→ VOC 整数含端点框 [xmin,ymin,xmax,ymax]。

    返回 None 表示退化框（裁到画面后任一边不足 1px）：这种框对检测训练无意义，也过不了
    VOC 校验工具（要求 xmin<xmax），直接跳过。
    """
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x2 < x1:                 # 容忍左右写反，不宽容就静默丢框
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(x1, float(width)))      # 夹进画面
    x2 = max(0.0, min(x2, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    xmin = int(x1 // 1) + 1                   # 第一个被覆盖的像素（1-based）
    xmax = int(-((-x2) // 1))                 # ceil：最后一个被覆盖像素的 1-based 序号
    ymin = int(y1 // 1) + 1
    ymax = int(-((-y2) // 1))
    xmin = max(1, min(xmin, width))
    xmax = max(1, min(xmax, width))
    ymin = max(1, min(ymin, height))
    ymax = max(1, min(ymax, height))
    if xmin >= xmax or ymin >= ymax:          # 夹紧后仍退化（画面极窄时可能发生）
        return None
    return xmin, ymin, xmax, ymax


def _add(parent, tag, text):
    """追加一个文本节点（None / 空串写成空节点）。"""
    child = ET.SubElement(parent, tag)
    child.text = "" if text is None else str(text)
    return child


def object_element(ob):
    """归一后的目标 → <object> 元素（含扩展节点与 difficult_level）。

    VOC 标准节点全部按官方顺序写；扩展节点统一放在 <bndbox> 之后 —— 官方工具与大多数解析器
    （xml.etree、labelImg、mmdetection VOCDataset）都是「按标签名找」，不关心顺序，
    唯有一些严格 schema 校验器会告警「未知节点」，所以默认不丢弃、也不改名。
    """
    obj = ET.Element("object")
    _add(obj, "name", ob.get("class_name"))
    _add(obj, "pose", "Unspecified")
    _add(obj, "truncated", 1 if ob.get("truncated") else 0)
    # ⚠️口径(d)：VOC 的 <difficult> 只有 0/1，difficult_level 是 0~3 → 这里只映射 >0；
    #   原值另写在扩展节点 <difficult_level> 里，需要分级时读它。
    lvl = int(ob.get("difficult_level") or 0)
    _add(obj, "difficult", 1 if lvl > 0 else 0)
    box = ET.SubElement(obj, "bndbox")
    for tag, val in zip(("xmin", "ymin", "xmax", "ymax"), ob["voc_box"]):
        _add(box, tag, int(val))
    # —— 以下为自定义扩展节点（VOC 解析器忽略）——
    _add(obj, "distance", "%.3f" % float(ob.get("distance") or 0.0))
    _add(obj, "occlusion", "%.4f" % float(ob.get("occlusion_ratio") or 0.0))
    _add(obj, "visible_ratio", "%.4f" % float(ob.get("visible_ratio") or 0.0))
    _add(obj, "light_level", ob.get("light_level"))
    _add(obj, "weather", ob.get("weather"))
    _add(obj, "wear_level", "%.4f" % float(ob.get("wear_level") or 0.0))
    _add(obj, "difficult_level", lvl)
    _add(obj, "class_id", int(ob.get("class_id") or 0))
    if ob.get("instance_id") is not None:
        _add(obj, "instance_id", int(ob.get("instance_id") or 0))
    return obj


def build_annotation(fr, norm, image_dir="", check_image=True, img_name=None):
    """一帧真值 → (ET.ElementTree, 统计 dict)。

    统计 dict：objects（写进 XML 的对象数） / degenerate（跳过框数） / missing（是否因缺图跳过）。
    img_name 是 --images 核对到的实际文件名（缺省用成品名 <frame_id>.png）。
    """
    frame_id = str(fr["frame_id"])
    width, height = frame_size(fr)
    stats = {"objects": 0, "degenerate": 0, "missing": False}
    fname = img_name or image_name(fr)

    img_dir_attr = VOC_FOLDER
    if image_dir:
        img_dir_attr = os.path.basename(os.path.normpath(image_dir)) or VOC_FOLDER

    root = ET.Element("annotation")
    _add(root, "folder", img_dir_attr)
    _add(root, "filename", fname)
    _add(root, "path", os.path.abspath(os.path.join(image_dir or ".", fname)))
    source = ET.SubElement(root, "source")
    _add(source, "database", "UE5 Synth")
    size = ET.SubElement(root, "size")
    _add(size, "width", width)
    _add(size, "height", height)
    _add(size, "depth", VOC_DEPTH)
    _add(root, "segmented", 0)
    # 帧级扩展：标明光照/天气来源，便于按工况切子集训练
    _add(root, "light_level", norm.get("light_level"))
    _add(root, "weather", norm.get("weather"))

    if check_image and image_dir and not find_image(image_dir, fr):
        stats["missing"] = True
        return ET.ElementTree(root), stats

    for ob in norm.get("objects") or []:
        box = to_voc_box(ob.get("bbox") or ob.get("bbox_2d"), width, height)
        if box is None:
            stats["degenerate"] += 1
            continue
        item = dict(ob)
        item["voc_box"] = box
        root.append(object_element(item))
        stats["objects"] += 1
    return ET.ElementTree(root), stats


def element_bytes(root):
    """元素树 → 带声明的缩进 XML 字节串（给人工抽检看；解析器对缩进/空节点都无所谓）。

    ElementTree 自带的 write 不缩进（整份 XML 一行，人没法读），这里用标准库 minidom 只做格式化：
    先 tostring 成单行，再 toprettyxml 加缩进，去掉空行；编码固定 utf-8。
    """
    raw = ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    lines = [ln for ln in pretty.splitlines() if ln.strip()]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_file(tree, path):
    """落盘：先写 .tmp 再原子替换（kill 中断不留半个 XML 被下游当完整读）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(element_bytes(tree.getroot()))
    os.replace(tmp, path)


def normalize_frames(frames, enabled=True):
    """整帧归一（或原样返回，仅补出 object 里的字段名）。"""
    if not enabled:
        return frames
    return [ts.normalize_frame(fr) for fr in frames]


def main(argv=None):
    ap = argparse.ArgumentParser(description="逐帧真值 JSON → PASCAL VOC XML")
    ap.add_argument("--meta", required=True, help="逐帧真值 JSON 目录")
    ap.add_argument("--images", default="", help="成品图目录（给定时核对图片存在性，缺图的帧跳过）")
    ap.add_argument("--out", default="", help="输出目录（默认 <meta 同级>/voc）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 帧（0=全部）")
    ap.add_argument("--no-normalize", action="store_true",
                    help="不调用 truth_schema.normalize_frame（直接按旧字段写，用于对照）")
    ap.add_argument("--clear", action="store_true", help="导出前清空 <out>/Annotations 里的旧 XML")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.meta):
        sys.stderr.write("真值目录不存在：%s\n" % a.meta)
        return 2
    if a.images and not os.path.isdir(a.images):
        sys.stderr.write("图片目录不存在：%s（去掉 --images 可跳过图片核对）\n" % a.images)
        return 2
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.meta)), "voc")
    ann_dir = os.path.join(out, "Annotations")
    sets_dir = os.path.join(out, "ImageSets", "Main")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(sets_dir, exist_ok=True)
    if a.clear:
        for fn in os.listdir(ann_dir):
            if fn.endswith(".xml") or fn.endswith(".xml.tmp"):
                os.remove(os.path.join(ann_dir, fn))

    frames = load_frames(a.meta)
    if a.limit and a.limit > 0:
        frames = frames[:a.limit]
    if not frames:
        sys.stderr.write("没有可用帧（meta 下没有真值 JSON）：%s\n" % a.meta)
        return 2

    n_xml = n_obj = n_deg = n_missing = 0
    written = []
    for fr, norm in zip(frames, normalize_frames(frames, not a.no_normalize)):
        img_name = None
        if a.images:
            img_name = find_image(a.images, fr)
            if img_name is None:
                n_missing += 1
                print("WARN 缺成品图，跳过该帧: %s（找过 %s）"
                      % (fr["frame_id"], " / ".join(image_candidates(fr))), file=sys.stderr)
                continue
        tree, stats = build_annotation(fr, norm, a.images, img_name=img_name)
        write_file(tree, os.path.join(ann_dir, "%s.xml" % fr["frame_id"]))
        written.append(str(fr["frame_id"]))
        n_xml += 1
        n_obj += stats["objects"]
        n_deg += stats["degenerate"]

    if not written:
        sys.stderr.write("没有写出任何 XML（全部帧缺图或为空）\n")
        return 2
    for name in ("train", "val", "test"):        # VOC 三件套都给一份全量，切分交给 pack_dataset
        with open(os.path.join(sets_dir, "%s.txt" % name), "w", encoding="utf-8") as f:
            f.write("\n".join(written) + "\n")
    print("VOC: %d 帧 / %d 个 object → %s（退化框跳过 %d，缺图跳过 %d）" % (
        n_xml, n_obj, ann_dir, n_deg, n_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
