# -*- coding: utf-8 -*-
# postprocess.py — 后处理七步（方案 4.5，系统 Python、纯 CPU、退出引擎后执行）
# 干什么：16bit 线性图 → 曝光/sRGB → 读出噪声 → 暗角 → 色散 → 运动模糊 → 域对齐 → 8bit 成品。
# 怎么跑：python postprocess.py --raw /data/render/raw --meta /data/render/meta \
#              --out /data/render/final --params params.json [--real /实采图目录] [--limit N] [--seed 20261113]
# 成功判据：out 下每帧一张 <frame_id>.png（与 meta/<frame_id>.json 同主文件名）；
#           params_log.jsonl 每帧一条含全部退化参数、位深与工况标签；重跑跳过已有成品。
#          退出码 0 = 全部成功；1 = 有帧失败（已写错误记录，批次未中断）。
# ⚠️ 顺序固定、参数逐帧落日志（可回滚可复现）；线性原图与真值**永不覆盖**（分布变了只重跑本脚本）。
# ⚠️ 16bit RGB PNG 必须用 OpenCV 读：Pillow 对 16bit RGB 会静默截断成 8bit（实测 Pillow 12），
#    静默截断比报错更糟，所以这里缺失 cv2 时直接报错并给出安装命令。深度通道的 16bit 灰度
#    Pillow 本身支持，不需要 cv2。
import argparse
import json
import os
import struct
import sys
import zlib

# 依赖缺失时给一句能照做的话，而不是 traceback（线上系统 python3 可能没装 numpy/Pillow）
_DEP_HINT = ("  Ubuntu：sudo apt install python3-numpy python3-pil python3-opencv\n"
             "  Windows：pip install numpy pillow opencv-python\n")
try:
    import numpy as np
except ImportError:                      # noqa: N816
    sys.exit("后处理需要 numpy，当前 Python（%s）没有装：\n%s" % (sys.executable, _DEP_HINT))
try:
    from PIL import Image
except ImportError:                      # noqa: N816
    sys.exit("后处理需要 Pillow，当前 Python（%s）没有装：\n%s" % (sys.executable, _DEP_HINT))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
try:                      # 可选依赖：16bit PNG 与 EXR 用到；缺失时明确报错
    import cv2
except Exception:         # pragma: no cover - 环境相关
    cv2 = None

# 默认参数（表：白天噪声 0.001~0.003、黄昏 0.003~0.006；暗角 0.2~0.35；色散边缘 1~2px；运动模糊 0~13px）
DEFAULT_PARAMS = {
    "exposure_ev": 0.0,          # 曝光值（EV），工况表未命中时使用
    "read_noise": 0.002,         # 读出噪声方差
    "vignette": 0.25,            # 暗角强度 0.2~0.35
    "ca_edge_px": 1.5,           # 色散边缘偏移像素 0.5~2
    "blur_px": 0.0,              # 运动模糊核长 0~13（任务行带 speed 时按 speed 覆盖）
    "domain_align": True,        # 与实采图做通道均值方差颜色迁移
}

# 光照工况 → 退化参数（方案 4.5 的取值区间；仅填任务行 lighting.condition 能命中的键）
CONDITION_PARAMS = {
    "front_light": {"exposure_ev": 0.0, "read_noise": 0.0015, "vignette": 0.22, "ca_edge_px": 1.0},
    "back_light":  {"exposure_ev": -0.3, "read_noise": 0.0020, "vignette": 0.25, "ca_edge_px": 2.0},
    "dusk":        {"exposure_ev": 0.3, "read_noise": 0.0045, "vignette": 0.32, "ca_edge_px": 1.5},
    "night":       {"exposure_ev": 1.2, "read_noise": 0.0025, "vignette": 0.35, "ca_edge_px": 1.8},
    "rain":        {"exposure_ev": 0.0, "read_noise": 0.0030, "vignette": 0.30, "ca_edge_px": 1.2},
}

# 运动模糊核长 = 车速(m/s) × 本系数，裁到 0~13px。系数按「30fps、180° 快门、近景 10m」粗算，
# 正式批次要拿实采图上的标志牌边缘做一次标定再固化（方案 4.5 只给了 0~13px 的区间）。
BLUR_PX_PER_MPS = 0.45


class RawFormatError(Exception):
    """原始图格式/依赖不满足：宁可报错，不做静默降级。"""


def png_info(path):
    """读 PNG IHDR：返回 (width, height, bit_depth, color_type)；不是 PNG 直接抛。"""
    with open(path, "rb") as f:
        head = f.read(26)
    if len(head) < 26 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise RawFormatError("不是 PNG 文件（深度/语义通道当前只按 PNG 处理）: %s" % path)
    width, height = struct.unpack(">II", head[16:24])
    return width, height, head[24], head[25]


def load_linear_rgb(path):
    """读线性 RGB，返回 (HxWx3 float32 归一化到 0~1, 位深)。

    8bit：Pillow；16bit：OpenCV（IMREAD_UNCHANGED，BGR→RGB）。位深按 IHDR 判定，
    不再用「像素最大值是否 >255」猜——夜间/弱光线性图的 max 可能小于 255。
    """
    if path.lower().endswith(".exr"):
        if cv2 is None:
            raise RawFormatError("EXR 需要 OpenCV 解码: %s" % path)
        bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] < 3 or bgr.dtype not in (np.float16, np.float32):
            raise RawFormatError("OpenCV 无法解码线性 RGB EXR: %s" % path)
        return np.ascontiguousarray(bgr[..., :3][..., ::-1], dtype=np.float32), 16 if bgr.dtype == np.float16 else 32
    _w, _h, bit_depth, color_type = png_info(path)
    if bit_depth == 8:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return arr, 8
    if bit_depth != 16:
        raise RawFormatError("不支持的 PNG 位深 %d（只支持 8/16）: %s" % (bit_depth, path))
    if color_type not in (2, 6):
        raise RawFormatError("16bit RGB 通道要求 color_type=2/6，实际 %d: %s" % (color_type, path))
    if cv2 is None:
        raise RawFormatError(
            "16bit RGB PNG 需要 OpenCV 解码（Pillow 会静默截断成 8bit）。"
            "Ubuntu: sudo apt install python3-opencv ；Windows: pip install opencv-python 。文件: %s" % path)
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RawFormatError("OpenCV 读不了该 PNG: %s" % path)
    if bgr.dtype != np.uint16:
        raise RawFormatError("OpenCV 返回 %s 而非 uint16（该版本不支持 16bit PNG）: %s"
                             % (bgr.dtype, path))
    if bgr.ndim == 2:
        raise RawFormatError("16bit 灰度图不能当 RGB 通道用: %s" % path)
    rgb = np.ascontiguousarray(bgr[..., :3][..., ::-1], dtype=np.float32) / 65535.0
    return rgb, 16


def srgb_gamma(x):
    """线性 → sRGB 传输曲线（第 1 步：曝光缩放后施加伽马，模拟相机 ISP 色调曲线）。"""
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(np.clip(x, 0, None), 1 / 2.4) - 0.055)


def add_read_noise(x, var, rng):
    """第 2 步：读出噪声，方差按等效感光度取值。"""
    if var <= 0:
        return x
    return x + rng.normal(0.0, np.sqrt(var), x.shape)


def vignette(x, strength):
    """第 3 步：径向亮度衰减。"""
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    return x * (1.0 - strength * np.clip(r, 0, 1) ** 2)[..., None]


def chromatic(x, edge_px):
    """第 4 步：按半径对 RGB 做亚像素偏移（径向 UV 重映射近似色散）。"""
    if edge_px <= 0:
        return x
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    r = np.maximum(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2), 1e-6)
    rn = r / r.max()
    out = x.astype(np.float32).copy()
    for ch, sign in ((0, -1.0), (2, 1.0)):     # R 外推、B 内收，G 不动
        dx = (sign * edge_px * rn * (xx - cx) / r).astype(np.int32)
        dy = (sign * edge_px * rn * (yy - cy) / r).astype(np.int32)
        sx = np.clip(xx + dx, 0, w - 1).astype(np.int32)
        sy = np.clip(yy + dy, 0, h - 1).astype(np.int32)
        out[..., ch] = x[..., ch][sy, sx]
    return out


def motion_blur(x, px):
    """第 5 步：核长 = 车速 × 系数换算，范围 0~13。默认水平核（前视行驶方向），积分图实现免 scipy。

    积分图滑窗：宽 = w + 2*pad（pad = k//2），对每个输出列取 k 个样本的平均值；
    注意窗口数必须正好 w 个（旧实现漏了首列，输出宽度变成 w-1）。
    """
    px = float(px or 0)
    if px < 0.5:
        return x
    k = int(min(13, max(1, round(px))))
    if k % 2 == 0:
        k -= 1
    if k <= 1:
        return x
    pad = k // 2
    h, w = x.shape[:2]
    xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    csum = np.cumsum(xp, axis=1, dtype=np.float64)
    out = np.empty((h, w, x.shape[2]), dtype=np.float64)
    out[:, 0] = csum[:, k - 1]                      # 第一列没有左侧前驱
    if w > 1:
        out[:, 1:] = csum[:, k:k - 1 + w] - csum[:, 0:w - 1]
    return out / float(k)


def domain_align(x, ref_mean, ref_std):
    """第 6 步：通道均值方差颜色迁移（简单颜色迁移；直方图匹配可后续替换，参数同样落日志）。"""
    m = x.reshape(-1, x.shape[-1]).mean(0)
    s = x.reshape(-1, x.shape[-1]).std(0)
    s = np.where(s < 1e-6, 1e-6, s)
    return (x - m) / s * np.maximum(ref_std, 1e-6) + ref_mean


def load_ref_stats(ref_dir):
    """实采图通道统计（一次性算好，逐帧复用）。"""
    vals = []
    files = sorted(os.listdir(ref_dir))[:200]     # 200 张足够稳定，别全量扫
    for fn in files:
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        a = np.asarray(Image.open(os.path.join(ref_dir, fn)).convert("RGB"),
                       dtype=np.float32) / 255.0
        vals.append(a.reshape(-1, 3))
    if not vals:
        raise SystemExit("实采图目录无图像: %s" % ref_dir)
    v = np.concatenate(vals)
    return v.mean(0), v.std(0)


def frame_rng(stem, seed):
    """逐帧确定性随机源：用 crc32 而不是内置 hash（内置 hash 每个进程随机化，重跑结果会变）。"""
    key = (zlib.crc32(stem.encode("utf-8")) ^ (int(seed) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return np.random.default_rng(key)


def condition_overrides(lighting):
    """按真值 JSON 的 lighting.condition 取工况参数；未命中返回空。"""
    if not isinstance(lighting, dict):
        return {}
    key = lighting.get("condition") or lighting.get("weather")
    return dict(CONDITION_PARAMS.get(str(key), {})) if key else {}


def resolve_params(explicit, lighting, speed):
    """参数优先级：DEFAULT < 工况表 < params.json 显式键 < 由 speed 换算的模糊核长。

    返回 (effective, source)：source 说明工况是否命中，写进逐帧日志便于回溯。
    """
    eff = dict(DEFAULT_PARAMS)
    cond = condition_overrides(lighting)
    source = "default"
    if cond:
        eff.update(cond)
        source = "condition:%s" % (lighting.get("condition") or lighting.get("weather"))
    if explicit:
        eff.update(explicit)
        source += "+explicit"
    if speed and speed > 0:
        eff["blur_px"] = min(13.0, max(0.0, float(speed) * BLUR_PX_PER_MPS))
        source += "+speed"
    eff["blur_px"] = min(13.0, max(0.0, float(eff["blur_px"])))
    return eff, source


def process_one(src, dst, p, ref, rng):
    """单帧七步；返回实际读取的位深（写进参数日志）。"""
    x, bit_depth = load_linear_rgb(src)
    x = srgb_gamma(x * (2.0 ** float(p["exposure_ev"])))     # 第 1 步：曝光 + sRGB
    x = add_read_noise(x, float(p["read_noise"]), rng)       # 第 2 步
    x = vignette(x, float(p["vignette"]))                    # 第 3 步
    x = chromatic(x, float(p["ca_edge_px"]))                 # 第 4 步
    x = motion_blur(x, float(p["blur_px"]))                  # 第 5 步
    if p.get("domain_align") and ref is not None:            # 第 6 步
        x = domain_align(x, *ref)
    x = np.clip(x, 0, 1)
    Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(dst)   # 第 7 步：8bit 落盘
    return bit_depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="线性图目录（含 *_rgb.png 或 *_rgb.exr）")
    ap.add_argument("--meta", required=True, help="逐帧真值 JSON 目录（同主文件名）")
    ap.add_argument("--out", required=True, help="8bit 成品目录（写 <frame_id>.png）")
    ap.add_argument("--params", default="", help="参数 json（显式键优先于工况表）")
    ap.add_argument("--real", default="", help="实采图目录（域对齐参照；留空则跳过第 6 步）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261113, help="逐帧确定性噪声种子")
    a = ap.parse_args()
    if not os.path.isdir(a.raw):
        raise SystemExit("raw 目录不存在: %s" % a.raw)
    explicit = {}
    if a.params:
        with open(a.params, encoding="utf-8") as f:
            raw_params = json.load(f)
        unknown = [k for k in raw_params if k not in DEFAULT_PARAMS]
        if unknown:
            print("WARN 忽略未知参数键: %s" % ", ".join(unknown), file=sys.stderr)
        explicit = {k: v for k, v in raw_params.items() if k in DEFAULT_PARAMS}
    os.makedirs(a.out, exist_ok=True)
    # 同一帧两种格式并存时优先保留高动态范围 EXR。
    by_stem = {}
    for f in sorted(os.listdir(a.raw)):
        if f.endswith(("_rgb.png", "_rgb.exr")):
            stem = f[:-8]
            if stem not in by_stem or f.endswith(".exr"):
                by_stem[stem] = f
    frames = [by_stem[k] for k in sorted(by_stem)]
    if a.limit:
        frames = frames[:a.limit]
    if not frames:
        raise SystemExit("raw 目录没有 *_rgb.png 或 *_rgb.exr: %s" % a.raw)

    # 先探查首帧位深：16bit 缺 OpenCV 时在批处理前给出可执行的处置，而不是每帧刷同一条错误
    first = os.path.join(a.raw, frames[0])
    first_depth = 16 if first.endswith(".exr") else png_info(first)[2]
    if first_depth == 16 and cv2 is None:
        raise SystemExit(
            "首帧是 16bit RGB PNG 或 EXR，但当前 Python 没有 OpenCV。\n"
            "  Ubuntu: sudo apt install python3-opencv\n"
            "  Windows: pip install opencv-python\n"
            "  （深度通道的 16bit 灰度不受影响；只有 RGB 16bit 需要）")

    ref = None
    if a.real:
        if explicit.get("domain_align", DEFAULT_PARAMS["domain_align"]):
            ref = load_ref_stats(a.real)
        else:
            print("WARN domain_align=false，忽略 --real", file=sys.stderr)
    else:
        if explicit.get("domain_align", DEFAULT_PARAMS["domain_align"]):
            print("WARN 未给 --real，本批跳过第 6 步域对齐（参数日志会写明）", file=sys.stderr)

    log_path = os.path.join(a.out, "params_log.jsonl")
    done = skipped = failed = 0
    depth_stat = {}
    with open(log_path, "a", encoding="utf-8") as logf:
        for fn in frames:
            stem = fn[:-len("_rgb.png")]
            src = os.path.join(a.raw, fn)
            dst = os.path.join(a.out, "%s.png" % stem)
            if os.path.exists(dst):        # 已存在则跳过（续跑）
                skipped += 1
                continue
            try:
                meta_p = os.path.join(a.meta, "%s.json" % stem)
                meta = {}
                if os.path.exists(meta_p):
                    with open(meta_p, encoding="utf-8") as f:
                        meta = json.load(f)
                lighting = meta.get("lighting") or {}
                speed = float(meta.get("speed") or 0.0)
                params, source = resolve_params(explicit, lighting, speed)
                if ref is None:            # 没有参照图时不做域对齐，日志里要保持真实
                    params["domain_align"] = False
                bit_depth = process_one(src, dst, params, ref, frame_rng(stem, a.seed))
                depth_stat[bit_depth] = depth_stat.get(bit_depth, 0) + 1
                logf.write(json.dumps({
                    "frame": stem, "src": os.path.basename(src), "dst": os.path.basename(dst),
                    "bit_depth": bit_depth, "condition": lighting.get("condition"),
                    "lighting_source": source, "speed_mps": speed,
                    "seed": a.seed, "params": params,
                }, ensure_ascii=False) + "\n")
                logf.flush()
                done += 1
            except Exception as e:         # 单帧失败不断批，记日志跳过
                failed += 1
                logf.write(json.dumps({"frame": stem, "error": str(e)[:300]},
                                      ensure_ascii=False) + "\n")
                logf.flush()
                print("FAIL 帧 %s: %s" % (stem, e), file=sys.stderr)
    print("完成 new=%d skip=%d fail=%d → %s（参数日志 %s，位深分布 %s）" % (
        done, skipped, failed, a.out, log_path, depth_stat))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
