# -*- coding: utf-8 -*-
# batch_capture.py — 引擎内批量捕获（放进 UE 工程的 Content/Python/ 下）
# 干什么：按任务清单逐帧执行「加载关卡 → 清场并生成/随机化目标 → 摆相机调光照 → 四通道捕获 → 写逐帧 JSON」，
#   断点续跑（done 标记 + 输出已存在跳过），单帧异常写错误日志后继续下一条（方案 3.5 四条工程约束）。
# 怎么跑：由 linux/run_batch.sh 用 UnrealEditor-Cmd -ExecutePythonScript 调起，绝不能加 -headless（那是 NullRHI）：
#   UnrealEditor-Cmd <xxx.uproject> -ExecutePythonScript=<本文件> \
#     -RenderOffscreen -unattended -nosplash -noSound -NoLiveView -Vulkan -log=/data/logs/batch.log
# 成功判据：/data/render/raw 下每帧 <frame_id>_rgb.png / _depth.png / _sem.png 与 /data/render/meta 下
#   <frame_id>.json 齐全、/data/queue/done 下 <task_id>.done 齐全；人为 kill 后重跑，已完成项被跳过。
#
# ⚠️ 未经编辑器验证（本机没有 UE5 与 N 卡，只在引擎外做过纯逻辑验证）——以下两处必须在阶段三 T0
#   对着真实工程逐条核对，文件内所有 "# TODO(阶段三)" 都指向待核对点：
#   1) RGB 16bit 线性落盘：SceneCapture2D → FloatRGBA(RTF_RGBA16F) → PNG 这条链里，
#      UImageWriteBlueprintLibrary 是否真的保留 16bit 分量、是否悄悄做了 sRGB/色调映射，未验证；
#      若最终只能靠 RenderingLibrary.export_render_target 落盘（通常是 8bit/压缩），16bit 承诺不成立，
#      必须改走自定义后处理材质 + ImageWriteQueue，或退回「引擎只出 RT、由系统 Python 读 RT 存盘」的方案。
#   2) 语义模板值：CustomDepth stencil 能否被 SceneCapture2D 的后期材质读到、天空是否稳定拿到 255、
#      项目设置是否 r.CustomDepth=3，均未验证。模板值表必须与 post/pack_dataset.py 的类别顺序同源。
#
# ⚠️ 职责边界（方案 3.5 / 5.2 双 Python 环境）：引擎内只做调度与落盘，禁止 numpy/OpenCV 与任何图像重计算；
#   16bit→8bit、曝光、噪声、暗角、色散、运动模糊、颜色迁移全部留给 post/postprocess.py（系统 Python）。
# 兼容：UE 5.3 内置 Python 3.9（不用 3.10+ 语法）；只依赖标准库 + unreal + 同目录 capture_math。
import datetime
import json
import os
import random
import sys
import time
import traceback
import zlib

# 同目录的纯数学模块（相机基向量/投屏/投影/裁切/深度归一化/遮挡近似）。
# -ExecutePythonScript 场景下 __file__ 存在，但脚本目录不一定在 sys.path，这里显式补一次。
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _HERE and _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import unreal  # noqa: E402  引擎内才有；引擎外测试时由测试脚本向 sys.modules 注入假模块
from capture_math import (  # noqa: E402  投影/归一化只用这一份，禁止在引擎内重写一套
    clip_bbox, boxes_overlap_ratio, hfov_from_focal, normalize_depth, project_aabb,
)

# ── 目录与运行参数（全部可用环境变量覆盖；默认值对齐方案附录 A 的落盘结构）──────────────
QUEUE = os.environ.get("AD_UE_QUEUE", "/data/queue/tasks.jsonl")
RAW = os.environ.get("AD_UE_RAW", "/data/render/raw")
META = os.environ.get("AD_UE_META", "/data/render/meta")
DONE = os.environ.get("AD_UE_DONE", "/data/queue/done")
ERRLOG = os.environ.get("AD_UE_ERRLOG", "/data/logs/capture_errors.log")
SEED = int(os.environ.get("AD_UE_SEED", "20261113"))

DEFAULT_WIDTH = int(os.environ.get("AD_UE_WIDTH", "1920"))     # 方案表 6：12GB 显存下的稳定档位
DEFAULT_HEIGHT = int(os.environ.get("AD_UE_HEIGHT", "1080"))
BIT_DEPTH = 16                # RGB/深度都是 16bit（方案 3.4）
DEPTH_FAR = 200.0             # 深度归一化上限（米），与 capture_math.normalize_depth 的 far 一致
PROGRESS_EVERY = 20           # 每 20 帧打一次进度

# 语义模板值（方案 3.4）：类别 → 写进 CustomDepth stencil 的模板值。改这里必须同步 post/pack_dataset.py。
SEMANTIC_STENCIL = {"road": 1, "vehicle": 2, "pedestrian": 3,
                    "traffic_sign": 4, "cone": 5, "vegetation": 6, "sky": 255}

# 每类目标的默认尺寸（完整尺寸，米：X 长 / Y 宽 / Z 高）。仅在任务行没给 size 时兜底，
# 真实工程的资产尺寸应以任务行配置为准（config/ 由 Lead 维护）。
CLASS_DEFAULT_SIZE = {
    "vehicle": (4.60, 1.85, 1.55),
    "pedestrian": (0.55, 0.55, 1.72),
    "traffic_sign": (0.65, 0.10, 0.85),
    "cone": (0.38, 0.38, 0.70),
    "vegetation": (2.00, 2.00, 3.50),
    "road": (4.00, 20.00, 0.10),
}
# 目标抖动幅度默认值 (dx, dy, dz, dyaw)：单位米 / 度，可被任务行的 jitter[dx,dy,dz,dyaw] 覆盖。
# 只抖 X/Y 而不抖 Z（除非配置显式给了 dz），避免目标悬空或入地。
JITTER_DEFAULT = (2.0, 2.0, 0.0, 5.0)
MATERIAL_VARIANT_COUNT = 4    # 材质随机化档位数：从任务 rng 抽 0~3，具体映射见 TODO(阶段三)
NIGHT_APERTURE = 1.8          # 方案表 6：夜间 f/1.8、白天按 camera.aperture（默认 f/2.0）


def _log(msg):
    print("[capture] %s" % msg)
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════════
# 一、纯逻辑（不依赖 unreal）——本地可用假 unreal 模块直接测
# ══════════════════════════════════════════════════════════════════════════════════

def frame_id_of(row):
    """任务行 → 帧名。frame_id 与图像/JSON 同主文件名（方案附录 B），缺省退回 task id。"""
    row = row or {}
    return str(row.get("frame_id") or row.get("id") or "frame")


def stable_seed(base_seed, task_id):
    """跨进程稳定的种子：zlib.crc32（不能用内置 hash —— 它每个进程都加盐，重跑会变）。

    方案 3.5 第一条：随机种子固定且可复现。整批共用一个 base_seed，任务 id 参与哈希，
    因此「同一个任务行 → 同一个随机流」，与任务在队列里的先后顺序、并发与否都无关。
    """
    key = ("%s|%s" % (int(base_seed), str(task_id))).encode("utf-8")
    return zlib.crc32(key) & 0xFFFFFFFF


def task_rng(row):
    """任务行 → 随机数发生器（random.Random，不是全局 random）。"""
    row = row or {}
    base = row.get("seed", SEED)
    try:
        base = int(base)
    except (TypeError, ValueError):
        base = SEED
    tid = row.get("id") or row.get("frame_id") or ""
    return random.Random(stable_seed(base, tid))


def _as_float(value, default):
    """宽松取浮点：None/坏值 → default。任务行是外部 JSON，不能假设类型正确。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_vec3(value, default=(0.0, 0.0, 0.0)):
    """宽松取三维向量：支持 [x,y,z] / {"x","y","z"} / unreal.Vector（有 .x/.y/.z 属性）。"""
    if value is None:
        return (float(default[0]), float(default[1]), float(default[2]))
    if isinstance(value, dict):
        return (_as_float(value.get("x"), float(default[0])),
                _as_float(value.get("y"), float(default[1])),
                _as_float(value.get("z"), float(default[2])))
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return (float(value.x), float(value.y), float(value.z))
    try:
        seq = list(value)
    except TypeError:
        return (float(default[0]), float(default[1]), float(default[2]))
    if len(seq) >= 3:
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    if len(seq) == 2:
        return (float(seq[0]), float(seq[1]), float(default[2]))
    if len(seq) == 1:
        return (float(seq[0]), float(default[1]), float(default[2]))
    return (float(default[0]), float(default[1]), float(default[2]))


def _as_jitter(value, scalar_fallback=None):
    """抖动幅度归一化成 (dx, dy, dz, dyaw)。

    config 的 item.jitter 是 [dx,dy,dz,dyaw]（米/米/米/度）；也接受
    {"dx","dy","dz","dyaw"} / {"x","y","z","yaw"}；单个数（旧写法 jitter_m）按 X/Y 同幅处理、
    不抖 Z、yaw 抖 5°。缺省一律用 JITTER_DEFAULT。
    """
    if value is None:
        if scalar_fallback is None:
            return JITTER_DEFAULT
        amp = _as_float(scalar_fallback, JITTER_DEFAULT[0])
        return (amp, amp, 0.0, JITTER_DEFAULT[3])
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return (float(value.x), float(value.y), float(value.z),
                _as_float(getattr(value, "yaw", None), JITTER_DEFAULT[3]))
    if isinstance(value, dict):
        return (_as_float(value.get("dx", value.get("x")), JITTER_DEFAULT[0]),
                _as_float(value.get("dy", value.get("y")), JITTER_DEFAULT[1]),
                _as_float(value.get("dz", value.get("z")), JITTER_DEFAULT[2]),
                _as_float(value.get("dyaw", value.get("yaw")), JITTER_DEFAULT[3]))
    try:
        seq = list(value)
    except TypeError:
        amp = _as_float(value, JITTER_DEFAULT[0])
        return (amp, amp, 0.0, JITTER_DEFAULT[3])
    out = list(JITTER_DEFAULT[:len(JITTER_DEFAULT)])
    for i in range(min(len(seq), len(out))):
        out[i] = _as_float(seq[i], out[i])
    return tuple(out)


def parse_camera_spec(cam_spec, condition=None):
    """任务行 camera 段 → 归一化相机参数（纯字典，build_manifest 的唯一相机输入）。

    字段名与 config/ 的四维样例一致（不要自造键名）：
      name / location[x,y,z] / rotation[pitch,yaw,roll] / focal_length /
      film_back_width / film_back_height / aperture / shutter_deg / speed_mps
    约定（config 的契约）：
      * rotation 按 UE FRotator 的字段顺序 [pitch, yaw, roll]，不是 [x,y,z]；
        也接受 {"pitch","yaw","roll"} 或显式 pitch_deg/yaw_deg/roll_deg。
      * hfov 一律由 capture_math.hfov_from_focal(focal_length, film_back_width) 反算
        （任务行若显式给了 hfov_deg 才用它），保证与 SceneCapture2D 的 FOV 同源，
        2D 框才能和 RGB 像素对齐。
      * condition == "night" 时按方案表 6 把光圈覆盖为 f/1.8（其余工况取 camera.aperture）。
    默认值取方案表 6：胶片背板 36×24mm、焦距 12mm（广角档）、快门角 180°。
    """
    c = cam_spec if isinstance(cam_spec, dict) else {}
    loc = _as_vec3(c.get("location"), (0.0, 0.0, 300.0))

    rot_in = c.get("rotation")
    if isinstance(rot_in, dict):
        pitch = _as_float(rot_in.get("pitch", rot_in.get("x")), 0.0)
        yaw = _as_float(rot_in.get("yaw", rot_in.get("y")), 0.0)
        roll = _as_float(rot_in.get("roll", rot_in.get("z")), 0.0)
    else:
        # config/ 的契约：rotation 是 UE FRotator 顺序 [pitch, yaw, roll]。
        # （queue_gen 原样透传 cameras.json；若某天配置改成 [yaw, pitch, roll]，这里必须换位，
        #   否则整批 2D 框会整体错位 —— 阶段三用二十帧人工抽检确认。）
        pitch, yaw, roll = _as_vec3(rot_in, (0.0, 0.0, 0.0))
    pitch = _as_float(c.get("pitch_deg"), pitch)
    yaw = _as_float(c.get("yaw_deg"), yaw)
    roll = _as_float(c.get("roll_deg"), roll)

    focal = _as_float(c.get("focal_length"), 12.0)
    if focal <= 0:
        focal = 12.0
    film_w = _as_float(c.get("film_back_width"), 36.0)
    film_h = _as_float(c.get("film_back_height"), 24.0)
    if film_w <= 0:
        film_w = 36.0
    if film_h <= 0:
        film_h = 24.0

    hfov = _as_float(c.get("hfov_deg"), 0.0)
    if not (0.0 < hfov < 180.0):
        hfov = hfov_from_focal(focal, film_w)      # config 没有 hfov，一律由焦距+背板反算
    hfov = min(179.0, max(1.0, hfov))

    # 夜间工况按方案表 6 覆盖光圈：f/1.8；其余工况取 camera.aperture（默认 f/2.0）
    aperture = _as_float(c.get("aperture"), 2.0)
    if str(condition or "").lower() == "night":
        aperture = NIGHT_APERTURE

    width = int(_as_float(c.get("width"), DEFAULT_WIDTH))
    height = int(_as_float(c.get("height"), DEFAULT_HEIGHT))
    if width <= 0:
        width = DEFAULT_WIDTH
    if height <= 0:
        height = DEFAULT_HEIGHT

    return {
        "location": [loc[0], loc[1], loc[2]],
        "rotation": [pitch, yaw, roll],
        "focal_length": focal,
        "film_back_width": film_w,
        "film_back_height": film_h,
        "aperture": aperture,
        "shutter_deg": _as_float(c.get("shutter_deg"), 180.0),
        "speed_mps": _as_float(c.get("speed_mps"), 0.0),   # m/s，进 manifest.speed（post 用它算运动模糊核长）
        "hfov_deg": hfov,
        "width": width,
        "height": height,
    }


def shutter_deg_to_speed_seconds(shutter_deg, fps=24.0):
    """快门角度 → CineCameraComponent.current_shutter_speed（秒）：角度/360 × 1/fps。

    TODO(阶段三): 项目实际帧率（24 / 30）未定；这里按 24 折算，需按工程设置核对。
    """
    fps = fps if fps > 0 else 24.0
    return (_as_float(shutter_deg, 180.0) / 360.0) / float(fps)


def lighting_with_condition(light_spec):
    """lighting 段 = 任务行原样 + condition（下游按 condition 选后处理参数）。

    config 的字段名：condition / label / sun_elevation_deg / sun_azimuth_deg / sun_intensity /
    skylight_intensity / weather / street_lights —— 原样进真值 JSON，一个键都不改名。
    任务行缺 condition 时按太阳高度角粗分：<-5° 夜、<10° 黄昏、其余白天。
    这是标签口径（供 postprocess 映射曝光/噪声），不参与成像计算。
    """
    light = dict(light_spec) if isinstance(light_spec, dict) else {}
    if not light.get("condition"):
        elevation = light.get("sun_elevation_deg", light.get("sun_altitude_deg"))
        if elevation is None:
            light["condition"] = "day"
        else:
            elevation = _as_float(elevation, 30.0)
            if elevation < -5.0:
                light["condition"] = "night"
            elif elevation < 10.0:
                light["condition"] = "dusk"
            else:
                light["condition"] = "day"
    return light


def plan_targets(targets_spec, rng):
    """任务行 targets 段 → 逐目标生成计划（纯函数，不碰引擎）。

    config 的 targets 形态（Lead 定稿）：**组合的列表**，每个组合是 item 列表；
    item = class / actor_class / location[x,y,z] / rotation[pitch,yaw,roll] /
           jitter[dx,dy,dz,dyaw]（每帧确定性抖动的幅度上限）。
    若任务行已经把组合展开成扁平 item 列表，也照常接受（queue_gen 透传的两种形态都兼容）。

    抖动与随机化（方案 3.5 第一条：随机种子固定且可复现）：
      1. 嵌套形态下先用任务 rng 取一个组合号（同任务重跑取到同一个组合）；
      2. 实例号起点从任务 rng 抽一次，帧内自增保证唯一；
      3. 每个目标：位置在 base ± jitter 的均匀分布内抖（Z 默认不抖，避免悬空/入地），
         朝向在 base yaw ± jitter[3] 内抖，材质档位从 rng 抽 0~MATERIAL_VARIANT_COUNT-1。
    返回列表元素：
      {class, instance_id, location[x,y,z], yaw(度), size[sx,sy,sz](完整尺寸,米),
       actor_class(可缺省), material_variant(0..3), combo_index}
    """
    items, combo_index = select_target_items(targets_spec, rng)

    # 实例号起点：从 rng 抽，保证「同任务重跑一致」且不同任务大概率不同
    instance_base = rng.randrange(0, 9000)
    plans = []
    idx = 0
    for item in items:
        if isinstance(item, str):
            item = {"class": item}
        if not isinstance(item, dict):
            continue
        cls = str(item.get("class") or item.get("category") or "vehicle")
        try:
            count = int(item.get("count", 1) or 1)
        except (TypeError, ValueError):
            count = 1
        count = max(1, min(64, count))      # 单类上限 64：同屏目标过多会撞显存（方案 5.4）
        base_loc = _as_vec3(item.get("location"), (0.0, 0.0, 0.0))
        base_yaw = _as_vec3(item.get("rotation"), (0.0, 0.0, 0.0))[1]   # rotation = [pitch,yaw,roll]
        # TODO(阶段三): 目标 rotation 的 pitch/roll 目前只用于摆放，投影只按 yaw 近似（capture_math
        #   的投影是绕 Z 旋转的轴对齐框）；交通标志的俯仰若明显，需要用完整旋转矩阵另算。
        jitter = _as_jitter(item.get("jitter"), item.get("jitter_m"))
        size = item.get("size")
        if size is None:
            size = CLASS_DEFAULT_SIZE.get(cls, (1.0, 1.0, 1.0))
        size = _as_vec3(size, (1.0, 1.0, 1.0))
        for _ in range(count):
            loc = (base_loc[0] + rng.uniform(-jitter[0], jitter[0]),
                   base_loc[1] + rng.uniform(-jitter[1], jitter[1]),
                   base_loc[2] + rng.uniform(-jitter[2], jitter[2]))
            yaw = base_yaw + rng.uniform(-jitter[3], jitter[3])
            variant = rng.randrange(MATERIAL_VARIANT_COUNT)
            plans.append({
                "class": cls,
                "instance_id": int(instance_base + idx),
                "location": [loc[0], loc[1], loc[2]],
                "yaw": yaw,
                "size": [size[0], size[1], size[2]],
                "actor_class": item.get("actor_class"),
                "material_variant": int(variant),
                "combo_index": combo_index,
            })
            idx += 1
    return plans


def _is_item_seq(seq):
    """列表里全是 dict/str → 已经是扁平的 item 列表（不是「组合的列表」）。"""
    return all(isinstance(x, (dict, str)) for x in seq)


def select_target_items(targets_spec, rng):
    """targets 段归一化成 (item 列表, 组合号)。

    兼容三种形态：None / 单个 dict / 扁平 item 列表 / 嵌套组合列表。
    嵌套时用任务 rng 确定性取一个组合（randrange 只在该形态下消耗一次随机数，
    因此扁平形态的随机流不受影响，历史帧仍可复现）。
    """
    if targets_spec is None:
        return [], -1
    if isinstance(targets_spec, dict):
        return [targets_spec], -1
    if isinstance(targets_spec, str):
        return [{"class": targets_spec}], -1
    try:
        seq = list(targets_spec)
    except TypeError:
        return [], -1
    if not seq:
        return [], -1
    if _is_item_seq(seq):
        return seq, -1
    combos = [list(c) if isinstance(c, (list, tuple)) else [c] for c in seq]
    idx = rng.randrange(len(combos))
    return combos[idx], idx


def build_manifest(row, cam_state, objects, width=None, height=None, timestamp=0.0,
                   bit_depth=None):
    """任务行 + 相机 + 目标世界框 → 逐帧真值 JSON（方案附录 B，post/pack 直接消费）。

    参数：
      row       任务行（只用 frame_id / lighting，保证与队列一一对应）
      cam_state parse_camera_spec() 的输出（不要另外塞字典进来）
      objects   [{class, instance_id, center[x,y,z], size[sx,sy,sz](完整尺寸,米), yaw(度)}]
                —— 世界坐标系的轴对齐尺寸 + 绕 Z 的 yaw，和 UE 的 FBox 口径一致
      width/height 缺省取 cam_state 里的（文档 3.4 固定 1920×1080）
      timestamp 仿真内时间戳（秒），由调用方给，保持本函数纯函数
      bit_depth 实际落盘的 RGB 位深（由调用方读 PNG 头得到）；缺省按 16bit 契约

    产出字段（附录 B + 与 post/postprocess.py、post/pack_dataset.py 的契约）：
      frame_id / timestamp / width / height / speed / camera{...} / lighting{...} /
      objects[]{class, instance_id, bbox_3d[7], bbox_2d[4], occlusion, distance, truncated} /
      image_files{rgb, depth, semantic, bit_depth}
      * image_files.rgb = "<frame_id>_rgb.png"，meta 文件 = "<frame_id>.json"（同主文件名）
      * bbox_2d = [x1,y1,x2,y2] 像素坐标（左上原点，已裁进画面内）
      * speed 取 camera.speed_mps（m/s），缺省 0 —— postprocess 用它换算运动模糊核长
    """
    row = row or {}
    cam_state = cam_state or parse_camera_spec(None)
    frame_id = frame_id_of(row)
    w = int(width or cam_state.get("width") or DEFAULT_WIDTH)
    h = int(height or cam_state.get("height") or DEFAULT_HEIGHT)

    cam_loc = tuple(_as_vec3(cam_state.get("location"), (0.0, 0.0, 0.0)))
    rot = cam_state.get("rotation") or [0.0, 0.0, 0.0]
    pitch = _as_float(rot[0], 0.0)
    cam_yaw = _as_float(rot[1], 0.0)
    hfov = _as_float(cam_state.get("hfov_deg"), 60.0)

    projected = []
    dropped = 0
    for ob in (objects or []):
        center = _as_vec3(ob.get("center"), (0.0, 0.0, 0.0))
        size = _as_vec3(ob.get("size"), (0.0, 0.0, 0.0))
        extent = (size[0] * 0.5, size[1] * 0.5, size[2] * 0.5)     # project_aabb 要半尺寸
        obj_yaw = _as_float(ob.get("yaw"), 0.0)
        res = project_aabb(center, extent, obj_yaw, cam_loc, cam_yaw, pitch, hfov, w, h)
        if res is None:          # 整个盒子在相机后方 / 完全出画：不是错误，只是这一帧看不见
            dropped += 1
            continue
        x1, y1, x2, y2, depth, truncated = res
        # 先落成 2 位小数（真值文件可读、跨平台比对稳定），再裁一次防止四舍五入越界
        box = clip_bbox((round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)), w, h)
        if box is None:
            dropped += 1
            continue
        projected.append({
            "class": str(ob.get("class") or "unknown"),
            "instance_id": int(_as_float(ob.get("instance_id"), 0)),
            "bbox_3d": [round(center[0], 3), round(center[1], 3), round(center[2], 3),
                        round(size[0], 3), round(size[1], 3), round(size[2], 3),
                        round(obj_yaw, 3)],
            "bbox_2d": [box[0], box[1], box[2], box[3]],
            "distance": round(float(depth), 3),
            "truncated": bool(truncated),
            "_sort": (depth, int(_as_float(ob.get("instance_id"), 0))),
        })

    # 遮挡比例：被「更近」的框覆盖的面积占自身面积之比（capture_math 的近似，上界估计）。
    # 真实遮挡必须在编辑器内用深度/模板逐帧复核（阶段三验收项），这里只给训练侧做分层统计用。
    all_boxes = [(p["bbox_2d"], p["distance"]) for p in projected]
    for p in projected:
        p["occlusion"] = round(
            boxes_overlap_ratio(p["bbox_2d"], p["distance"], all_boxes), 4)

    # 近 → 远排序（同距离按实例号），保证同一输入的 JSON 字节稳定；也方便人工抽检遮挡关系
    projected.sort(key=lambda p: p["_sort"])

    objects_out = []
    for p in projected:
        objects_out.append({
            "class": p["class"],
            "instance_id": p["instance_id"],
            "bbox_3d": p["bbox_3d"],
            "bbox_2d": p["bbox_2d"],
            "occlusion": p["occlusion"],
            "distance": p["distance"],
            "truncated": p["truncated"],
        })

    return {
        "frame_id": frame_id,
        "timestamp": _as_float(timestamp, 0.0),
        "width": w,
        "height": h,
        "speed": _as_float(cam_state.get("speed_mps"), 0.0),
        "camera": {
            "location": [cam_loc[0], cam_loc[1], cam_loc[2]],
            "rotation": [pitch, cam_yaw, _as_float(rot[2] if len(rot) > 2 else 0.0, 0.0)],
            "focal_length": _as_float(cam_state.get("focal_length"), 12.0),
            "film_back_width": _as_float(cam_state.get("film_back_width"), 36.0),
            "film_back_height": _as_float(cam_state.get("film_back_height"), 24.0),
            "aperture": _as_float(cam_state.get("aperture"), 2.0),
            "shutter_deg": _as_float(cam_state.get("shutter_deg"), 180.0),
            "hfov_deg": hfov,
        },
        "lighting": lighting_with_condition(row.get("lighting")),
        "objects": objects_out,
        "image_files": {
            "rgb": "%s_rgb.png" % frame_id,
            "depth": "%s_depth.png" % frame_id,
            "semantic": "%s_sem.png" % frame_id,
            "bit_depth": int(bit_depth or BIT_DEPTH),
        },
    }


def object_specs_from_plans(plans, fallback_bounds=True):
    """生成计划 → 对象世界框（引擎侧 actor 不可用时的兜底，用计划里的位置/尺寸/朝向）。

    正常路径应走 actor_to_object_spec()（用 actor 的真实包围盒，含碰撞体缩放/贴地修正）；
    这里只在跳过生成、或 actor 取不到包围盒时兜底，保证 manifest 仍能落盘。
    """
    out = []
    for p in (plans or []):
        loc = _as_vec3(p.get("location"), (0.0, 0.0, 0.0))
        size = _as_vec3(p.get("size"), (1.0, 1.0, 1.0))
        out.append({
            "class": p.get("class") or "unknown",
            "instance_id": int(_as_float(p.get("instance_id"), 0)),
            "center": [loc[0], loc[1], loc[2] + (size[2] * 0.5 if fallback_bounds else 0.0)],
            "size": [size[0], size[1], size[2]],
            "yaw": _as_float(p.get("yaw"), 0.0),
        })
    return out


def depth_encode_reference(depths=None):
    """深度编码参照表：米 → 0~65535（capture_math.normalize_depth）。

    只用于「配置/核对」深度材质把 0~200m 线性映射到 RG16 的参数，不参与像素计算。
    ⚠️ 真正的线性化必须在材质里做：SceneCapture 的 SCS_SCENE_DEPTH 是非线性 NDC 深度，
    不能直接当米用（阶段三要验证材质输出与这张参照表一致）。
    """
    if depths is None:
        depths = (0.0, 50.0, 100.0, 150.0, DEPTH_FAR)
    return [(float(d), normalize_depth(float(d), DEPTH_FAR)) for d in depths]


def frame_paths(frame_id):
    """一帧的四条落盘路径：三通道 PNG + 同名 JSON（契约见 manifest 的 image_files）。"""
    return (os.path.join(RAW, "%s_rgb.png" % frame_id),
            os.path.join(RAW, "%s_depth.png" % frame_id),
            os.path.join(RAW, "%s_sem.png" % frame_id),
            os.path.join(META, "%s.json" % frame_id))


def png_header_bit_depth(path):
    """读 PNG IHDR 的位深（纯标准库，引擎内没有 Pillow）。读不出返回 None。

    image_files.bit_depth 必须写「实际落盘位深」：若 16bit 路径退回 8bit 兜底，真值里要如实记 8，
    否则下游会按 16bit 解释线性值，整批图像分布错一层（方案 3.4 的 16bit 承诺要能被审计）。
    """
    try:
        with open(path, "rb") as f:
            head = f.read(26)
    except Exception:
        return None
    if len(head) < 26 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return int(head[24])


def load_tasks(path):
    """每行一个 JSON 任务；任务清单是断点续跑的唯一事实来源（方案 4.4）。

    坏行直接报错并带行号：清单坏了应该人工介入，不能静默跳过（否则会悄悄少跑一批）。
    """
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except ValueError as e:
                raise ValueError("任务清单 %s 第 %d 行不是合法 JSON: %s" % (path, lineno, e))
    return tasks


def is_done(task_id):
    return os.path.exists(os.path.join(DONE, "%s.done" % task_id))


def mark_done(task_id, note="ok"):
    os.makedirs(DONE, exist_ok=True)
    with open(os.path.join(DONE, "%s.done" % task_id), "w", encoding="utf-8") as f:
        f.write("%s\n" % note)


def skip_existing(frame_id):
    """输出已存在则跳过（方案 3.5 第二条）——帧名由任务行确定性生成，同名即同帧。

    严格四条都齐才算「已存在」：少任何一条（例如 kill 在写 meta 之前）都要重跑，
    因为半成品帧既不能训练也不能当跳过依据。
    """
    paths = frame_paths(frame_id)
    return os.path.exists(paths[3]) and all(
        os.path.exists(p) or os.path.exists(os.path.splitext(p)[0] + ".exr")
        for p in paths[:3])


def write_manifest(manifest, path):
    """逐帧 JSON 落盘：先写 .tmp 再原子替换，避免 kill 留下半个 JSON 被下游当完整读。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    return path


# ══════════════════════════════════════════════════════════════════════════════════
# 二、引擎侧调度（下面每个 unreal.* 调用都按「不确定就标 TODO」处理，未经编辑器验证）
# ══════════════════════════════════════════════════════════════════════════════════

def _try_set_editor_property(obj, name, value, label=""):
    """保守地设编辑器属性：属性名不对只告警不中断。

    无人值守优先（一批几万帧，不能因为一个属性名拼错整批挂掉），但每次失败都打日志，
    阶段三必须对着真实工程把下面所有属性名逐条核对完，否则会出现「跑完了但参数没生效」的假数据。
    """
    if obj is None:
        return False
    try:
        obj.set_editor_property(name, value)
        return True
    except Exception as e:
        _log("⚠️ 设置属性失败(%s.%s): %s" % (label or type(obj).__name__, name, e))
        return False


def _actor_subsystem():
    """UE 5.3 的 Actor 操作走 EditorActorSubsystem；旧工程退回 EditorLevelLibrary。"""
    getter = getattr(unreal, "get_editor_subsystem", None)
    cls = getattr(unreal, "EditorActorSubsystem", None)
    if getter is not None and cls is not None:
        try:
            sub = getter(cls)
            if sub is not None:
                return sub
        except Exception as e:
            _log("取 EditorActorSubsystem 失败: %s（退回 EditorLevelLibrary）" % e)
    return None


def _find_actors(cls):
    """按类取当前关卡里的 actor；两条接口都试，取不到返回空列表。"""
    sub = _actor_subsystem()
    if sub is not None and hasattr(sub, "get_all_level_actors"):
        try:
            return [a for a in sub.get_all_level_actors() if isinstance(a, cls)]
        except Exception as e:
            _log("EditorActorSubsystem.get_all_level_actors 失败: %s" % e)
    lib = getattr(unreal, "EditorLevelLibrary", None)
    if lib is not None and hasattr(lib, "get_all_actors_of_class"):
        try:
            return list(lib.get_all_actors_of_class(cls))
        except Exception as e:
            _log("EditorLevelLibrary.get_all_actors_of_class 失败: %s" % e)
    return []


def _editor_world():
    """编辑器世界（创建 RT、导出 RT 都要 world context）。"""
    # TODO(阶段三): UE 5.3 推荐 unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()；
    #   旧接口 EditorLevelLibrary.get_editor_world() 已标记 deprecated，两条都试，仍拿不到就让上层报错。
    sub_cls = getattr(unreal, "UnrealEditorSubsystem", None)
    getter = getattr(unreal, "get_editor_subsystem", None)
    if sub_cls is not None and getter is not None:
        try:
            world = getter(sub_cls).get_editor_world()
            if world is not None:
                return world
        except Exception as e:
            _log("取编辑器世界失败(UnrealEditorSubsystem): %s" % e)
    lib = getattr(unreal, "EditorLevelLibrary", None)
    if lib is not None and hasattr(lib, "get_editor_world"):
        try:
            return lib.get_editor_world()
        except Exception as e:
            _log("取编辑器世界失败(EditorLevelLibrary): %s" % e)
    return None


def load_level(map_path):
    """加载关卡；失败抛给上层记错误日志（关卡路径来自任务行，脚本内不硬编码资产路径）。

    TODO(阶段三): 核对关卡资产路径写法（/Game/Maps/xxx 还是 xxx.xxx）与 5.3 的接口选择：
      EditorLoadingAndSavingUtils.load_map 是老接口，5.3 更稳的是
      unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(path)。
      load_map 返回 bool，失败不抛异常 —— 这里显式转成异常，避免「加载失败还接着用旧关卡」。
    """
    lib = getattr(unreal, "LevelEditorSubsystem", None)
    getter = getattr(unreal, "get_editor_subsystem", None)
    if lib is not None and getter is not None:
        try:
            sub = getter(lib)
            if sub is not None and hasattr(sub, "load_level"):
                ok = sub.load_level(map_path)
                if ok is False:
                    raise RuntimeError("LevelEditorSubsystem.load_level 返回 False: %s" % map_path)
                return True
        except Exception as e:
            _log("LevelEditorSubsystem.load_level 失败(%s): %s" % (map_path, e))
    els = getattr(unreal, "EditorLoadingAndSavingUtils", None)
    if els is not None and hasattr(els, "load_map"):
        ok = els.load_map(map_path)
        if ok is False:
            raise RuntimeError("EditorLoadingAndSavingUtils.load_map 返回 False: %s" % map_path)
        return True
    raise RuntimeError("没有可用的关卡加载接口（阶段三核对 LevelEditorSubsystem）")


def _resolve_class(actor_class):
    """actor_class 可以是字符串资产路径，也可以已经是 unreal.Class。"""
    if actor_class is None:
        return None
    if not isinstance(actor_class, str):
        return actor_class
    # TODO(阶段三): 核对 unreal.load_class(None, path) 在 5.3 对 /Game/ 蓝图类与 /Script/ 原生类的
    #   可用性；蓝图类通常需要 unreal.load_class(None, "/Game/.../BP_Car.BP_Car_C")。
    return unreal.load_class(None, actor_class)


def _spawn_actor(actor_class, location, rotation):
    """生成一个 actor；返回 None 表示没生成（调用方按计划兜底投影，不中断整帧）。"""
    cls = _resolve_class(actor_class)
    if cls is None:
        return None
    sub = _actor_subsystem()
    if sub is not None and hasattr(sub, "spawn_actor_from_class"):
        return sub.spawn_actor_from_class(cls, location, rotation)
    lib = getattr(unreal, "EditorLevelLibrary", None)
    if lib is not None and hasattr(lib, "spawn_actor_from_class"):
        return lib.spawn_actor_from_class(cls, location, rotation)
    raise RuntimeError("没有可用的 spawn_actor_from_class 接口（阶段三核对 EditorActorSubsystem）")


def clear_spawned(pairs):
    """清场：销毁上一帧生成的 actor（方案 3.5 的「清场并生成目标」）。

    TODO(阶段三): 若目标实际放在子关卡里，可改成每次 reload 子关卡；当前用逐个 destroy 更省时，
      但要确认被销毁的 actor 不会残留在 CustomDepth/场景里（尤其带附着 child actor 的蓝图）。
    """
    sub = _actor_subsystem()
    for actor, _plan in (pairs or []):
        if actor is None:
            continue
        try:
            if sub is not None and hasattr(sub, "destroy_actor"):
                sub.destroy_actor(actor)
            else:
                lib = getattr(unreal, "EditorLevelLibrary", None)
                if lib is not None and hasattr(lib, "destroy_actor"):
                    lib.destroy_actor(actor)
        except Exception as e:
            _log("销毁 actor 失败: %s" % e)


def spawn_targets(plans):
    """按生成计划生成目标，返回 [(actor, plan)]。

    TODO(阶段三): 材质随机化还没接真实资产库 —— material_variant 只是抽到的档位号，
      需要 config/ 给出「类别 → 材质路径列表」，再用 set_material(0, ...) 逐个贴。
      蓝图类需要先取到静态网格体组件，再对其 set_material；不同工程资产结构不同，必须对着真实工程写。
    """
    pairs = []
    for p in (plans or []):
        loc = p.get("location") or [0.0, 0.0, 0.0]
        # TODO(阶段三): 核对 unreal.Vector/Rotator 的构造参数顺序（Rotator 是 Pitch/Yaw/Roll，
        #   不是 x/y/z），以及目标是否需要「落地修正」（把 Z 抬到包围盒一半高度）。
        actor = _spawn_actor(p.get("actor_class"),
                             unreal.Vector(x=loc[0], y=loc[1], z=loc[2]),
                             unreal.Rotator(pitch=0.0, yaw=p.get("yaw", 0.0), roll=0.0))
        if actor is None:
            _log("⚠️ 未生成 class=%s（缺 actor_class 或类解析失败），该目标只在 JSON 里按计划兜底"
                 % p.get("class"))
            continue
        pairs.append((actor, p))
    return pairs


def actor_to_object_spec(actor, plan):
    """actor → 世界框（center/size/yaw），进 build_manifest 的 objects。

    用 actor.get_actor_bounds(only_if_collision=False) 拿真实包围盒（含缩放），
    取不到时退回计划里的位置与尺寸，保证这一帧仍有真值可写（宁可有近似真值，也不丢帧）。
    TODO(阶段三): 核对 get_actor_bounds 对「无碰撞体的蓝图目标」是否返回有效值；
      必要时改成遍历 PrimitiveComponent 的 bounds 求并集。
    """
    cls = plan.get("class") if plan else "unknown"
    iid = int(_as_float(plan.get("instance_id"), 0)) if plan else 0
    try:
        origin, extent = actor.get_actor_bounds(False)
        rot = actor.get_actor_rotation()
        return {
            "class": cls,
            "instance_id": iid,
            "center": [float(origin.x), float(origin.y), float(origin.z)],
            "size": [float(extent.x) * 2.0, float(extent.y) * 2.0, float(extent.z) * 2.0],
            "yaw": float(rot.yaw),
        }
    except Exception as e:
        _log("⚠️ 取包围盒失败(actor)，退回计划值: %s" % e)
        fallback = object_specs_from_plans([plan or {}])
        return fallback[0] if fallback else {"class": cls, "instance_id": iid,
                                             "center": [0.0, 0.0, 0.0],
                                             "size": [1.0, 1.0, 1.0], "yaw": 0.0}


def apply_semantic_stencil(pairs):
    """给目标打 CustomDepth 模板值（语义通道的来源，方案 3.4 模板表）。

    TODO(阶段三): 三个待核对点 ——
      1) 项目设置必须 r.CustomDepth=3 打开模板，否则 CustomDepth 不渲染；
      2) SceneCapture2D 的后期材质要能采样 SceneTexture:CustomStencil，且捕获源不能把 stencil 洗掉；
      3) 天空固定 255 需要天空 actor / 天空材质单独给值，本函数只处理目标 actor。
    """
    prim_cls = getattr(unreal, "PrimitiveComponent", None)
    for actor, plan in (pairs or []):
        if actor is None:
            continue
        stencil = SEMANTIC_STENCIL.get((plan or {}).get("class"))
        if not stencil:
            continue      # 没有模板值的类别（例如未知类）不写，避免语义图出现错标签
        comps = []
        try:
            if prim_cls is not None and hasattr(actor, "get_components_by_class"):
                comps = list(actor.get_components_by_class(prim_cls))
            elif hasattr(actor, "get_components"):
                comps = [c for c in actor.get_components() if prim_cls is None
                         or isinstance(c, prim_cls)]
        except Exception as e:
            _log("⚠️ 取组件失败(语义模板): %s" % e)
        for comp in comps:
            _try_set_editor_property(comp, "render_custom_depth", True, "语义")
            _try_set_editor_property(comp, "custom_depth_stencil_value", int(stencil), "语义")


def place_camera(cam_spec, condition=None):
    """摆相机：返回 (cine_camera_actor, cam_state)；cam_state 是 parse_camera_spec 的纯字典。

    condition 传 lighting.condition（"night" 时光圈按方案表 6 覆盖为 f/1.8）。
    """
    cam_state = parse_camera_spec(cam_spec, condition=condition)
    loc = cam_state["location"]
    pitch, yaw, roll = cam_state["rotation"]
    cine_cls = getattr(unreal, "CineCameraActor", None)
    cam = None
    actors = _find_actors(cine_cls) if cine_cls is not None else []
    if actors:
        cam = actors[0]
    else:
        cam = _spawn_actor(cine_cls,
                           unreal.Vector(x=loc[0], y=loc[1], z=loc[2]),
                           unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll))
    if cam is None:
        raise RuntimeError("无法获取或生成 CineCameraActor（阶段三核对 CineCameraActor 是否可 Python 生成）")
    cam.set_actor_location(unreal.Vector(x=loc[0], y=loc[1], z=loc[2]), False, False)
    # TODO(阶段三): set_actor_rotation 的第二参数在 5.3 是 teleport（是否刷新物理/附着），核对签名。
    cam.set_actor_rotation(unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll), False)

    comp = None
    try:
        comp = cam.get_cine_camera_component()
    except Exception as e:
        _log("⚠️ 取 CineCameraComponent 失败: %s" % e)
    if comp is not None:
        _try_set_editor_property(comp, "current_focal_length", cam_state["focal_length"], "cam")
        _try_set_editor_property(comp, "current_aperture", cam_state["aperture"], "cam")
        # UE 5.3 CineCameraComponent 没有 current_shutter_speed；运动模糊由
        # 后处理按任务行 shutter_deg 计算，避免每帧刷无效属性告警。
        _try_set_editor_property(comp, "constrain_aspect_ratio", True, "cam")
        # 胶片背板：CameraFilmbackSettings 是结构体，整体 set 进去
        # TODO(阶段三): 核对 CameraFilmbackSettings 的字段名（sensor_width/sensor_height）
        #   与单位（毫米）；若字段名不同，这里会走告警分支，fov 仍由 hfov_deg 单独设到 SceneCapture 上。
        try:
            filmback = unreal.CameraFilmbackSettings()
            _try_set_editor_property(filmback, "sensor_width", cam_state["film_back_width"], "filmback")
            _try_set_editor_property(filmback, "sensor_height", cam_state["film_back_height"], "filmback")
            comp.set_editor_property("filmback", filmback)
        except Exception as e:
            _log("⚠️ 设置胶片背板失败: %s" % e)
    return cam, cam_state


def set_lighting(light_spec):
    """调光照：按 config 的 lighting 字段摆太阳并设强度。

    字段名（config 的契约，不改名）：condition / label / sun_elevation_deg / sun_azimuth_deg /
    sun_intensity / skylight_intensity / weather / street_lights。
      * DirectionalLight：pitch = -sun_elevation_deg（UE 里朝下打是负 pitch）、yaw = sun_azimuth_deg、
        intensity = sun_intensity；
      * SkyLight：intensity = skylight_intensity（环境光）。
    weather=rain 与 street_lights 目前只记录（已原样进真值 JSON），不驱动渲染。
    TODO(阶段三): (a) 核对 DirectionalLight/SkyLight 的组件属性名与 Python 取法；
      (b) 雨天需要接粒子/湿滑路面材质，路灯需要接场景里的路灯蓝图开关 —— 都要真实工程资产；
      (c) 核对「平行光 pitch 取负高度角」的约定（UE 的 DirectionalLight 朝向语义需实测一帧验证）。
    """
    light = lighting_with_condition(light_spec)

    elevation = _as_float(light.get("sun_elevation_deg"), 45.0)
    azimuth = _as_float(light.get("sun_azimuth_deg"), 0.0)
    sun_intensity = _as_float(light.get("sun_intensity"), 8.0)

    dir_cls = getattr(unreal, "DirectionalLight", None)
    for actor in (_find_actors(dir_cls) if dir_cls is not None else []):
        actor.set_actor_rotation(
            unreal.Rotator(pitch=-elevation, yaw=azimuth, roll=0.0), False)
        try:
            comp = actor.get_editor_property("directional_light_component")
        except Exception as e:
            _log("⚠️ 取 DirectionalLightComponent 失败: %s" % e)
            comp = None
        if comp is not None:
            _try_set_editor_property(comp, "intensity", sun_intensity, "sun")

    skylight_intensity = light.get("skylight_intensity")
    sky_cls = getattr(unreal, "SkyLight", None)
    if skylight_intensity is not None and sky_cls is not None:
        for actor in _find_actors(sky_cls):
            try:
                comp = actor.get_editor_property("light_component")
            except Exception as e:
                _log("⚠️ 取 SkyLightComponent 失败: %s" % e)
                comp = None
            if comp is not None:
                _try_set_editor_property(comp, "intensity",
                                         _as_float(skylight_intensity, 1.0), "sky")

    # TODO(阶段三): weather == "rain": 开降水粒子、湿滑路面材质（粗糙度 0.02~0.05）、压暗太阳；
    #   street_lights: 开场景路灯蓝图（夜间工况），并确认路灯本身不写进语义模板（否则会污染语义图）。
    return light


def _scene_capture_component(cap_actor):
    """取 SceneCapture2D 的捕获组件。"""
    # TODO(阶段三): 核对 ASceneCapture2D 组件属性的 Python 名（capture_component2d / capture_component）。
    for name in ("capture_component2d", "capture_component"):
        try:
            comp = cap_actor.get_editor_property(name)
            if comp is not None:
                return comp
        except Exception:
            continue
    try:
        return cap_actor.capture_component2d
    except Exception as e:
        raise RuntimeError("取 SceneCaptureComponent2D 失败: %s" % e)


def _ensure_scene_capture(cam_actor, cam_state):
    """复用/生成 SceneCapture2D，并把它摆到与 CineCamera 同一位姿、同一 FOV 上。

    为什么自己摆位姿而不是 attach_to_actor：attach 的 socket/相对变换在 Python 里易错，
    阶段三再换成 attach（相机若带跟随/动画，逐帧同步会有一帧迟滞）。
    """
    cap_cls = getattr(unreal, "SceneCapture2D", None)
    if cap_cls is None:
        raise RuntimeError("unreal.SceneCapture2D 不存在（阶段三核对）")
    actors = _find_actors(cap_cls)
    if actors:
        cap = actors[0]
    else:
        loc = cam_state["location"]
        cap = _spawn_actor(cap_cls, unreal.Vector(x=loc[0], y=loc[1], z=loc[2]),
                           unreal.Rotator(pitch=0.0, yaw=cam_state["rotation"][1], roll=0.0))
    if cap is None:
        raise RuntimeError("无法生成 SceneCapture2D")
    if cam_actor is not None:
        try:
            loc = cam_actor.get_actor_location()
            rot = cam_actor.get_actor_rotation()
            cap.set_actor_location(loc, False, False)
            cap.set_actor_rotation(rot, False)
        except Exception as e:
            _log("⚠️ SceneCapture2D 同步相机位姿失败: %s" % e)
    return cap


def _render_lib():
    """渲染目标库：UE5 的 Python 名通常是 KismetRenderingLibrary（旧版/别名是 RenderingLibrary）。"""
    for name in ("KismetRenderingLibrary", "RenderingLibrary"):
        lib = getattr(unreal, name, None)
        if lib is not None:
            return lib
    return None


_RT_CACHE = {}      # (通道名, 宽, 高, 格式) → TextureRenderTarget2D；批次内复用，避免每帧新建 UObject


def _create_render_target(name, width, height, format_name):
    """建/复用渲染目标。format_name: RTF_RGBA16F（RGB 16bit 线性）、RTF_RG16F（深度）、RTF_R8（语义）。

    为什么缓存：RT 是整块显存（1920×1080 RGBA16F 约 16MB），每帧新建 3 个的话 1 万帧会把显存
    和 UObject 表一起撑爆（方案 5.4 显存预算本来就紧）。同一批次分辨率固定，按通道复用一个即可；
    分辨率变化时按新 key 新建（老 RT 保留，规模以「批次内出现过几种分辨率」为上限，可接受）。

    TODO(阶段三): 两处待核对 ——
      1) KismetRenderingLibrary.create_render_target2d 的签名（world context 传 None 还是编辑器世界）；
      2) 16bit 格式枚举名（RTF_RGBA16F / RTF_R16F / RTF_RG16F）随引擎版本可能不同。
    """
    key = (str(name), int(width), int(height), str(format_name))
    cached = _RT_CACHE.get(key)
    if cached is not None:
        # ⚠️ 缓存的 RT 可能已被引擎回收：分辨率切换 / 关卡重载时 UObject 会失效，
        #   此时再拿去 export 会报 "TextureRenderTarget2D: Internal Error - ObjectInstance is null!"。
        #   这里先探活，失效就丢掉重建（实测连续两帧换分辨率时必现）。
        try:
            _ = cached.get_name()
            return cached
        except Exception:
            _RT_CACHE.pop(key, None)
            _log("⚠️ 渲染目标 %s 已失效，重建" % (name,))
    lib = _render_lib()
    creator = getattr(lib, "create_render_target2d", None) if lib is not None else None
    if creator is None:
        raise RuntimeError("没有 create_render_target2d（阶段三核对 KismetRenderingLibrary）")
    fmt_enum = getattr(unreal, "TextureRenderTargetFormat", None)
    fmt = getattr(fmt_enum, format_name, None) if fmt_enum is not None else None
    if fmt is None:
        _log("⚠️ 渲染目标格式 %s 不存在，用默认格式" % format_name)
    world = _editor_world()
    try:
        rt = creator(world, int(width), int(height), fmt)
    except TypeError:
        # 某些版本的签名把 format 放在末尾且非关键字，退回三参数调用
        rt = creator(int(width), int(height))
    except Exception as e:
        raise RuntimeError("创建渲染目标失败(%s): %s" % (name, e))
    _RT_CACHE[key] = rt
    return rt


def _save_render_target(rt, path):
    """RT → PNG 落盘。首选 ImageWriteBlueprintLibrary.export_to_disk（可保留 16bit 源），
    兜底 KismetRenderingLibrary.export_render_target（多为 8bit，见下方 TODO）。

    ⚠️ 16bit 是否真的落到 PNG 里，本机无法验证（见文件头）。两条路径都是「先写、后由上层确认文件存在」，
    写不出来就当整帧失败重跑，不写半成品。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.png"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    writer = getattr(unreal, "ImageWriteBlueprintLibrary", None)
    if writer is not None and hasattr(writer, "export_to_disk"):
        try:
            options = unreal.ImageWriteOptions()
            # UE 5.3 实测签名：export_to_disk(texture, filename, options) —— 只有 3 个参数，
            # 没有 world context（早期版本/文档里写 4 个是错的，传 4 个会报
            # "export_to_disk() takes at most 3 arguments"）。
            fmt_enum = getattr(unreal, "ImageWriteFormat", None)
            if fmt_enum is not None:
                _try_set_editor_property(options, "format", getattr(fmt_enum, "PNG", None),
                                         "ImageWriteOptions")
            _try_set_editor_property(options, "overwrite_file", True, "ImageWriteOptions")
            _try_set_editor_property(options, "compression_quality", 100, "ImageWriteOptions")
            # 5.3 的字段名是 async_（不是 async）；强制同步写盘，再给 0.5s 让文件出现
            _try_set_editor_property(options, "async_", False, "ImageWriteOptions")
            writer.export_to_disk(rt, tmp, options)
            deadline = time.time() + 0.5
            while not os.path.exists(tmp) and time.time() < deadline:
                time.sleep(0.02)
        except Exception as e:
            _log("⚠️ ImageWriteBlueprintLibrary 落盘失败(%s): %s（退回 export_render_target）"
                 % (path, e))
    if not os.path.exists(tmp):
        # 兜底：RenderingLibrary.export_render_target(world_context, texture, filename)
        # （5.3 里模块名是 RenderingLibrary，不是 KismetRenderingLibrary）
        lib = getattr(unreal, "RenderingLibrary", None) or getattr(
            unreal, "KismetRenderingLibrary", None)
        exporter = getattr(lib, "export_render_target", None) if lib is not None else None
        if exporter is None:
            raise RuntimeError("没有可用的 RT 落盘接口（ImageWriteBlueprintLibrary 与 "
                               "RenderingLibrary 都不可用）")
        # TODO(阶段三): export_render_target 产出多为 8bit、且可能带 sRGB 编码 —— 与「16bit 线性」冲突，
        #   只当兜底用；正式路径必须是 16bit（否则整批高光动态范围永久丢失，方案 3.4）。
        # UE 5.3 实测签名：export_render_target(world_context, render_target, file_path, file_name)
        # 注意：RT 是 16bit 浮点时该函数强制写 EXR，8bit 才写 PNG —— 所以扩展名与传入的 tmp 名不一致，
        #       写完后要把实际产物找出来改名（见下方 glob）。
        exporter(_editor_world(), rt,
                 os.path.dirname(os.path.abspath(tmp)),
                 os.path.basename(tmp))
        # 找出实际落盘的文件（可能是 tmp.png 或 tmp.exr）
        import glob as _glob
        stem = os.path.splitext(tmp)[0]
        produced = [p for p in _glob.glob(stem + ".*") if p != tmp]
        if produced:
            # EXR 的内容不能伪装成 PNG：下游按 PNG 解码和读取文件头。
            png = next((p for p in produced if p.lower().endswith(".png")), None)
            if png:
                os.replace(png, tmp)
            else:
                exr = next((p for p in produced if p.lower().endswith(".exr")), None)
                if exr:
                    actual = os.path.splitext(path)[0] + ".exr"
                    os.replace(exr, actual)
                    return actual
                raise RuntimeError("RT 兜底导出格式未知: %s" % produced[0])
    if not os.path.exists(tmp):
        raise RuntimeError("渲染目标落盘失败（两条路径都没产出文件）: %s" % path)
    os.replace(tmp, path)
    return path


# 捕获源候选链：不同 UE 小版本的 SceneCaptureSource 枚举名不一样（5.3 起 SCS_SCENE_COLOR /
# SCS_FINAL_COLOR 已被移除，改成 _HDR/_LDR 变体）。按顺序找第一个存在的，找不到才算失败。
CAPTURE_SOURCE_CANDIDATES = {
    # RGB：优先 HDR 线性（16bit 承诺依赖它），再退 LDR
    "rgb": ["SCS_SCENE_COLOR_HDR", "SCS_SCENE_COLOR_HDR_NO_ALPHA",
            "SCS_FINAL_COLOR_HDR", "SCS_SCENE_COLOR", "SCS_FINAL_COLOR_LDR"],
    # 深度：场景深度（NDC，线性化在材质里做，见文件头）
    "depth": ["SCS_SCENE_DEPTH", "SCS_DEVICE_DEPTH", "SCS_SCENE_COLOR_SCENE_DEPTH"],
    # 语义：后处理前的最终颜色（模板值来自 CustomDepth，见 capture_math）
    "sem": ["SCS_FINAL_COLOR_LDR", "SCS_BASE_COLOR", "SCS_FINAL_COLOR_HDR", "SCS_FINAL_COLOR"],
}


def _resolve_capture_source(channel):
    """按候选链找到本机引擎真实存在的 SceneCaptureSource 枚举值；返回 (枚举值, 名称)。"""
    src_enum = getattr(unreal, "SceneCaptureSource", None)
    if src_enum is None:
        return None, None
    for name in CAPTURE_SOURCE_CANDIDATES.get(channel, []):
        val = getattr(src_enum, name, None)
        if val is not None:
            return val, name
    return None, None


def _configure_capture(comp, rt, channel, cam_state):
    """把 SceneCaptureComponent2D 配成指定通道：目标 RT + 捕获源 + FOV。"""
    _try_set_editor_property(comp, "texture_target", rt, "capture")
    _try_set_editor_property(comp, "capture_every_frame", False, "capture")   # 手动 capture_scene
    _try_set_editor_property(comp, "fov_angle", cam_state["hfov_deg"], "capture")
    src, name = _resolve_capture_source(channel)
    if src is None:
        _log("⚠️ 通道 %s 找不到可用的 SceneCaptureSource（候选 %s 都不存在）"
             % (channel, CAPTURE_SOURCE_CANDIDATES.get(channel)))
        return None
    _try_set_editor_property(comp, "capture_source", src, "capture")
    return name


def _capture_and_save(comp, rt, channel, cam_state, path):
    """配一次 → capture_scene → 落盘。返回实际使用的捕获源名（便于排查黑帧）。"""
    used = _configure_capture(comp, rt, channel, cam_state)
    if not hasattr(comp, "capture_scene"):
        raise RuntimeError("SceneCaptureComponent2D.capture_scene 不存在（阶段三核对）")
    # 每个通道首次捕获先预热两帧，丢弃着色器/RT 初始化阶段的结果。
    warm_key = (id(comp), channel)
    if warm_key not in _CAPTURE_WARMED:
        comp.capture_scene()
        comp.capture_scene()
        _CAPTURE_WARMED.add(warm_key)
    comp.capture_scene()
    actual = _save_render_target(rt, path)
    return used, actual


_CAPTURE_WARMED = set()


def capture_channels(frame_id, cam_actor, cam_state):
    """四通道捕获里的前三通道（方案 3.4）：RGB 16bit 线性 / 深度 0~200m / 语义模板值。

    第四通道（3D 框）由 build_manifest 的 project_aabb 投影得到，不占渲染时间。
    ⚠️ 未经编辑器验证：16bit 落盘与 CustomDepth 模板值两处见文件头说明。
    """
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(META, exist_ok=True)
    rgb_p, depth_p, sem_p, _meta_p = frame_paths(frame_id)
    w, h = cam_state["width"], cam_state["height"]

    cap = _ensure_scene_capture(cam_actor, cam_state)
    comp = _scene_capture_component(cap)

    # 1) RGB：FloatRGBA + 场景颜色（不做色调映射）
    # TODO(阶段三): SCS_SCENE_COLOR 是否已绕过后处理/色调曲线未验证；若要严格线性，
    #   需要自定义后处理材质直接取 SceneColor 并关掉 tone curve，再配 RTF_RGBA16F。
    # 通道名固定（不带 frame_id）→ 同一批次复用同一个 RT，避免每帧新建撑爆显存（见 _create_render_target）
    rgb_rt = _create_render_target("rgb", w, h, "RTF_RGBA16F")
    used_rgb, rgb_p = _capture_and_save(comp, rgb_rt, "rgb", cam_state, rgb_p)
    _log_once("rgb_src", "RGB 捕获源 = %s" % used_rgb)

    # 2) 深度：捕获源切到场景深度；0~200m 线性化 + 归一化在材质里做，Python 只给配置参照
    # ⚠️ 实测（本机 Vulkan）：RTF_RG16F / RTF_R8 / RTF_R32F 落盘一律 0 字节，
    #   只有 RGBA 系（RTF_RGBA8 / RTF_RGBA16F）能写出文件。所以深度通道改用 RTF_RGBA16F，
    #   深度值走 R 通道（16F 保留精度），G/B 留 0 —— 后处理按 R 通道读即可。
    # TODO(阶段三): SCS_SCENE_DEPTH 给的是非线性 NDC 深度，必须挂「线性化 + /200m」的后处理材质；
    #   本函数目前只切捕获源，材质参数（DepthFarMeters=200、EncodeMax=65535）待接。
    depth_rt = _create_render_target("depth", w, h, "RTF_RGBA16F")
    _log_once("depth_ref", "深度编码参照 米→0~65535: %s" % depth_encode_reference())
    _apply_post_material(comp, DEPTH_MATERIAL_PATH, "depth")
    _, depth_p = _capture_and_save(comp, depth_rt, "depth", cam_state, depth_p)

    # 3) 语义：把每个物体的材质临时替换成「类别灰度 Unlit 材质」，捕获后还原。
    # ⚠️ 不能用后处理材质：实测 SceneCapture.post_process_settings 与 PostProcessVolume
    #   两条路径在无头 -ExecutePythonScript 下都完全不生效（采样值与基线逐位相同）。
    #   材质替换实测有效（road 区域灰度精确等于 stencil/255）。实现见 semantic_capture.py。
    # ⚠️ 同深度通道：RTF_R8 落盘 0 字节，改用 RTF_RGBA8。
    sem_rt = _create_render_target("sem", w, h, "RTF_RGBA8")
    try:
        # 脚本目录要在 sys.path 里（-ExecutePythonScript 不保证当前目录是本脚本所在目录）
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import semantic_capture as _sem
        sem_p = _sem.capture_semantic(comp, sem_rt, _save_render_target, sem_p)
    except Exception as e:
        import traceback as _tb
        _log("⚠️ 语义通道材质替换失败(%s)，退回普通捕获（内容将是颜色图）" % e)
        _log("   详情: " + _tb.format_exc()[:400].replace(chr(10), " | "))
        _, sem_p = _capture_and_save(comp, sem_rt, "sem", cam_state, sem_p)
    return (rgb_p, depth_p, sem_p)


# 后处理材质路径（setup_assets.py 生成在 /Game/Validation/ 下）。
# 找不到时只告警不中断 —— 老环境没有这两个材质也能跑出 RGB。
DEPTH_MATERIAL_PATH = "/Game/Validation/M_LinearDepth"
SEMANTIC_MATERIAL_PATH = "/Game/Validation/M_SemanticMask"


def _apply_post_material(comp, material_path, channel):
    """把后处理材质挂到/摘离 SceneCapture 的 post_process_settings。

    为什么必须挂：深度通道拿到的是非线性 NDC 深度、语义通道拿到的是渲染颜色，
    都靠后处理材质在 GPU 端换算成 0~1 的可用值；不挂的话输出是原始值，
    转 8bit 后通常全白或全红（实测现象）。
    """
    try:
        if material_path is None:
            settings = comp.get_editor_property("post_process_settings")
            try:
                settings.set_editor_property("weighted_blendables", unreal.WeightedBlendables())
                comp.set_editor_property("post_process_settings", settings)
            except Exception:
                pass
            return True

        mat = unreal.load_asset(material_path)
        if mat is None:
            _log_once("mat_" + channel,
                      "⚠️ 后处理材质缺失 %s，%s 通道将输出原始值（转 8bit 后可能过曝）"
                      % (material_path, channel))
            return False

        settings = comp.get_editor_property("post_process_settings")
        # ⚠️ 实测：直接构造 WeightedBlendables 并 set 进去不生效（读回 len=0）；
        #   必须取出 array → append → 回写，最后整体 set 回组件。
        blend = settings.get_editor_property("weighted_blendables")
        arr = blend.get_editor_property("array")
        entry = unreal.WeightedBlendable()
        _try_set_editor_property(entry, "weight", 1.0, "post")
        _try_set_editor_property(entry, "object", mat, "post")
        arr.append(entry)
        blend.set_editor_property("array", arr)
        settings.set_editor_property("weighted_blendables", blend)
        comp.set_editor_property("post_process_settings", settings)
        _log_once("mat_" + channel, "%s 通道已挂后处理材质 %s" % (channel, material_path))
        return True
    except Exception as e:
        _log_once("mat_err_" + channel, "⚠️ 后处理材质挂载失败(%s): %s" % (channel, e))
        return False


_LOG_ONCE = set()


def _log_once(key, msg):
    """同一批只打一次的日志（例如深度参照表），避免几万帧把日志刷满。"""
    if key in _LOG_ONCE:
        return
    _LOG_ONCE.add(key)
    _log(msg)


def capture_frame(row, cam_actor, cam_state, plans, pairs=None):
    """一帧的完整落盘：三通道 PNG → 逐帧 JSON → 返回 manifest（done 标记由 main 写）。

    时序上先出图、后写 JSON：JSON 里的 bbox 只依赖世界坐标与相机参数，不依赖像素；
    但如果先写 JSON 再崩在出图上，重跑时 skip_existing 会因为缺图而重跑，不会误跳过。
    pairs = [(actor, plan)]（实际生成的）；生成失败的目标按计划兜底进真值，宁可近似也不丢帧。
    """
    frame_id = frame_id_of(row)
    rgb_p, depth_p, sem_p = capture_channels(frame_id, cam_actor, cam_state)
    actual_bit = 16 if rgb_p.lower().endswith(".exr") else png_header_bit_depth(rgb_p)
    if actual_bit is None:
        raise RuntimeError("RGB 输出格式无法识别: %s" % rgb_p)
    if actual_bit != BIT_DEPTH:
        _log_once("bitdepth", "⚠️ RGB 实际落盘 %dbit（期望 %dbit）：16bit 链路未打通，"
                              "见文件头 TODO(阶段三)，真值已按实际位深记录" % (actual_bit, BIT_DEPTH))

    objects = []
    spawned_plans = set()
    for actor, plan in (pairs or []):
        if actor is None:
            continue
        objects.append(actor_to_object_spec(actor, plan))
        spawned_plans.add(id(plan))
    missing = [p for p in (plans or []) if id(p) not in spawned_plans]
    if missing:
        # 生成失败的目标也写进真值（用计划位置/尺寸），免得框和图像都不对却查不出来
        objects.extend(object_specs_from_plans(missing))
        _log("⚠️ %s: %d 个目标未生成，暂按计划值写真值" % (frame_id, len(missing)))

    # timestamp：优先任务行给的值，保证同参数重跑 JSON 逐字节一致；缺省 0.0。
    # TODO(阶段三): 若必须用真实仿真时间，可改成 unreal.GameplayStatics.get_time_seconds()，
    #   但那样同参数重跑的时间戳会变（可复现性下降），要与「断点续跑/重跑一致」权衡。
    timestamp = _as_float(row.get("timestamp"), 0.0)
    manifest = build_manifest(row, cam_state, objects, timestamp=timestamp,
                              bit_depth=actual_bit)
    manifest["image_files"].update(rgb=os.path.basename(rgb_p),
                                   depth=os.path.basename(depth_p),
                                   semantic=os.path.basename(sem_p))
    _rgb_p, _depth_p, _sem_p, meta_p = frame_paths(frame_id)
    write_manifest(manifest, meta_p)
    return manifest


def _write_error(task_id, exc):
    """单帧异常写错误日志（带时间）后继续；日志本身失败也不能掩盖原始错误。"""
    try:
        d = os.path.dirname(os.path.abspath(ERRLOG))
        if d:
            os.makedirs(d, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ERRLOG, "a", encoding="utf-8") as f:
            f.write("[%s] task=%s err=%s\n%s\n" % (ts, task_id, exc, traceback.format_exc()))
    except Exception as log_exc:
        _log("⚠️ 错误日志写入失败: %s（原始错误 task=%s: %s）" % (log_exc, task_id, exc))


def main():
    t0 = time.time()
    try:
        tasks = load_tasks(QUEUE)
    except Exception as e:
        _write_error("<queue>", e)
        _log("任务清单读取失败，整批不启动：%s" % e)
        return 1

    _log("清单 %s 共 %d 条｜%dx%d %dbit｜深度 0~%gm｜模板值 %s"
         % (QUEUE, len(tasks), DEFAULT_WIDTH, DEFAULT_HEIGHT, BIT_DEPTH,
            DEPTH_FAR, SEMANTIC_STENCIL))
    if not tasks:
        _log("清单为空，直接退出")
        return 0

    ok, skipped, failed = 0, 0, 0
    current_map = None
    spawned = []
    total = len(tasks)
    for i, row in enumerate(tasks, 1):
        row = row or {}
        tid = str(row.get("id") or frame_id_of(row))
        frame_id = frame_id_of(row)

        if is_done(tid) or skip_existing(frame_id):
            skipped += 1
        else:
            try:
                # 先清上一帧目标，再换关卡：换关卡后旧 actor 已失效，避免对着失效对象销毁刷日志
                clear_spawned(spawned)
                spawned = []
                map_path = row.get("map")
                if map_path and map_path != current_map:
                    load_level(map_path)
                    current_map = map_path

                rng = task_rng(row)
                plans = plan_targets(row.get("targets"), rng)
                spawned = spawn_targets(plans)
                apply_semantic_stencil(spawned)

                light = lighting_with_condition(row.get("lighting"))
                set_lighting(light)
                # 夜间工况把光圈覆盖成 f/1.8（方案表 6），所以 condition 要传给相机
                cam_actor, cam_state = place_camera(row.get("camera"), light.get("condition"))
                capture_frame(row, cam_actor, cam_state, plans, spawned)
                mark_done(tid)
                ok += 1
            except Exception as e:
                # 单帧异常写日志并继续（方案 3.5 第三条）：不允许整批中断
                failed += 1
                _write_error(tid, e)
                _log("task=%s 失败: %s（继续下一条）" % (tid, e))

        if i % PROGRESS_EVERY == 0:
            _log("进度 %d/%d  ok=%d skip=%d fail=%d  已用 %.1fs"
                 % (i, total, ok, skipped, failed, time.time() - t0))

    elapsed = time.time() - t0
    rate = (ok / elapsed) if elapsed > 0 and ok else 0.0
    _log("完成 ok=%d skip=%d fail=%d 共 %d 条 用时 %.1fs（%.2f 帧/秒）→ %s / %s"
         % (ok, skipped, failed, total, elapsed, rate, RAW, META))
    if failed:
        _log("⚠️ 有 %d 帧失败，详见 %s；批次已跑完，修完可重跑续跑" % (failed, ERRLOG))
    # 退出码：有失败帧就返回 1（批次仍跑完、可续跑），让 tmux/看门脚本和 dashboard 能看见
    return 1 if failed else 0


if __name__ == "__main__":
    main()
