#!/usr/bin/env python3
"""按参数渲染 OBJ 合成图（前端实时调参用）。

参数：
  count      物体数量
  cls        类别
  dist_min   最近距离 m
  dist_max   最远距离 m
  lat        横向范围 ±m
  scale      尺寸倍率
  shadow     阴影强度
  occlusion  遮挡裁剪
  light      auto/day/night
"""
import argparse
import glob
import json
import os
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")
from asset_lib import OBJ_ROOT, pick, refresh
from obj_render_fixes import (crop_sprite_by_alpha, solidify_alpha,
                             contact_shadow)
from data_source import list_clips
from label_gate import list_clip_frames, load_frame_labels
from real_calib import body_to_pixel_real

REAL_ASSETS = "/data/obj_assets_real"
# PATCH-SPRITE: Blender 渲染的 GLB 贴片库（按类别 + yaw 角度）
PATCH_ROOT = "/data/assets_global/OBJ_PATCH"

try:
    from ground_masks import foreground_mask, hood_mask, road_mask
    HAS_MASKS = True
except ImportError:
    HAS_MASKS = False


def _pick_patch(cls, prefer_yaw_deg=180.0):
    """PATCH-SPRITE: 从 OBJ_PATCH 按 yaw 选最接近的贴片。"""
    d = os.path.join(PATCH_ROOT, cls)
    if not os.path.isdir(d):
        return None, None
    fs = sorted(glob.glob(os.path.join(d, "*_yaw*.png")))
    if not fs:
        return None, None
    best = None
    best_d = 1e9
    for f in fs:
        b = os.path.basename(f)
        try:
            yaw = int(b.split("_yaw")[1].split(".")[0])
        except Exception:
            continue
        # 环形角差
        diff = abs((yaw - prefer_yaw_deg + 180) % 360 - 180)
        if diff < best_d:
            best_d = diff
            best = f
    if best is None:
        return None, None
    im = cv2.imread(best, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None, None
    return im, {"size_m": _real_patch_size(cls),
                "source": "blender_patch",
                "file": os.path.basename(best),
                "yaw": int(os.path.basename(best).split("_yaw")[1].split(".")[0])}


def _real_patch_size(cls):
    """OBJ_PATCH 的真实尺寸（米），与 batch_kenney_render.py 的 CLASS_SPEC 一致。"""
    return {
        "car":   {"width": 1.80, "height": 1.50, "length": 4.50},
        "suv":   {"width": 1.90, "height": 1.75, "length": 4.70},
        "van":   {"width": 1.95, "height": 2.10, "length": 5.20},
        "truck": {"width": 2.30, "height": 2.90, "length": 7.00},
        "cone":  {"width": 0.40, "height": 0.70, "length": 0.40},
        # PATCH-SIZE-ALL: 补齐 bus/bicycle/motorcycle/barrier
        "bus":      {"width": 2.50, "height": 3.20, "length": 11.0},
        "bicycle":  {"width": 0.60, "height": 1.20, "length": 1.80},
        "motorcycle": {"width": 0.80, "height": 1.30, "length": 2.20},
        "barrier":  {"width": 0.50, "height": 0.90, "length": 2.00},
    }.get(cls, {"width": 1.80, "height": 1.50, "length": 4.50})


_GOOD_CACHE = {}


def _load_good_list():
    """GOOD-LIST: 加载质检合格的抠图清单（按类别分组）。"""
    global _GOOD_CACHE
    if _GOOD_CACHE:
        return _GOOD_CACHE
    # DISABLE-OLD-CUTOUT: 老抠图已废弃，改用 canonical 3D
    fp = "/opt/ue5_synth/tools/__disabled__rear_cutouts.json"
    out = {}
    if os.path.isfile(fp):
        try:
            import json as _j
            raw = _j.load(open(fp, encoding="utf-8"))
            # REAR-STRUCT-FIX: rear_cutouts.json 是 {cls: [item...]}
            if isinstance(raw, dict):
                for _c, _items in raw.items():
                    out[_c] = [it["path"] for it in _items
                               if isinstance(it, dict) and it.get("path")]
            else:
                for it in raw:
                    out.setdefault(it["cls"], []).append(it["path"])
        except Exception as e:
            print("WARN good_cutouts 加载失败:", e)
            out = {}
    _GOOD_CACHE = out
    return out


CANON_INDEX = "/data/assets_global/OBJ_3D/asset_index_obj3d.json"
_CANON_CACHE = {}


def _load_canon_index():
    """CANON-3D-FIRST: 加载规范化 3D 资产索引。"""
    global _CANON_CACHE
    if _CANON_CACHE:
        return _CANON_CACHE
    out = {}
    if os.path.isfile(CANON_INDEX):
        try:
            import json as _j
            d = _j.load(open(CANON_INDEX, encoding="utf-8"))
            for a in (d.get("assets") or []):
                c = a.get("true_class") or "car"
                out.setdefault(c, []).append(a)
        except Exception:
            out = {}
    _CANON_CACHE = out
    return out


def _pick_canonical(cls, asset_id=None):
    """返回 (None, meta) 表示走 3D 路径；这里只做标记。"""
    idx = _load_canon_index()
    lst = idx.get(cls) or []
    if asset_id:
        lst = [a for a in lst if a.get('uid') == asset_id]
        if not lst:
            raise ValueError('指定素材不存在或与类别不匹配: %s' % asset_id)
    if not lst:
        return None
    import hashlib as _hl
    import random as _rnd
    _seed = int.from_bytes(_hl.sha256(str(cls).encode("utf-8")).digest()[:8],
                           "little")
    a = lst[_rnd.Random(_seed).randrange(len(lst))]
    sz = a.get("lwh") or [4.6, 1.9, 1.4]
    # 返回特殊标记，调用方据此走 Blender 3D
    return ({"_canon3d": True, "path": a.get("path"),
             "uid": a.get("uid"), "lwh": sz,
             "size_m": {"length": sz[0], "width": sz[1],
                        "height": sz[2]}},
            {"size_m": {"length": sz[0], "width": sz[1],
                        "height": sz[2]},
             "source": "canon3d", "file": a.get("uid"),
             "path": a.get("path")})


def find_sprite(cls, light, scale=1.0, yaw_deg=180.0, prefer=None, asset_id=None):
    """REAL-FIRST: 真实抠图优先（质量最好），Blender 贴片兜底。

    prefer:
        'real'  -> 强制用真实抠图
        'patch' -> 强制用 Blender 贴片
        None    -> 真实抠图优先，没有再退贴片
    """
    # CANON-3D-FIRST: 优先 canonical 3D 模型
    _canon = _pick_canonical(cls, asset_id=asset_id)
    if _canon is not None:
        return _canon

    if prefer != "patch":
        # GOOD-LIST: 只从质检合格的抠图里选（剔除碎片）
        _good = _load_good_list().get(cls) or []
        if _good:
            import random as _rnd
            _pickf = _good[_rnd.Random(hash(cls) & 0xFFFF).randrange(
                len(_good))]
            _im = cv2.imread(_pickf, cv2.IMREAD_UNCHANGED)
            if _im is not None and _im.ndim == 3 and _im.shape[2] == 4:
                return _im, {"size_m": _real_size(cls),
                             "source": "real_cutout",
                             "file": os.path.basename(_pickf)}

    # Blender 贴片
    im, meta = _pick_patch(cls, prefer_yaw_deg=yaw_deg)
    if im is not None:
        return im, meta

    # 真实素材（旧路径兜底）
    if os.path.isdir(os.path.join(REAL_ASSETS, cls)):
        fs = sorted(glob.glob(os.path.join(REAL_ASSETS, cls, "*.png")))
        if fs:
            im = cv2.imread(fs[0], cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                return im, {"size_m": _real_size(cls), "source": "real_extract"}

    path, meta = pick(cls, "front", light, OBJ_ROOT)
    if path and os.path.isfile(path):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            return im, meta
    return None, None


def _real_size(cls):
    """真实素材的默认尺寸（米）。"""
    return {
        "car": {"width": 1.80, "height": 1.50, "length": 4.30},
        "suv": {"width": 1.90, "height": 1.75, "length": 4.50},
        "truck": {"width": 2.00, "height": 2.30, "length": 5.20},
        "van": {"width": 2.00, "height": 2.20, "length": 5.00},
        "pedestrian": {"width": 0.55, "height": 1.70, "length": 0.50},
        "rider": {"width": 0.70, "height": 1.75, "length": 1.00},
        "three": {"width": 1.30, "height": 1.60, "length": 1.80},
    }.get(cls, {"width": 1.0, "height": 1.5, "length": 2.0})


def project_quad(cam, x, y, l_m, h_m, w_m=None):
    """QUAD-3D-FIX: 用真实 3D 点投影四边形，保证轮胎精确贴地。

    参数:
        x, y    目标中心（body 坐标，米；x 纵向，y 横向）
        l_m     车长（沿 x 轴）
        h_m     车高
        w_m     车宽（沿 y 轴；两轮车用窄值，四轮车可传）

    返回 4 点 [(lt),(rt),(rb),(lb)]，依次为左上/右上/右下/左下。
        - 底边两点：地面 z=0 上 (x ± l/2, y)
        - 顶边两点：地面 z=h 上 (x ± l/2, y)
    这样无论透视如何，底边都严格落在地面，车顶按真实高度。
    """
    half = l_m / 2.0
    # 四个 3D 点（左下、右下用地面 z=0；左上、右上用 z=h）
    pts = [
        [x - half, y, 0.0],    # 0 地面-左
        [x + half, y, 0.0],    # 1 地面-右
        [x - half, y, h_m],    # 2 顶-左
        [x + half, y, h_m],    # 3 顶-右
    ]
    uv, z, ok = body_to_pixel_real(cam, pts)
    if not ok.all() or not np.isfinite(uv).all():
        return None
    # 顺序：左上、右上、右下、左下
    return np.array([uv[2], uv[3], uv[1], uv[0]], np.float32)


def _run_canonical_branch(cfg, pp, frames, out, sp, cls,
                          dmin, dmax, lat, scl):
    """CANON-BRANCH: 用 Blender 渲染 canonical 3D 模型。"""
    import subprocess
    import tempfile
    BL = "/data/blender-4.2.23-linux-x64/blender"
    RP = "/opt/ue5_synth/tools/render_canonical_one.py"
    _checker = None
    _extractor = None
    Kn_ = Rr = tv_ = None

    lwh = sp.get("lwh") or [4.6, 1.9, 1.4]
    from obj_harmonize import ForegroundHarmonizer
    harmonizer = ForegroundHarmonizer() if pp.get('harmonize', True) else None
    count = int(pp.get("count", 3))
    os.makedirs(out, exist_ok=True)

    # 每个 GLB 的最低网格不一定是轮胎底，资产级标定优先于类别默认值。
    asset_ground = {}
    ground_cfg = "/opt/ue5_synth/config/obj_asset_ground_offsets.json"
    try:
        with open(ground_cfg, encoding="utf-8") as f:
            asset_ground = (json.load(f).get("by_asset_uid") or {})
    except Exception as e:
        print("  接地偏移配置不可用:", str(e)[:80])

    # CLIP-TRACK: 整段共享同一世界轨迹，禁止逐帧重新随机落点。
    from obj_clip_trajectory import (compute_kiss_poses,
                                     compute_xml_ego_poses, plan_tracks_3d,
                                     pose_at, stable_seed)
    stems = [os.path.splitext(os.path.basename(p))[0] for p, _ in frames]
    clip_root = os.path.dirname(os.path.dirname(frames[0][0]))
    cache_key = "%016x" % stable_seed(clip_root, *stems)
    cache_path = os.path.join("/data/avsynth/cache/ego_motion",
                              cache_key + ".json")
    try:
        ego_poses, trajectory_mode = compute_kiss_poses(
            clip_root, stems, cache_path=cache_path)
    except Exception as e:
        print("  KISS 轨迹不可用，改用 XML EgoPose SE2:", str(e)[:100])
        try:
            ego_poses = compute_xml_ego_poses(clip_root, stems)
            trajectory_mode = "xml_ego_se2"
        except Exception as e2:
            print("  XML 轨迹也不可用，退回固定相对位置（不跳变）:",
                  str(e2)[:100])
            ego_poses = {s: np.eye(4) for s in stems}
            trajectory_mode = "fixed_relative_fallback"
    relative_speed = float(pp.get("relative_speed_mps", -1.0))
    # 先读完整片段真实 3D 框，再全局搜索连续轨迹；禁止仅看首帧。
    labels_by_stem = {}
    for (frame_path, xml_name), s in zip(frames, stems):
        try:
            d = load_frame_labels(clip_root, xml_path=xml_name)
            labels_by_stem[s] = list((d or {}).get("objects") or [])
        except Exception:
            labels_by_stem[s] = []
    # SMOOTH-TRACK-INTEGRATE: 片段级平滑轨迹（复演自车轨迹 + 预检 + 锁死）
    from smooth_track import (plan_smooth_track, smooth_ground_z,
                              build_ego_forward_track, world_to_lidar)
    from lidar_depth_core import load_points as _lp2
    from lidar_depth_core import fit_ground_plane as _fg2
    from cloud_collision import extract_obstacle_points as _eop2

    # 预加载 50 帧障碍点云 + 地面高度
    _obst_by_stem, _planes_by_stem, _z_raw = {}, {}, []
    for _s in stems:
        _bp = os.path.join(clip_root, "lidar_hs_pandar128_topmiddle",
                           _s + ".bin")
        if not os.path.isfile(_bp):
            _obst_by_stem[_s] = np.zeros((0, 3), np.float32)
            _planes_by_stem[_s] = None
            _z_raw.append(None)
            continue
        try:
            _xyz = _lp2(_bp)[:, :3]
            _pl = _fg2(_xyz)
            _planes_by_stem[_s] = _pl
            _obst_by_stem[_s] = (_eop2(_xyz, _pl) if _pl
                                 else np.zeros((0, 3), np.float32))
            _z_raw.append(_pl["ground_z_at_15m"]
                          if _pl and _pl.get("sane") else None)
        except Exception:
            _obst_by_stem[_s] = np.zeros((0, 3), np.float32)
            _planes_by_stem[_s] = None
            _z_raw.append(None)

    # 地面高度时序平滑（消除上下抖动）
    _z_smooth = smooth_ground_z(_z_raw, window=11, poly=2)
    _gz_by_stem = {s: _z_smooth[i] for i, s in enumerate(stems)}
    if _z_raw:
        _vals = [v for v in _z_raw if v is not None]
        if _vals:
            print("  地面 Z: 原始波动 %.3f -> 平滑后 %.3f" % (
                max(_vals) - min(_vals),
                max(_z_smooth) - min(_z_smooth)))

    # 片段级平滑轨迹规划
    _plan = plan_smooth_track(
        ego_poses, stems, _obst_by_stem, _planes_by_stem, lwh,
        offset_candidates=(12, 15, 18, 21, 25),
        lateral_candidates=tuple(float(v) for v in
                                 pp.get("lateral_candidates", (0.0, -3.5, 3.5))),
        max_collision_ratio=float(pp.get("max_collision_ratio", 0.20)),
        world_speed_mps=float(pp.get("world_speed_mps", 8.0)))

    smooth_lidar = {}
    _plan_ok = False
    if _plan:
        smooth_lidar = _plan["lidar"]
        _plan_ok = bool(_plan["precheck"]["ok"])
        print("  平滑轨迹 %s  预检 %s  碰撞 %d/%d" % (
            _plan["track"]["track_id"],
            "通过" if _plan["precheck"]["ok"] else "未通过",
            _plan["precheck"]["collision_frames"],
            _plan["precheck"]["total"]))
    else:
        print("  平滑轨迹规划失败，回退旧逻辑")
        tracks = plan_tracks_3d(
            ego_poses, stems, labels_by_stem, count, cls, lwh,
            (dmin, dmax), lat, relative_speed,
            stable_seed(clip_root, cls, sp.get("uid"), "clip_plan"),
            candidates_per_track=int(pp.get("trajectory_candidates", 500)))
        smooth_lidar = {}

    tracks = []
    trajectory_records = []

    # 稳定种子不依赖 Python 进程的 hash 随机化。
    import random as _rnd
    rng = _rnd.Random(stable_seed(clip_root, cls, sp.get("uid")))

    n = 0
    for frame_idx, (img_p, xml_n) in enumerate(frames):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        clip_dir = os.path.dirname(os.path.dirname(img_p))
        # XML-STEM-FIX: 同名 XML 自动补齐（优先 _align 目录）
        if not xml_n:
            import glob as _gx
            for sub in ("label_lidar_hs_pandar128_topmiddle_align/"
                        "cam_st_x2f_front",
                        "label_lidar_hs_pandar128_topmiddle"):
                c = _gx.glob(os.path.join(clip_dir, sub, stem + ".xml"))
                if c:
                    xml_n = c[0]
                    break
        # 每帧必须使用同 stem 的 XML；复用首帧标定/标签会造成时序错配。
        K, T, frame_labels = None, None, None
        try:
            if not xml_n:
                raise ValueError("missing same-stem XML")
            frame_labels = load_frame_labels(clip_dir, xml_path=xml_n)
            cam = frame_labels["camera"]
            Kn = np.asarray(cam["K"], np.float64)
            R = np.asarray(cam["R"], np.float64)
            tv = np.asarray(cam["t"], np.float64)
            raw = cv2.imread(img_p)
            if raw is None:
                raise ValueError("unreadable frame")
            H, W = raw.shape[:2]
            K = {"fx": float(Kn[0, 0]), "fy": float(Kn[1, 1]),
                 "cx": float(Kn[0, 2]), "cy": float(Kn[1, 2]),
                 "W": W, "H": H}
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tv
            Kn_ = {"fx": K["fx"], "fy": K["fy"],
                   "cx": K["cx"], "cy": K["cy"]}
            Rr, tv_ = R, tv
        except Exception as e:
            print("  [%s] 同帧标定/标签不可用: %s" % (stem, str(e)[:80]))
        if K is None:
            continue

        # Blender 是针孔相机；统一使用保持原 K 的去畸变画面。
        dist = np.asarray(cam.get("distortion") or [], np.float64)
        und_p = os.path.join(out, "_und_%s.jpg" % stem)
        if dist.size:
            raw = cv2.undistort(raw, Kn, dist, None, Kn)
        cv2.imwrite(und_p, raw, [cv2.IMWRITE_JPEG_QUALITY, 96])

        # 同 stem LiDAR + 当前帧地面，供 3D 碰撞与遮挡共同使用。
        lidar_xyz, plane = None, None
        try:
            from lidar_depth_core import load_points, fit_ground_plane
            lp = os.path.join(clip_dir, "lidar_hs_pandar128_topmiddle",
                              stem + ".bin")
            if os.path.isfile(lp):
                lidar_xyz = load_points(lp)[:, :3]
                plane = fit_ground_plane(lidar_xyz,
                                         seed=stable_seed(stem) & 0xffff)
                if not plane or not plane.get("sane"):
                    plane = None
        except Exception as e:
            print("  [%s] LiDAR/地面不可用: %s" % (stem, str(e)[:80]))

        # 每帧重算语义掩码；模型对象复用，禁止复用首帧掩码。
        vehicles = []
        try:
            from batch_canonical_v2 import MaskChecker
            _checker = MaskChecker(raw, Kn_, Rr, tv_, extractor=_extractor)
            _extractor = _checker.ext
        except Exception as e:
            print("  [%s] MaskChecker 不可用: %s" % (stem, str(e)[:60]))
            _checker = None

        gate3d = bool(frame_labels and lidar_xyz is not None and plane)
        placed_labels = list(frame_labels.get("objects") or []) if frame_labels else []

        _tries = 0
        _tid0 = ("ego_fwd_%s" % cls) if smooth_lidar else ("synthetic_0")
        for i in range(count):
            placed = False
            # SMOOTH-USE-IN-LOOP: 优先用平滑轨迹（片段级锁死，不重采样）
            _sd = smooth_lidar.get(stem) if smooth_lidar else None
            # 当前平滑规划器只生成一条轨迹；不得为每个数量重复使用它。
            if _sd is not None and i > 0:
                print("  平滑轨迹仅支持一个目标，跳过额外目标 %d" % i)
                break
            if smooth_lidar and _sd is None and i == 0 and _tries < 2:
                print("  [STEM-DEBUG] stem=%r  keys=%r" % (
                    stem, list(smooth_lidar.keys())[:2]))
            existing_track = tracks[i] if i < len(tracks) else None
            max_tries = 1 if (_sd is not None or existing_track is not None) else 0
            local_tries = 0
            fail_reason = "no_candidate"
            while local_tries < max_tries and not placed:
                local_tries += 1
                _tries += 1
                if _sd is not None:
                    x = _sd["x"]
                    y = _sd["y"]
                    yaw = math.degrees(_sd["yaw_rad"])
                    if i == 0 and _tries < 3:
                        print("  [XY-DEBUG] stem=%s x=%.2f y=%.2f yaw=%.1f lat=%.1f" % (
                            stem, x, y, yaw, lat))
                    if x < 2.0 or x > 80.0 or abs(y) > max(lat + 2.0, 8.0):
                        fail_reason = "track_out_of_range"
                        continue
                elif existing_track is not None:
                    x, y, yaw = pose_at(existing_track,
                                        ego_poses[stem], stem)
                    if x < 2.0 or x > 80.0 or abs(y) > max(lat + 2.0, 8.0):
                        fail_reason = "track_out_of_range"
                        continue
                else:
                    fail_reason = "no_clip_level_track"
                    continue
                # 道路门禁与真实物体门禁分离。3D 数据齐全时允许自然的
                # 2D 重叠；数据缺失时退回保守的像素级拒绝。
                if _checker is not None:
                    okf, foot, why = _checker.footprint_ok(
                        x, y, lwh[0], lwh[1], yaw,
                        reject_vehicle=not gate3d)
                    if not okf:
                        fail_reason = why
                        continue
                if gate3d:
                    from lidar_depth_core import ground_z as _gzp
                    # SMOOTH-GZ: 用时序平滑后的地面高度
                    gz = _gz_by_stem.get(stem)
                    if gz is None:
                        gz = _gzp(plane, x, y)
                    # CLOUD-ONLY-GATE: 只用点云盒体判据（片段级预检已保证）
                    #   不再用 XML 标注框 margin=0.25 硬拒（标注本身有误差）
                    if _sd is not None and _plan_ok:
                        pass          # 片段级预检已通过 -> 直接放行
                    else:
                        from cloud_collision import check_placement_cloud
                        _vo = {"x": x, "y": y, "z": 0.0, "yaw": yaw,
                               "size": list(lwh)}
                        _okc, _ic = check_placement_cloud(
                            _obst_by_stem.get(stem,
                                              np.zeros((0, 3), np.float32)),
                            _vo, _planes_by_stem.get(stem),
                            max_pts=15)
                        if not _okc:
                            fail_reason = "cloud_collision"
                            continue
                else:
                    gz = 0.0
                # 许多 GLB 的几何最低点并不是轮胎最低点（保险杠、排气或
                # 隐藏网格会更低）。仅按整车 bbox.min_z 接地会造成轮胎悬空。
                # 预览阶段做小幅接地沉降；物理碰撞框仍以真实地面 gz 为准。
                default_sink = {
                    "car": 0.08, "suv": 0.09, "van": 0.10,
                    "truck": 0.12, "bus": 0.12,
                    "motorcycle": 0.04, "bicycle": 0.03,
                    "cone": 0.01, "barrier": 0.02,
                }.get(cls, 0.06)
                calibrated_sink = (asset_ground.get(str(sp.get("uid"))) or
                                   {}).get("contact_sink_m", default_sink)
                contact_sink = max(0.0, min(
                    float(pp.get("contact_sink_m", calibrated_sink)), 0.20))
                render_z = gz - contact_sink
                _cur_tid = ((existing_track or {}).get("track_id")
                            or _tid0)
                vehicles.append({
                    "uid": sp.get("uid"), "glb": sp.get("path"),
                    "x": round(x, 2), "y": round(y, 2),
                    "z": round(render_z, 3),
                    "ground_z": round(gz, 3),
                    "contact_sink_m": round(contact_sink, 3),
                    "yaw": round(yaw, 1), "size": lwh, "class": cls,
                    "track_id": _cur_tid
                })
                # TRACKID-NONE-FIX: 用 existing_track 或平滑轨迹兜底
                if existing_track is not None:
                    vehicles[-1]["track_id"] = existing_track["track_id"]
                    vehicles[-1]["velocity_xy"] = [
                        round(float(q), 4)
                        for q in existing_track.get("velocity_xy", [0, 0])]
                    vehicles[-1]["motion_model"] = existing_track.get(
                        "motion_model", "ego_fwd")
                    vehicles[-1]["relative_speed_mps"] = existing_track.get(
                        "relative_speed_mps", 0.0)
                else:
                    vehicles[-1]["track_id"] = _tid0
                    _positions = _plan["track"]["positions"] if _plan else {}
                    _i0 = max(0, frame_idx - 1)
                    _i1 = min(len(stems) - 1, frame_idx + 1)
                    _dt = float(stems[_i1]) - float(stems[_i0])
                    if _dt > 0 and stems[_i0] in _positions and stems[_i1] in _positions:
                        _a = _positions[stems[_i0]]
                        _b = _positions[stems[_i1]]
                        vehicles[-1]["velocity_xy"] = [round((_b[k] - _a[k]) / _dt, 4) for k in (0, 1)]
                    else:
                        vehicles[-1]["velocity_xy"] = None
                    vehicles[-1]["motion_model"] = "ego_fwd_uniform"
                    vehicles[-1]["relative_speed_mps"] = None
                placed_labels.append({
                    "kind": "synthetic_" + cls, "delete": 0,
                    "Wx": x, "Wy": y, "Wz": gz + lwh[2] / 2.0,
                    "L": lwh[0], "W": lwh[1], "H": lwh[2],
                    "yaw": np.deg2rad(yaw),
                    "tracking_id": _cur_tid
                })
                trajectory_records.append({
                    "stem": stem, "track_id": _cur_tid,
                    "x": round(x, 4), "y": round(y, 4),
                    "ground_z": round(gz, 4), "yaw_deg": round(yaw, 4),
                    "rendered": True, "reason": "ok"
                })
                placed = True
            if not placed:
                # TRACKS-NONE-FIX: 有平滑轨迹或旧轨迹都算"存在"，不跳过
                if existing_track is not None or _sd is not None:
                    _tid = (existing_track["track_id"]
                            if existing_track is not None
                            else "ego_fwd")
                    trajectory_records.append({
                        "stem": stem, "track_id": _tid,
                        "rendered": False, "reason": fail_reason
                    })
                    print("  [%s] 轨迹目标暂时隐藏: %s" % (stem, fail_reason))
                    continue
                print("  [%s] 没有可用的片段级轨迹（跳过）" % stem)
                break

        ann_dir = os.path.join(out, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        if not vehicles:
            # 轨迹帧被遮挡/碰撞时保留原图，避免播放器缺帧；BBox 为空。
            cv2.imwrite(os.path.join(out, stem + "_obj.jpg"), raw,
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            with open(os.path.join(ann_dir, stem + ".json"), "w",
                      encoding="utf-8") as f:
                json.dump({"frame": stem, "width": K["W"], "height": K["H"],
                           "objects": [], "frame_status": "track_hidden",
                           "trajectory_mode": trajectory_mode}, f,
                          ensure_ascii=False, indent=2)
            try:
                os.remove(und_p)
            except OSError:
                pass
            continue

        # 逐目标生成 3D 八角点及投影框；可见框在合成后进一步收紧。
        ann_objects = []
        occ_stat = None
        for v in vehicles:
            L, Wd, Hh = [float(q) for q in v["size"]]
            cx, cy = float(v["x"]), float(v["y"])
            cz = float(v["z"]) + Hh / 2.0
            yr = np.deg2rad(float(v["yaw"]))
            co, si = np.cos(yr), np.sin(yr)
            corners3d = []
            for dz in (-Hh / 2.0, Hh / 2.0):
                for dx, dy in ((-L/2, -Wd/2), (L/2, -Wd/2),
                               (L/2, Wd/2), (-L/2, Wd/2)):
                    corners3d.append([cx + dx*co - dy*si,
                                      cy + dx*si + dy*co, cz + dz])
            uv, zz, ok = body_to_pixel_real(cam, corners3d)
            valid = ok & np.isfinite(uv).all(axis=1)
            corners2d = [[round(float(a), 2), round(float(b), 2)]
                         for a, b in uv] if valid.all() else []
            if valid.any():
                vv = uv[valid]
                bbox = [max(0.0, float(vv[:, 0].min())),
                        max(0.0, float(vv[:, 1].min())),
                        min(float(K["W"] - 1), float(vv[:, 0].max())),
                        min(float(K["H"] - 1), float(vv[:, 1].max()))]
                bbox = [round(q, 2) for q in bbox]
            else:
                bbox = None
            ann_objects.append({
                "source": "synthetic", "class": cls,
                "track_id": v["track_id"], "in_front": bool(valid.any()),
                "bbox2d": bbox, "bbox2d_sim": bbox,
                "corners": corners2d,
                "box3d": {"x": cx, "y": cy, "z": float(v["ground_z"]),
                          "yaw_deg": float(v["yaw"]),
                          "dimensions_m": [L, Wd, Hh]},
                "velocity_mps": v.get("velocity_xy"),
                "velocity_frame": "first_lidar_world_xy",
                "relative_speed_mps": v.get("relative_speed_mps"),
                "motion_model": v.get("motion_model"),
                "contact_sink_m": float(v["contact_sink_m"]),
            })

        # REAL-BOX-EXPORT: 真实车 3D 框 -> 2D BBox
        try:
            import math as _m2
            _fl = (frame_labels or {}).get("objects") or []
            if stem.endswith(".664") and len(_fl) > 0:
                print("  [REALBOX-DEBUG] frame_labels objects=%d  Rr=%s tv_=%s" % (
                    len(_fl), type(Rr).__name__,
                    None if tv_ is None else np.asarray(tv_).shape))
            for _it in _fl:
                if _it.get("delete"):
                    continue
                try:
                    _L = float(_it.get("L") or 0)
                    _W2 = float(_it.get("W") or 0)
                    _H2 = float(_it.get("H") or 0)
                    _cx = float(_it.get("Wx") or 0)
                    _cy = float(_it.get("Wy") or 0)
                    _cz = float(_it.get("Wz") or 0)
                    _yw = float(_it.get("yaw") or 0)
                    if _L <= 0.1 or _W2 <= 0.1:
                        continue
                    _co, _si = _m2.cos(_yw), _m2.sin(_yw)
                    _rc = []
                    for _dx in (-_L/2, _L/2):
                        for _dy in (-_W2/2, _W2/2):
                            for _dz in (-_H2/2, _H2/2):
                                _rc.append([_cx + _dx*_co - _dy*_si,
                                            _cy + _dx*_si + _dy*_co,
                                            _cz + _dz])
                    # PP-IMPORT-FIX: 手动投影（避免依赖未导入函数）
                    from lidar_depth_core import project_points as _ppj
                    _Km = np.array([[Kn_["fx"], 0, Kn_["cx"]],
                                    [0, Kn_["fy"], Kn_["cy"]],
                                    [0, 0, 1.0]])
                    _uv, _zz, _ok = _ppj(
                        np.array(_rc), _Km, Rr, tv_)
                    _val = _ok & np.isfinite(_uv).all(axis=1)
                    if not _val.any():
                        continue
                    _vv = _uv[_val]
                    _bb = [round(float(max(0.0, _vv[:,0].min())), 2),
                           round(float(max(0.0, _vv[:,1].min())), 2),
                           round(float(min(K["W"]-1, _vv[:,0].max())), 2),
                           round(float(min(K["H"]-1, _vv[:,1].max())), 2)]
                    if _bb[2] - _bb[0] < 2 or _bb[3] - _bb[1] < 2:
                        continue
                    ann_objects.append({
                        "source": "real",
                        "class": str(_it.get("kind") or "vehicle"),
                        "track_id": str(_it.get("tracking_id") or ""),
                        "in_front": True,
                        "bbox2d": _bb, "bbox2d_sim": _bb,
                        "box3d": {"x": _cx, "y": _cy, "z": _cz,
                                  "yaw_deg": _m2.degrees(_yw),
                                  "dimensions_m": [_L, _W2, _H2]},
                    })
                except Exception as _e1:
                    if stem.endswith(".664"):
                        print("  [REALBOX-ERR1]", str(_e1)[:100])
                    continue
        except Exception as _e2:
            print("  [REALBOX-ERR2]", str(_e2)[:120])

        fg_p = os.path.join(out, "_fg_%s.png" % stem)
        dep_p = os.path.join(out, "_depth_%s.npy" % stem)
        for stale in (fg_p, dep_p):
            if os.path.isfile(stale):
                os.remove(stale)
        c = {"bg": und_p, "K": K, "T": T.tolist(),
             "vehicles": vehicles, "out": fg_p,
             "depth_npy": dep_p if gate3d and pp.get("occlusion", True) else None,
             "hdri_enabled": bool(pp.get("hdri_enabled", False)),
             "paint_coat_enabled": bool(pp.get("paint_coat_enabled", False)),
             "seed": stable_seed(stem) & 0xFFFF}
        cfp = os.path.join(out, "_cfg_%s.json" % stem)
        with open(cfp, "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False)

        try:
            result = subprocess.run([BL, "--background", "--python-exit-code", "7", "--python", RP, "--", cfp],
                                    capture_output=True, timeout=900)
            if result.returncode:
                print("  [%s] Blender exit %d: %s" % (stem, result.returncode,
                      result.stderr.decode("utf-8", "replace")[-400:]))
                continue
        except Exception as e:
            print("  blender fail:", str(e)[:60])
            continue

        # 合成
        bg = cv2.imread(und_p)
        if bg is not None and os.path.isfile(fg_p):
            fg = cv2.imread(fg_p, cv2.IMREAD_UNCHANGED)
            if fg is not None and fg.shape[2] == 4:
                if fg.shape[:2] != bg.shape[:2]:
                    fg = cv2.resize(fg, (bg.shape[1], bg.shape[0]))
                if gate3d and pp.get("occlusion", True) and os.path.isfile(dep_p):
                    from obj_occlusion_3d import (foreground_depth,
                                                  radial_to_camera_z,
                                                  composite)
                    rd, rd_stat = foreground_depth(
                        lidar_xyz, Kn, Rr, tv_, _checker.veh,
                        max_fill_px=6.0)
                    sd = radial_to_camera_z(np.load(dep_p), Kn)
                    comp, occ_stat = composite(bg, fg, sd, rd, _checker.veh, harmonizer=harmonizer)
                    print("  [%s] 3D门禁+遮挡 visible=%.3f lidar=%d" % (
                        stem, occ_stat["visible_ratio"], rd_stat["samples"]))
                elif gate3d and pp.get("occlusion", True):
                    print("  [%s] 缺少合成深度，跳过该帧避免错误遮挡" % stem)
                    continue
                else:
                    a = fg[:, :, 3:4].astype(np.float32) / 255.0
                    comp = (fg[:, :, :3].astype(np.float32) * a +
                            bg.astype(np.float32) * (1 - a))
                cv2.imwrite(os.path.join(out, stem + "_obj.jpg"),
                            np.clip(comp, 0, 255).astype(np.uint8),
                            [cv2.IMWRITE_JPEG_QUALITY, 93])
                synth_objects = [o for o in ann_objects if o.get("source") == "synthetic"]
                if len(synth_objects) == 1 and occ_stat:
                    synth = synth_objects[0]
                    if occ_stat.get("visible_bbox"):
                        synth["bbox2d_sim"] = occ_stat["visible_bbox"]
                    else:
                        synth["bbox2d_sim"] = None
                        synth["bbox2d_projected"] = synth["bbox2d"]
                        synth["bbox2d"] = None
                        synth["in_front"] = False
                    synth["visible_ratio"] = round(
                        float(occ_stat["visible_ratio"]), 4)
                    synth["occluded"] = bool(
                        occ_stat["visible_ratio"] < 0.999)
                with open(os.path.join(ann_dir, stem + ".json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"frame": stem, "width": K["W"],
                               "height": K["H"], "objects": ann_objects,
                               "frame_status": "synthetic",
                               "trajectory_mode": trajectory_mode}, f,
                              ensure_ascii=False, indent=2, allow_nan=False)
                n += 1
        for p in (fg_p, cfp, dep_p, und_p):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    manifest = {
        "schema_version": 1, "trajectory_mode": trajectory_mode,
        "ego_pose_cache": cache_path, "relative_speed_mps": relative_speed,
        "track_count": len(tracks), "frames_requested": len(frames),
        "frames_rendered": n, "records": trajectory_records,
        "training_eligible": False,
        "note": "persistent trajectory verified for preview; labels/export gate pending"
    }
    with open(os.path.join(out, "trajectory_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("canonical 产出 %d 帧, 轨迹模式=%s" % (n, trajectory_mode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    if (cfg.get("params") or {}).get("render_mode") == "real3d_preview":
        from obj_real3d_preview import main as run_real3d_preview
        return run_real3d_preview(a.config)
    batch, clip = cfg["batch"], cfg["clip"]
    pp = cfg["params"]
    out = cfg["out"]

    refresh()

    count = int(pp.get("count", 3))
    cls = str(pp.get("cls", "cone"))
    dmin = float(pp.get("dist_min", 8))
    dmax = float(pp.get("dist_max", 28))
    lat = float(pp.get("lat", 3.0))
    scl = float(pp.get("scale", 1.0))
    shadow = float(pp.get("shadow", 0.35))
    use_occ = bool(pp.get("occlusion", True))
    light_mode = str(pp.get("light", "auto"))

    # 定位 clip
    clips = list_clips()
    clip_dir = None
    for name, p, src in clips:
        if name != batch:
            continue
        cand = os.path.join(p, clip)
        if any(os.path.isdir(os.path.join(cand, cam)) for cam in ("cam_st_x2f_front", "cam_st_x8d_front", "cam_ld_x8d_front", "cam_hk_x8b_front")):
            clip_dir = cand
            break
    if not clip_dir:
        raise SystemExit("找不到 clip: %s/%s" % (batch, clip))

    # FULL-CLIP-OBJ: OBJ 以 clip 为单位，渲染整个 clip
    _all_frames = list_clip_frames(clip_dir)
    _nf_lim = cfg.get("frames", None)
    try:
        _nf_lim = int(_nf_lim) if _nf_lim not in (None, "", 0) else 0
    except Exception:
        _nf_lim = 0
    frames = _all_frames[:_nf_lim] if _nf_lim > 0 else _all_frames
    print("渲染整 clip: %d 帧" % len(frames))
    if not frames:
        raise SystemExit("无帧")

    # 标定
    lab0 = load_frame_labels(clip_dir, xml_path=frames[0][1])
    if lab0 is None:
        raise SystemExit("无标定")
    cam = lab0["camera"]

    is_night = "night" in batch.lower()
    light = ("night" if (light_mode == "auto" and is_night)
             else ("auto" if light_mode == "auto" else light_mode))
    if light == "auto":
        light = "day"

    sp, meta = find_sprite(cls, light, asset_id=pp.get('asset_id') or None)
    if sp is None:
        raise SystemExit("无素材: " + cls)

    # CANON-BRANCH: canonical 3D 模型 -> 走 Blender 渲染
    if isinstance(sp, dict) and sp.get("_canon3d"):
        print("渲染(canonical 3D): cls=%s uid=%s lwh=%s" % (
            cls, sp.get("uid"), sp.get("lwh")))
        _run_canonical_branch(cfg, pp, frames, out, sp, cls,
                              dmin, dmax, lat, scl)
        print("产出完成 -> %s" % out)
        return
    # SIZE-UNIFY + AR-ADAPT: 尺寸计算
    #   高度 H 取类别标准值（决定接地与高度）
    #   横向宽度按【贴片自身 alpha 宽高比】反推，避免拉长压扁
    sz = (meta or {}).get("size_m") or _real_patch_size(cls)
    h_m = float(sz.get("height", 1.5)) * scl
    w_m = float(sz.get("width", 1.0)) * scl

    # REAR-VIEW-MODE: 后视图抠图 -> 横向宽度用【车宽】，不是车长
    _src = (meta or {}).get("source", "")
    if _src == "real_cutout":
        l_m = w_m                       # 后视图看到的就是车宽
    else:
        l_m = float(sz.get("length", 4.5)) * scl

    print("渲染: cls=%s count=%d dist=[%.0f,%.0f] lat=±%.1f scale=%.2f sprite=%s"
          % (cls, count, dmin, dmax, lat, scl,
             (meta or {}).get("source", "?")))
    # SIZE-DEBUG
    print("SIZE: h_m=%.2f  l_m=%.2f  w_m=%.2f  sprite=%dx%d  ar=%.2f"
          % (h_m, l_m, w_m, sp.shape[1], sp.shape[0],
             sp.shape[1] / max(sp.shape[0], 1)))

    n = 0
    for img_p, xml_n in frames:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        bg = cv2.imread(img_p)
        if bg is None:
            continue
        h, w = bg.shape[:2]

        if HAS_MASKS and use_occ:
            try:
                road = road_mask(bg)
                fg = foreground_mask(bg, road)
                hood = hood_mask(bg, frac=0.18)
                protect = np.clip(np.maximum(fg, hood), 0, 1)
            except Exception:
                protect = np.zeros((h, w), np.float32)
        else:
            protect = np.zeros((h, w), np.float32)

        out_im = bg.copy()

        # 均匀分布在 [dmin, dmax] × [-lat, lat]
        import random
        rng = random.Random(hash(stem) & 0xFFFFFFFF)
        positions = []
        for i in range(count):
            x = dmin + (dmax - dmin) * (i + 0.5) / max(count, 1)
            y = rng.uniform(-lat, lat)
            positions.append((x, y))
        positions.sort(key=lambda p: -p[0])

        # OBJ-3FIX: 贴片先裁紧 + 不透明化，保证轮胎贴地
        sp_c, _cbox = crop_sprite_by_alpha(sp)
        sp_c = solidify_alpha(sp_c, edge_px=1, feather_px=1)

        for x, y in positions:
            quad = project_quad(cam, x, y, l_m, h_m, w_m)
            if quad is None:
                continue
            if quad[:, 0].max() < 0 or quad[:, 0].min() > w:
                continue
            if quad[:, 1].max() < 0 or quad[:, 1].min() > h:
                continue

            sh, sw = sp_c.shape[:2]
            src = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1],
                            [0, sh - 1]], np.float32)
            Pm = cv2.getPerspectiveTransform(src, quad)

            a = sp_c[..., 3].astype(np.float32) / 255.0
            rgb = sp_c[..., :3].astype(np.float32)
            prem = rgb * a[..., None]

            a_w = cv2.warpPerspective(a, Pm, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
            p_w = cv2.warpPerspective(prem, Pm, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
            a_w = np.clip(a_w, 0, 1)
            straight = np.divide(p_w, np.maximum(a_w[..., None], 1e-6))
            straight[a_w <= 1e-4] = 0

            # OBJ-3FIX: 接触阴影（紧贴底边，窄带，非正圆盘）
            if shadow > 0.01:
                shm = contact_shadow(h, w, quad, strength=1.0,
                                     opacity=0.32 * (shadow / 0.35
                                                     if shadow > 0 else 1.0),
                                     spread=0.18, rise=0.10)
                out_im = (out_im.astype(np.float32)
                          * (1 - shm[..., None])).astype(np.uint8)

            # OBJ-3FIX: 不再对前景整体乘全局透明度
            #   只在车道线保护区域做混合（车辆遮挡车道线时）
            keep = a_w * (1.0 - protect)
            out_im = (out_im.astype(np.float32) * (1 - keep[..., None])
                      + straight * keep[..., None])
            # 车身主体直接覆盖（alpha 已是 255）
            body = (a_w > 0.98) & (protect < 0.01)
            if body.any():
                out_im[body] = straight[body]
            out_im = np.clip(out_im, 0, 255).astype(np.uint8)

        op = os.path.join(out, stem + "_obj.jpg")
        cv2.imwrite(op, out_im, [cv2.IMWRITE_JPEG_QUALITY, 93])
        n += 1

    print("产出 %d 帧 → %s" % (n, out))


if __name__ == "__main__":
    main()
