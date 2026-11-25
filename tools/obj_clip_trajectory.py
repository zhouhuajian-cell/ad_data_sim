#!/usr/bin/env python3
"""Clip-level ego poses and persistent synthetic-object trajectories."""
import glob
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np


DYNAMIC_CLASSES = {"car", "suv", "van", "truck", "bus",
                   "motorcycle", "bicycle", "rider"}


def compute_xml_ego_poses(clip_dir, stems):
    """Build deterministic SE(2)+height poses from per-frame EgoPoseParam.

    Heading is derived from the position tangent rather than trusting the
    vendor-specific XML Euler convention.  The returned frame matches KISS:
    first pose is identity and +X follows ego travel.
    """
    label_roots = [os.path.join(clip_dir, x) for x in os.listdir(clip_dir)
                   if x.startswith("label_lidar") and not x.endswith("_align")]
    if not label_roots:
        raise ValueError("missing label directory for XML ego poses")
    label_root = label_roots[0]
    xyz = []
    for stem in stems:
        path = os.path.join(label_root, stem + ".xml")
        if not os.path.isfile(path):
            raise ValueError("missing ego XML: " + stem)
        node = ET.parse(path).getroot().find(".//EgoPoseParam")
        if node is None or node.get("validFlag", "0") != "1":
            raise ValueError("invalid EgoPoseParam: " + stem)
        xyz.append([float(node.findtext(k)) for k in ("wx", "wy", "wz")])
    xyz = np.asarray(xyz, float)
    if not np.isfinite(xyz).all():
        raise ValueError("non-finite XML ego position")
    if len(xyz) > 1:
        step = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        if np.max(step) > 5.0:
            raise ValueError("XML ego position discontinuity")
    tangent = np.empty((len(xyz), 2), float)
    if len(xyz) == 1:
        tangent[0] = [1., 0.]
    else:
        tangent[0] = xyz[1, :2] - xyz[0, :2]
        tangent[-1] = xyz[-1, :2] - xyz[-2, :2]
        if len(xyz) > 2:
            tangent[1:-1] = xyz[2:, :2] - xyz[:-2, :2]
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm < 1e-5):
        raise ValueError("stationary/ambiguous XML ego heading")
    tangent /= norm[:, None]
    global_poses = []
    for pos, direction in zip(xyz, tangent):
        c, s = direction
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        T[:3, 3] = pos
        global_poses.append(T)
    inv0 = np.linalg.inv(global_poses[0])
    return {stem: inv0 @ T for stem, T in zip(stems, global_poses)}


def stable_seed(*parts):
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def compute_kiss_poses(clip_dir, stems, cache_path=None):
    """Return stem -> 4x4 LiDAR pose in the first-frame world frame."""
    if cache_path and os.path.isfile(cache_path):
        try:
            data = json.load(open(cache_path, encoding="utf-8"))
            poses = {x["stem"]: np.asarray(x["pose"], float)
                     for x in data.get("poses", [])}
            if all(s in poses for s in stems):
                return poses, "kiss_cache"
        except Exception:
            pass

    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP
    from lidar_depth_core import load_points

    cfg = KISSConfig()
    cfg.data.deskew = False
    cfg.mapping.voxel_size = 1.0
    cfg.registration.max_num_threads = 4
    icp = KissICP(cfg)
    poses = {}
    records = []
    for stem in stems:
        path = os.path.join(clip_dir, "lidar_hs_pandar128_topmiddle",
                            stem + ".bin")
        if not os.path.isfile(path):
            raise ValueError("missing LiDAR for trajectory: " + stem)
        xyz = load_points(path)[:, :3].astype(np.float64)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        icp.register_frame(xyz, np.empty(0, dtype=np.float64))
        pose = np.asarray(icp.last_pose, float)
        poses[stem] = pose
        records.append({"stem": stem, "pose": pose.tolist()})
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "poses": records,
                       "deskew": False}, f, ensure_ascii=False)
        os.replace(tmp, cache_path)
    return poses, "kiss_computed"


def motion_summary(poses, stems):
    ts = np.asarray([float(s) for s in stems], float)
    xy = np.asarray([poses[s][:2, 3] for s in stems], float)
    if len(stems) < 2:
        return {"speed_mps": 0.0, "direction": np.array([1., 0.])}
    dt = np.diff(ts)
    delta = np.diff(xy, axis=0)
    valid = dt > 1e-5
    speeds = np.linalg.norm(delta[valid], axis=1) / dt[valid]
    direction = np.median(delta[valid], axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1., 0.])
    else:
        direction = direction / norm
    return {"speed_mps": float(np.median(speeds)) if len(speeds) else 0.0,
            "direction": direction}


def create_track(track_id, cls, initial_local_xy, initial_yaw_deg,
                 poses, stems, relative_speed_mps=-1.0, anchor_stem=None):
    """Create one world-frame track from a first-frame body-frame placement."""
    first = anchor_stem or stems[0]
    T0 = poses[first]
    local = np.array([initial_local_xy[0], initial_local_xy[1], 0., 1.])
    world0 = T0 @ local
    summary = motion_summary(poses, stems)
    is_dynamic = cls in DYNAMIC_CLASSES
    if is_dynamic:
        speed = max(0.0, summary["speed_mps"] + float(relative_speed_mps))
    else:
        speed = 0.0
    ego_yaw0 = math.atan2(T0[1, 0], T0[0, 0])
    return {"track_id": str(track_id), "class": cls,
            "motion_model": ("ego_lane_relative" if is_dynamic
                             else "world_static"),
            "relative0": [float(initial_local_xy[0]),
                          float(initial_local_xy[1])],
            "relative_yaw_deg": float(initial_yaw_deg),
            "world0": world0[:3], "world_yaw": ego_yaw0 +
            math.radians(float(initial_yaw_deg)),
            "velocity_xy": summary["direction"] * speed,
            "t0": float(first), "relative_speed_mps": float(relative_speed_mps)}


def pose_at(track, pose, stem):
    """Transform a persistent world track into the current LiDAR/body frame."""
    dt = float(stem) - float(track["t0"])
    if track.get("motion_model") == "ego_lane_relative":
        # 动态交通参与者沿自车参考车道平滑运动。相对速度非零，因此不会
        # 粘在自车上；同时不把 GPS/ICP 的单帧横向和航向噪声传给目标。
        x = float(track["relative0"][0]) + float(
            track["relative_speed_mps"]) * dt
        y = float(track["relative0"][1])
        yaw = float(track["relative_yaw_deg"])
        return x, y, yaw
    world = np.asarray(track["world0"], float).copy()
    world[:2] += np.asarray(track["velocity_xy"], float) * dt
    local = np.linalg.inv(np.asarray(pose, float)) @ np.r_[world, 1.0]
    ego_yaw = math.atan2(pose[1, 0], pose[0, 0])
    yaw = math.degrees(float(track["world_yaw"]) - ego_yaw)
    yaw = (yaw + 180.0) % 360.0 - 180.0
    return float(local[0]), float(local[1]), float(yaw)


def plan_tracks_3d(poses, stems, labels_by_stem, count, cls, lwh,
                   distance_range, lateral_range, relative_speed_mps,
                   seed, candidates_per_track=500):
    """Search persistent tracks against all frames' annotated 3D boxes."""
    from obj_collision_3d import (annotation_box, candidate_box, overlaps)
    rng = np.random.RandomState(int(seed) & 0xffffffff)
    dmin, dmax = map(float, distance_range)
    lat = float(lateral_range)
    chosen = []
    chosen_boxes = {s: [] for s in stems}
    for track_index in range(int(count)):
        best = None
        for _ in range(int(candidates_per_track)):
            x0 = float(rng.uniform(dmin, dmax))
            y0 = float(rng.uniform(-lat, lat))
            yaw0 = float(180.0 + rng.uniform(-8.0, 8.0))
            track = create_track("synthetic_%d" % track_index, cls,
                                 (x0, y0), yaw0, poses, stems,
                                 relative_speed_mps=relative_speed_mps)
            valid = 0
            boxes = {}
            for stem in stems:
                x, y, yaw = pose_at(track, poses[stem], stem)
                if x < 2.0 or x > 80.0 or abs(y) > max(lat + 2.0, 8.0):
                    continue
                cand = candidate_box(x, y, 0.0, lwh, yaw)
                collision = False
                for item in labels_by_stem.get(stem) or []:
                    if item.get("delete"):
                        continue
                    try:
                        if overlaps(cand, annotation_box(item), margin=0.25):
                            collision = True
                            break
                    except (KeyError, TypeError, ValueError):
                        continue
                if not collision:
                    collision = any(overlaps(cand, other, margin=0.25)
                                    for other in chosen_boxes[stem])
                if collision:
                    continue
                valid += 1
                boxes[stem] = cand
            # Prefer full-clip validity, then farther placement for stability.
            rank = (valid, x0)
            if best is None or rank > best[0]:
                best = (rank, track, boxes)
                if valid == len(stems) and x0 > dmax - 0.5:
                    break
        if best is None or best[0][0] == 0:
            break
        track = best[1]
        track["planned_valid_frames"] = int(best[0][0])
        track["planned_total_frames"] = len(stems)
        chosen.append(track)
        for stem, box in best[2].items():
            chosen_boxes[stem].append(box)
    return chosen
