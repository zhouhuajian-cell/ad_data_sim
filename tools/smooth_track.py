#!/usr/bin/env python3
"""片段级平滑轨迹 + 地面时序平滑（修复帧间跳变）。

核心思路：
  ① 世界系轨迹：复演自车 KISS-ICP 轨迹前移 12~20m，
     天然落在可行驶路面上（不会开上地摊/货架）
  ② 片段级一次预检 + 锁死：整条 50 帧用 LiDAR 非地面点云校验，
     通过即锁死，中途绝不重采样
  ③ 地面高度时序平滑：Savitzky-Golay / 滑动平均，消除上下抖动
"""
import math
import os
import sys

import numpy as np


# ============================================================
# ③ 地面高度时序平滑
# ============================================================
def smooth_ground_z(z_list, window=11, poly=2):
    """对逐帧地面高度做时序平滑，消除 RANSAC 抖动。

    Args:
        z_list: 每帧的地面高度（None 表示该帧拟合失败）
    Returns:
        平滑后的列表（None 位置用相邻值填充）
    """
    n = len(z_list)
    if n == 0:
        return []
    arr = np.array([v if v is not None and np.isfinite(v) else np.nan
                    for v in z_list], dtype=float)
    # 先线性插值补空
    idx = np.arange(n)
    good = np.isfinite(arr)
    if good.sum() == 0:
        return [0.0] * n
    if good.sum() == 1:
        return [float(arr[good][0])] * n
    arr = np.interp(idx, idx[good], arr[good])

    # 滑动平均（窗口不超过序列长度）
    w = max(3, min(int(window), n if n % 2 else n - 1))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return arr.tolist()

    # Savitzky-Golay（无 scipy 时退化为滑动平均）
    try:
        from scipy.signal import savgol_filter
        p = min(int(poly), w - 1)
        return savgol_filter(arr, w, p).tolist()
    except Exception:
        kernel = np.ones(w) / w
        pad = w // 2
        padded = np.pad(arr, pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid").tolist()


# ============================================================
# ① 世界系平滑轨迹（复演自车轨迹前移）
# ============================================================
def build_ego_forward_track(poses, stems, offset_m=15.0,
                            lateral_m=0.0, speed_offset=0.0):
    """复演自车 KISS-ICP 轨迹，向前偏移 offset_m 米。

    这样生成的轨迹天然在自车走过的路面上，绝不会开上人行道/地摊。

    Args:
        poses: stem -> 4x4 位姿（第一帧世界系，LiDAR 系）
        stems: 帧序列
        offset_m: 沿自车前进方向的前移距离
        lateral_m: 横向偏移（车道选择）
        speed_offset: 相对自车的速度差（m/s），正=更快（会慢慢远离）
    Returns:
        track: dict(track_id, positions={stem: (x,y,z_world)})
    """
    # 自车世界系位置序列
    ego = []
    for s in stems:
        T = np.asarray(poses[s], float)
        ego.append(T[:3, 3])       # LiDAR 在世界系的位置
    ego = np.array(ego)

    # YAW-STABLE-FIX: 用全局主方向，避免相邻帧重合时 atan2 翻转
    n = len(stems)
    yaws = np.zeros(n)
    # 全局主方向：首尾位移
    d_main = ego[-1] - ego[0]
    if np.linalg.norm(d_main) < 0.5:
        # 自车几乎没动，用累积位移最大的方向
        initial = np.asarray(poses[stems[0]],float)[:3,0]
        d_main = np.array([initial[0], initial[1], 0.0])
        best = 0.5
        for i in range(n):
            for j in range(i + 1, n):
                dd = ego[j] - ego[i]
                nrm = np.linalg.norm(dd)
                if nrm > best:
                    best = nrm
                    d_main = dd
    yaw_main = math.atan2(d_main[1], d_main[0])

    # 每帧用窗口差分（前后各 5 帧），再退化到主方向
    win = 5
    for i in range(n):
        a = max(0, i - win)
        b = min(n - 1, i + win)
        d = ego[b] - ego[a]
        if np.linalg.norm(d[:2]) > 0.5:
            yaws[i] = math.atan2(d[1], d[0])
        else:
            yaws[i] = yaw_main

    # 角度解缠 + 平滑，消除残余抖动
    for i in range(1, n):
        while yaws[i] - yaws[i - 1] > math.pi:
            yaws[i] -= 2 * math.pi
        while yaws[i] - yaws[i - 1] < -math.pi:
            yaws[i] += 2 * math.pi
    if n >= 5:
        k = np.ones(5) / 5.0
        yaws = np.convolve(np.pad(yaws, 2, mode="edge"), k, mode="valid")

    # WORLD-ABS-FIX: 首帧定绝对世界坐标，之后按固定速度直线前进
    #   （不再"贴着自车"，避免永远和前方真车重叠）
    yaw0 = yaws[0]
    fx, fy = math.cos(yaw0), math.sin(yaw0)
    lx, ly = -math.sin(yaw0), math.cos(yaw0)
    # 首帧位置：自车位置 + 前移 offset_m + 横向 lateral_m
    px0 = ego[0][0] + fx * offset_m + lx * lateral_m
    py0 = ego[0][1] + fy * offset_m + ly * lateral_m
    # FIXED-SPEED: 虚拟车匀速直线（不跟自车变速）
    #   默认 8 m/s (29 km/h)；relative_speed_mps 若给出则用它
    if speed_offset is not None and abs(speed_offset) > 0.01:
        v_virt = abs(float(speed_offset))
    else:
        v_virt = 8.0

    positions = {}
    for i, s in enumerate(stems):
        t = i * 0.1                            # 秒
        px = px0 + fx * (v_virt * t)
        py = py0 + fy * (v_virt * t)
        positions[s] = (float(px), float(py), yaw0)

    return {"track_id": "ego_fwd_%.1f" % offset_m,
            "positions": positions,
            "offset_m": offset_m,
            "lateral_m": lateral_m,
            "speed_offset": speed_offset,
            "world_frame": "first_lidar"}


def build_ego_path_track(poses, stems, offset_m=18., lateral_m=0., relative_speed_mps=1.0, motion='straight', direction='same'):
    """World-space road-following preview using KISS poses; not a lane map.

    For near-stationary ego motion, use the first sensor forward axis rather
    than the direction of centimetre-scale positional noise. For moving ego,
    interpolate along its travelled path and extrapolate at the ends. Offsets
    and speeds are set once for the whole clip, never randomly per frame.
    """
    if not stems:raise ValueError('empty pose sequence')
    mats=[np.asarray(poses[s],float) for s in stems]
    if any(t.shape!=(4,4) or not np.isfinite(t).all() for t in mats):raise ValueError('invalid ego poses')
    xyz=np.array([t[:3,3] for t in mats]); times=np.array([float(s) for s in stems]);dt=times-times[0]
    if np.any(np.diff(times)<=0):raise ValueError('timestamps must increase')
    dist=np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1);arc=np.r_[0.,np.cumsum(dist)]
    spread=float(np.max(np.linalg.norm(xyz[:,:2]-xyz[0,:2],axis=1)))
    initial_yaw=math.atan2(mats[0][1,0],mats[0][0,0]);f0=np.array([math.cos(initial_yaw),math.sin(initial_yaw)])
    stationary=spread<0.5
    if not stationary:
        # Only trust a path with enough measurable translation; remove duplicate s.
        keep=np.r_[True,np.diff(arc)>.025]; ss=arc[keep];pts=xyz[keep,:2]
        if len(ss)<3:stationary=True
    def point_at(q):
        if stationary:
            return xyz[0,:2]+f0*q,f0
        def base(z):return np.array([np.interp(z,ss,pts[:,k]) for k in range(2)])
        z=max(0.,min(float(q),float(ss[-1])));pos=base(z)
        d=base(min(ss[-1],z+3.))-base(max(0.,z-3.)); norm=np.linalg.norm(d)
        tangent=d/norm if norm>.2 else f0
        # A near-stationary or reversing segment cannot define a safe forward road.
        if np.dot(tangent,f0)<-.25 and q<3.: tangent=f0
        if q>ss[-1]:pos=pos+tangent*(q-ss[-1])
        return pos,tangent
    positions={};v=float(relative_speed_mps)
    if not np.isfinite(v) or abs(v)>15:raise ValueError('unreasonable relative speed')
    for i,stem in enumerate(stems):
        progress=0. if stationary else float(arc[i]);q=max(2.,progress+float(offset_m)+v*float(dt[i]))
        center,tangent=point_at(q);normal=np.array([-tangent[1],tangent[0]])
        xy=center+normal*float(lateral_m);yaw=math.atan2(tangent[1],tangent[0])
        positions[stem]=(float(xy[0]),float(xy[1]),float(yaw))
    if motion not in ('straight','turn','right'):raise ValueError('invalid object motion')
    if direction not in ('same','oncoming'):raise ValueError('invalid travel direction')
    if motion in ('turn','right') or direction=='oncoming':
        origin=np.asarray(positions[stems[0]][:2],float);heading=float(positions[stems[0]][2])
        if direction=='oncoming':heading+=math.pi
        forward=np.array([math.cos(heading),math.sin(heading)]);left=np.array([-forward[1],forward[0]])
        measured=0. if stationary else float(arc[-1]/max(dt[-1],.1))
        speed=float(np.clip(measured+v,3.,18.));omega=math.radians(6.)*(1. if motion=='turn' else -1.) if motion!='straight' else 0.
        for i,stem in enumerate(stems):
            angle=omega*float(dt[i]);forward_dist=speed*math.sin(angle)/omega if omega else speed*float(dt[i])
            lateral_dist=speed*(1.-math.cos(angle))/omega if omega else 0.
            xy=origin+forward*forward_dist+left*lateral_dist
            positions[stem]=(float(xy[0]),float(xy[1]),heading+angle)
    return {'track_id':'ego_path_%.1f_%+.1f'%(offset_m,lateral_m),'positions':positions,
            'offset_m':offset_m,'lateral_m':lateral_m,'relative_speed_mps':v,'motion':motion,'direction':direction,
            'trajectory_kind':'ego_path_approx','stationary_ego':stationary,'world_frame':'first_lidar'}

def world_to_lidar(track, poses, stems):
    """世界系轨迹 -> 各帧 LiDAR 系坐标。"""
    out = {}
    for s in stems:
        T = np.asarray(poses[s], float)
        px, py, pyaw = track["positions"][s]
        # 世界 -> 当前 LiDAR
        P = np.array([px, py, 0.0])
        R = T[:3, :3]
        t = T[:3, 3]
        local = R.T @ (P - t)
        # yaw 转局部
        local_yaw = pyaw - math.atan2(R[1, 0], R[0, 0])
        out[s] = {"x": float(local[0]), "y": float(local[1]),
                  "yaw_rad": float(local_yaw)}
    return out


# ============================================================
# ② 片段级预检（整条轨迹一次校验，通过即锁死）
# ============================================================
def precheck_track(track_lidar, stems, obst_by_stem,
                   planes_by_stem, lwh,
                   max_collision_ratio=0.20,
                   max_pts=15,
                   expand_long=1.0, expand_lat=0.6,
                   x_min=2.0, x_max=80.0, y_max=10.0):
    """整条轨迹的片段级预检。

    Returns: dict(ok, collision_frames, total, ratio, reason)
    """
    from cloud_collision import check_placement_cloud

    n_coll = 0
    total = 0
    bad = []
    for s in stems:
        d = track_lidar.get(s)
        if d is None:
            continue
        x, y = d["x"], d["y"]
        # 视野范围
        if x < x_min or x > x_max or abs(y) > y_max:
            continue
        total += 1
        plane = planes_by_stem.get(s)
        virt = {"x": x, "y": y, "z": 0.0,
                "yaw": math.degrees(d["yaw_rad"]),
                "size": list(lwh)}
        ok, info = check_placement_cloud(
            obst_by_stem.get(s, np.zeros((0, 3), np.float32)),
            virt, plane, expand_long=expand_long,
            expand_lat=expand_lat, max_pts=max_pts)
        if not ok:
            n_coll += 1
            bad.append(s)

    ratio = (n_coll / total) if total else 1.0
    ok = (total >= max(3, len(stems) * 0.2)) and \
        (ratio <= max_collision_ratio)
    return {"ok": ok, "collision_frames": n_coll,
            "total": total, "ratio": round(ratio, 3),
            "bad_stems": bad[:10],
            "reason": "ok" if ok else
                      "碰撞率 %.0f%% > %.0f%%" % (ratio * 100,
                                                  max_collision_ratio * 100)}


def lock_track(track, stems):
    """锁死轨迹：返回不可变的逐帧位姿表。"""
    return {s: dict(track["positions"][s]) for s in stems}


# ============================================================
# 便捷入口
# ============================================================
def plan_smooth_track(poses, stems, obst_by_stem, planes_by_stem,
                      lwh, offset_candidates=(12, 15, 18, 21),
                      lateral_candidates=(0.0, -3.5, 3.5),
                      max_collision_ratio=0.20, max_pts=15,
                      world_speed_mps=8.0, trajectory_kind='straight', relative_speed_mps=1.0):
    """一次性规划并锁死整条轨迹（不逐帧重采样）。

    Returns: dict(track, lock, precheck) or None
    """
    best = None
    for off in offset_candidates:
        for lat in lateral_candidates:
            if trajectory_kind=='follow_road':
                trk = build_ego_path_track(poses, stems, offset_m=off, lateral_m=lat,
                                           relative_speed_mps=relative_speed_mps)
            else:
                trk = build_ego_forward_track(poses, stems, offset_m=off, lateral_m=lat,
                                              speed_offset=world_speed_mps)
            lid = world_to_lidar(trk, poses, stems)
            chk = precheck_track(lid, stems, obst_by_stem,
                                 planes_by_stem, lwh,
                                 max_collision_ratio=max_collision_ratio,
                                 max_pts=max_pts)
            # PLAN-RANK-FIX: 优先通过 + 碰撞率最低 + offset 适中（12~18）
            _off_pen = abs(off - 15.0)
            rank = (1 if chk["ok"] else 0,
                    -chk["ratio"],
                    chk["total"],
                    -abs(lat),
                    -_off_pen)
            if best is None or rank > best[0]:
                best = (rank, trk, lid, chk)
    if best is None:
        return None
    return {"track": best[1], "lidar": best[2], "precheck": best[3]}


def plan_multiple_tracks(poses, stems, obst_by_stem, planes_by_stem, asset_specs,
                         offset_candidates=(12.,18.,25.,32.,40.,50.,60.),
                         lateral_candidates=(0.,-3.5,3.5,-7.,7.), relative_speed_mps=1.,
                         image_pair_clear=None, visible_y_max=8.0):
    """Plan distinct targets over the clip; reject point-cloud and pairwise overlaps.

    Pairwise circumscribed circles are conservative for arbitrary yaw. This is
    a preview gate, not proof that unobserved real space is empty.
    """
    accepted=[]
    for index,spec in enumerate(asset_specs):
        size=np.asarray(spec['lwh'],float)
        if size.shape!=(3,) or np.any(~np.isfinite(size)) or np.any(size<=0):raise ValueError('invalid asset dimensions')
        best=None
        for off in offset_candidates:
            for lat in lateral_candidates:
                # Keep targets in separate lateral corridors; 3D-safe vehicles
                # placed one behind the other may otherwise collapse into one
                # confusing image-space overlap. This is only a preliminary gate.
                if any(abs(lat-p['track']['lateral_m'])<3.0 for p in accepted):continue
                if spec.get('direction')=='oncoming' and abs(lat)<3.:continue
                tr=build_ego_path_track(poses,stems,offset_m=off,lateral_m=lat,relative_speed_mps=relative_speed_mps,motion=spec.get('motion','straight'),direction=spec.get('direction','same'))
                tr['track_id']='synthetic_%d_%s'%(index,str(spec['uid']))
                local=world_to_lidar(tr,poses,stems)
                check=precheck_track(local,stems,obst_by_stem,planes_by_stem,size,max_collision_ratio=0.,y_max=visible_y_max)
                if not check['ok'] or check['total'] < max(3, math.ceil(len(stems)*0.8)):continue
                collision=False
                radius=.5*float(np.hypot(size[0],size[1]))
                for prev in accepted:
                    other=prev['size'];minimum=radius+.5*float(np.hypot(other[0],other[1]))+.5
                    for stem in stems:
                        a=tr['positions'][stem];b=prev['track']['positions'][stem]
                        if np.hypot(a[0]-b[0],a[1]-b[1])<minimum:
                            collision=True;break
                    if collision:break
                if collision:continue
                if image_pair_clear is not None and not image_pair_clear(local,size,accepted):continue
                rank=(check['total'],-abs(lat),-abs(off-(18.+index*10.)))
                if best is None or rank>best[0]:best=(rank,{'track':tr,'lidar':local,'precheck':check,'size':size.tolist()})
        if best is None:raise ValueError('无法为素材 %s 规划无冲突的连续轨迹'%spec['uid'])
        accepted.append(best[1])
    return accepted
