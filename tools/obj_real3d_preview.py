#!/usr/bin/env python3
"""Single-frame Blender OBJ preview for the dashboard (not training data)."""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, '/opt/ue5_synth/tools')
from data_source import list_clips
from ground_masks import road_mask
from label_gate import list_clip_frames, load_frame_labels
from real_calib import body_to_pixel_real

BLENDER = '/data/blender-4.2.23-linux-x64/blender'
RENDERER = '/opt/ue5_synth/tools/render_obj3d_frame.py'
INDEX = '/data/assets_global/OBJ_3D/asset_index_obj3d.json'
CANON = '/data/assets_global/OBJ_3D/canonical'
OBS = '/data/assets_global/OBJ_3D/_obs_dec'
# Nominal footprint for interactive preview only; models have no verified physical dimensions.
OBS_LWH = {'drum': (0.65,0.65,0.9), 'fence': (2.0,0.6,1.1),
           'forklift': (3.0,1.5,2.2), 'gate': (2.0,0.6,1.2),
           'pallet': (1.2,1.0,0.2), 'pedestrian': (0.6,0.6,1.7),
           'tire': (0.8,0.8,0.3), 'trashcan': (0.7,0.7,1.1),
           'water_barrier': (1.5,0.6,0.9), 'wheelbarrow': (1.4,0.8,0.9)}

def resolve_asset(uid, cls):
    if not uid or os.path.basename(uid) != uid:
        raise ValueError('请选择 3D 素材')
    if cls.startswith('OBS/'):
        category = cls[4:]
        if category not in OBS_LWH or not uid.isascii() or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in uid):
            raise ValueError('障碍物类别或素材无效')
        root = os.path.realpath(OBS)
        model = os.path.realpath(os.path.join(root, category, uid, 'mesh.glb'))
        if os.path.commonpath([model, root]) != root or not os.path.isfile(model):
            raise ValueError('障碍物模型不存在')
        return model, OBS_LWH[category]
    with open(INDEX, encoding='utf-8') as f:
        assets = json.load(f).get('assets', [])
    matches = [a for a in assets if a.get('uid') == uid and a.get('true_class') == cls]
    if len(matches) != 1:
        raise ValueError('素材不存在或与所选类别不匹配')
    a = matches[0]
    model = os.path.realpath(a.get('path', ''))
    root = os.path.realpath(CANON)
    if os.path.commonpath([model, root]) != root or not model.endswith('.glb') or not os.path.isfile(model):
        raise ValueError('素材文件不可用')
    lwh = a.get('lwh')
    if not isinstance(lwh, list) or len(lwh) != 3 or any(not isinstance(v, (int, float)) or not .1 < v < 30 for v in lwh):
        raise ValueError('素材尺寸信息无效')
    return model, lwh



def _mask_for_quad(shape, points):
    h, w = shape[:2]
    uv = np.asarray(points, dtype=np.float64)
    if uv.shape != (4, 2) or not np.isfinite(uv).all():
        return None
    if not ((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)).all():
        return None
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, np.rint(uv).astype(np.int32), 1)
    return m if m.sum() >= 30 else None


def _candidate(cam, road, blocked, x, y, length, width, ground_z):
    points = [[x+length/2, y+width/2, ground_z], [x-length/2, y+width/2, ground_z],
              [x-length/2, y-width/2, ground_z], [x+length/2, y-width/2, ground_z]]
    uv, _, ok = body_to_pixel_real(cam, points)
    if not np.all(ok):
        return False
    f = _mask_for_quad(road.shape, uv)
    if f is None:
        return False
    area = f > 0
    return float(np.mean(road[area] > .5)) >= .88 and float(np.mean(blocked[area])) < .005


def _clip(batch, clip):
    if not batch or not clip or os.path.basename(batch) != batch or os.path.basename(clip) != clip:
        raise ValueError('invalid batch or clip')
    for name, root, _ in list_clips():
        if name == batch:
            path = os.path.join(root, clip)
            if any(os.path.isdir(os.path.join(path, cam)) for cam in ("cam_st_x2f_front", "cam_st_x8d_front", "cam_ld_x8d_front", "cam_hk_x8b_front")):
                return path
    raise ValueError('clip not found')


def render_frame(config_path):
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)
    p = cfg.get('params') or {}
    if not isinstance(p.get('cls'), str) or p.get('edit_mode', 'add') != 'add':
        raise ValueError('写实 3D 预览目前只支持新增目标')
    if int(p.get('count', 1)) != 1:
        raise ValueError('写实 3D 预览仅支持单个车辆')
    asset, model_lwh = resolve_asset(p.get('asset_id', ''), p.get('cls'))
    if p.get('light', 'auto') == 'night':
        raise ValueError('写实 3D 预览尚未适配夜间光照')
    clip_dir = _clip(cfg.get('batch', ''), cfg.get('clip', ''))
    frames = list_clip_frames(clip_dir)
    requested_stem = str(p.get('preview_frame_stem') or '')
    if not requested_stem or os.path.basename(requested_stem) != requested_stem:
        raise ValueError('缺少当前帧标识，请重新选择帧')
    matches = [(img, xml) for img, xml in frames
               if os.path.splitext(os.path.basename(img))[0] == requested_stem]
    if len(matches) != 1:
        raise ValueError('当前帧与数据源不匹配，请重新选择片段')
    img_path, xml_name = matches[0]
    if not xml_name:
        raise ValueError('当前帧没有匹配的相机标定')
    labels = load_frame_labels(clip_dir, xml_path=xml_name)
    if not labels or not labels.get('camera') or not labels['camera'].get('valid', True):
        raise ValueError('当前帧相机标定不可用')
    cam = labels['camera']
    bg = cv2.imread(img_path)
    if bg is None:
        raise ValueError('当前帧图像不可读取')
    h, w = bg.shape[:2]
    road = road_mask(bg)
    if road is None or road.shape != (h, w):
        raise ValueError('当前帧路面掩码不可用')
    # The fixed lower-image exclusion is a conservative preview heuristic,
    # not a measured bonnet mask.
    road[int(h * .82):, :] = 0
    blocked = np.zeros((h, w), dtype=np.uint8)
    for item in labels.get('objects') or []:
        box = item.get('bbox2d')
        if box is None or len(box) != 4 or not np.isfinite(box).all():
            continue
        x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
        x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
        if x1 > x0 and y1 > y0:
            blocked[y0:y1, x0:x1] = 1
    blocked = cv2.dilate(blocked, np.ones((7, 7), np.uint8))
    scale = float(p.get('scale', 1.0))
    if not .5 <= scale <= 2:
        raise ValueError('尺寸倍率超出范围')
    length, width, height = [v * scale for v in model_lwh]
    ground_z = -.08  # visual preview adjustment, not surveyed road height
    dmin, dmax = float(p.get('dist_min', 8)), float(p.get('dist_max', 28))
    lat = float(p.get('lat', 3))
    if not (4 <= dmin <= dmax <= 80 and 0 <= lat <= 8):
        raise ValueError('距离范围无效')
    manual = p.get('manual_pos')
    if p.get('pos_on'):
        manual = [[float(p.get('pos_x', 20)), float(p.get('pos_y', 0))]]
    if manual:
        if len(manual) != 1 or len(manual[0]) != 2:
            raise ValueError('指定位置格式无效')
        # Dashboard says positive lateral is right; body coordinate is left-positive.
        choices = [(float(manual[0][0]), -float(manual[0][1]))]
    else:
        seed = int.from_bytes(hashlib.sha256(img_path.encode()).digest()[:8], 'big')
        rng = random.Random(seed)
        # Prefer useful mid/far candidates over visually oversized near targets.
        near = max(dmin, 15.0) if dmax >= 15.0 else dmin
        choices = [(rng.uniform(near, dmax), rng.uniform(-lat, lat)) for _ in range(300)]
    poses = [(x, y) for x, y in choices if 4 <= x <= 80 and abs(y) <= 8 and
             _candidate(cam, road, blocked, x, y, length, width, ground_z)]
    if not poses:
        raise ValueError('当前帧没有满足道路及真实目标约束的位置，请调整距离或横向范围')
    R = np.asarray(cam['R'], dtype=float)
    t = np.asarray(cam['t'], dtype=float)
    if R.shape != (3, 3) or t.size != 3 or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError('相机外参无效')
    K = dict(fx=cam['fx'], fy=cam['fy'], cx=cam['cx'], cy=cam['cy'], W=w, H=h,
             K=cam['K'], R_cam2body=R.T.tolist(), t_cam2body=(-R.T @ t.reshape(3)).tolist())
    out = cfg['out']
    os.makedirs(out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    tmp = tempfile.mkdtemp(prefix='.obj3d_preview_', dir=out)
    try:
        kfile = os.path.join(tmp, 'camera.json')
        with open(kfile, 'w', encoding='utf-8') as f:
            json.dump(K, f)
        for x, y in poses[:8]:
            item = os.path.join(tmp, 'render')
            os.makedirs(item, exist_ok=True)
            cmd = [BLENDER, '--background', '--python-exit-code', '7', '--python', RENDERER, '--',
                   '--glb', asset, '--bg', img_path, '--k', kfile, '--out', item,
                   '--lwh', '%.5f,%.5f,%.5f' % (length, width, height),
                   '--pose', '%.6f,%.6f,1.5' % (x, y), '--yaw', '180',
                   '--ground-z', str(ground_z), '--samples', '16',
                   '--hdri', '1' if p.get('hdri_enabled', False) else '0',
                   '--paint-coat', '1' if p.get('paint_coat_enabled', False) else '0']
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if r.returncode:
                raise RuntimeError('Blender 渲染失败: ' + (r.stderr or r.stdout)[-600:])
            fg = cv2.imread(os.path.join(item, 'fg_rgba.png'), cv2.IMREAD_UNCHANGED)
            sh = cv2.imread(os.path.join(item, 'vehicle_with_shadow.png'), cv2.IMREAD_UNCHANGED)
            if fg is None or sh is None or fg.shape != (h, w, 4) or sh.shape != fg.shape:
                raise RuntimeError('Blender 输出尺寸或透明通道无效')
            vehicle = fg[:, :, 3] > 16
            if vehicle.sum() < 100:
                continue
            # An image-space box is conservative: reject rather than guess occlusion depth.
            if np.mean(blocked[vehicle] > 0) > .01:
                continue
            sh[:, :, 3][vehicle] = fg[:, :, 3][vehicle]
            sh[:, :, 3][~vehicle] = (sh[:, :, 3][~vehicle].astype(np.float32) *
                                     np.clip(float(p.get('shadow', .35)), 0, 1)).astype(np.uint8)
            sh[:, :, 3][(~vehicle) & (blocked > 0)] = 0
            harmonization={'applied':False}
            if p.get('harmonize',True):
                from obj_harmonize import ForegroundHarmonizer
                sh, harmonization=ForegroundHarmonizer().apply(bg, sh, vehicle & (blocked == 0), blocked)
            alpha = sh[:, :, 3:4].astype(np.float32) / 255.0
            result = np.clip(sh[:, :, :3] * alpha + bg.astype(np.float32) * (1-alpha), 0, 255).astype(np.uint8)
            target = os.path.join(out, stem + '_obj.jpg')
            temp_out = os.path.join(tmp, 'result.jpg')
            if not cv2.imwrite(temp_out, result, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError('保存预览失败')
            metadata = {'preview_only': True, 'training_eligible': False, 'frame': stem,
                        'model': asset, 'asset_id': p.get('asset_id'), 'dimensions_source': 'nominal_unverified' if p['cls'].startswith('OBS/') else 'canonical_index', 'model_source_license': 'unverified',
                        'pose_body_m': [x, y, ground_z], 'yaw_deg': 180,
                        'road_mask': 'estimated_hsv', 'ground_z_note': 'visual adjustment, not surveyed',
                        'camera_source_xml': xml_name, 'lens_distortion_applied': False,
                        'occlusion': 'conservative_real_bbox_rejection', 'temporal_consistency': 'not established'}
            metadata['appearance']={'harmonization':harmonization,'hdri_enabled':bool(p.get('hdri_enabled',False)),
                                    'paint_coat_enabled':bool(p.get('paint_coat_enabled',False))}
            ys, xs = np.nonzero(vehicle)
            visible_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            with open(os.path.join(item, 'meta.json'), encoding='utf-8') as stream:
                rendered = json.load(stream)
            corners_raw = rendered.get('box3d_corners', [])
            ordered = [0, 2, 6, 4, 1, 3, 7, 5]
            corners = [corners_raw[i]['pix'] for i in ordered] if len(corners_raw) == 8 else []
            if len(corners) != 8 or not np.isfinite(np.asarray(corners)).all() or any(c.get('dist', -1) <= 0 for c in corners_raw):
                corners = []
            if corners:
                a = np.asarray(corners)
                if not (a[:, 0].min() - 3 <= visible_box[0] and a[:, 1].min() - 3 <= visible_box[1]
                        and a[:, 0].max() + 3 >= visible_box[2] and a[:, 1].max() + 3 >= visible_box[3]):
                    corners = []
            metadata['objects'] = [{'source': 'synthetic', 'class': p['cls'][4:] if p['cls'].startswith('OBS/') else p['cls'],
                'track_id': 'synthetic_1', 'bbox2d': visible_box, 'corners': corners,
                'box3d': {'x': x, 'y': y, 'z': ground_z, 'yaw_deg': 180,
                          'dimensions_m': rendered.get('dimensions_m')},
                'bbox3d_note': 'axis_aligned_enclosing_box' if corners else 'unavailable'}]
            with open(os.path.join(tmp, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            os.replace(temp_out, target)
            os.replace(os.path.join(tmp, 'meta.json'), os.path.join(out, stem + '_obj3d_meta.json'))
            print('写实 3D 单帧预览完成: %s pose=(%.2f, %.2f), 仅供预览' % (stem, x, y))
            return
        raise ValueError('候选位置与真实目标遮挡冲突，请调整距离或横向范围')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(config_path):
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)
    p = cfg.get('params') or {}
    if p.get('render_scope') != 'clip':
        return render_frame(config_path)
    if p.get('cls') not in ('car', 'suv', 'pickup', 'van'):
        raise ValueError('全片段目前只支持车辆；静态障碍物需要可靠的道路运动估计')
    clip_dir = _clip(cfg.get('batch', ''), cfg.get('clip', ''))
    frames = list_clip_frames(clip_dir)
    if not frames:
        raise ValueError('片段没有帧')
    payload = os.path.basename(config_path)
    if not payload.startswith('_p_obj_') or not payload.endswith('.json'):
        raise ValueError('任务标识无效')
    job = 'obj3d_' + payload[3:-5]
    if not job.replace('_', '').isalnum():
        raise ValueError('任务标识无效')
    out = os.path.join('/data/render_hybrid', job)
    os.makedirs(os.path.join(out, 'annotations'), exist_ok=False)
    os.symlink(out, os.path.join('/data/render', job), target_is_directory=True)
    first_stem = str(p.get('preview_frame_stem') or '')
    if first_stem not in [os.path.splitext(os.path.basename(img))[0] for img, _ in frames]:
        raise ValueError('所选参考帧不在片段中')
    sequence = sorted(frames, key=lambda pair: os.path.splitext(os.path.basename(pair[0]))[0] != first_stem)
    pose = None
    records = []
    for image, xml in sequence:
        stem = os.path.splitext(os.path.basename(image))[0]
        fp = dict(p, render_scope='frame', preview_frame_stem=stem)
        if pose is not None:
            fp['manual_pos'] = [[pose[0], -pose[1]]]
            fp['pos_on'] = False
        temp_cfg = os.path.join(out, '_frame_config.json')
        with open(temp_cfg, 'w', encoding='utf-8') as f:
            json.dump(dict(cfg, out=out, params=fp), f, ensure_ascii=False)
        try:
            render_frame(temp_cfg)
            with open(os.path.join(out, stem + '_obj3d_meta.json'), encoding='utf-8') as f:
                ann = json.load(f)
            if pose is None:
                pose = ann['pose_body_m']
            ann['temporal_consistency'] = 'fixed_body_relative_pose_assumption'
            ann['frame_status'] = 'synthetic'
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            if pose is None:
                raise ValueError('参考帧渲染失败：' + str(exc))
            if not cv2.imwrite(os.path.join(out, stem + '_obj.jpg'), cv2.imread(image)):
                raise RuntimeError('无法保存回退帧')
            ann = {'frame': stem, 'objects': [], 'frame_status': 'original_fallback',
                   'reason': str(exc), 'training_eligible': False}
        with open(os.path.join(out, 'annotations', stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump(ann, f, ensure_ascii=False, indent=2, allow_nan=False)
        records.append({'frame': stem, 'status': ann['frame_status']})
        print('clip frame %d/%d %s %s' % (len(records), len(frames), stem, ann['frame_status']), flush=True)
    manifest = {'job': job, 'batch': cfg['batch'], 'clip': cfg['clip'],
                'total': len(frames), 'synthetic': sum(r['status']=='synthetic' for r in records),
                'motion_assumption': 'fixed_body_relative_pose', 'training_eligible': False,
                'frames': sorted(records, key=lambda r: r['frame'])}
    for name in (os.path.join(out, 'manifest.json'), os.path.join(out, 'annotations', '__clip__.json')):
        with open(name, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--config':
        raise SystemExit('usage: obj_real3d_preview.py --config path')
    try:
        main(sys.argv[2])
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
