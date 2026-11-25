#!/usr/bin/env python3
"""Evaluate road-plane motion hypotheses; never produces training labels.

Uses the existing GND calibration and frame masks. Low-confidence frames are
rejected; lane segmentation is diagnostic data, not independent ground truth.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


def load_frame(path, clip_dir, width=960):
    image = cv2.imread(path)
    if image is None:
        raise ValueError('cannot read ' + path)
    h, w = image.shape[:2]
    height = round(h * width / w)
    gray = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    stem = os.path.splitext(os.path.basename(path))[0]
    def read(folder):
        p = os.path.join(clip_dir, folder, stem + '.png')
        return cv2.imread(p, cv2.IMREAD_UNCHANGED) if os.path.isfile(p) else None
    road, obj, lane = (read(x) for x in
                       ('freespace_mask_ori_3.0.0', 'obj_mask_ori_3.1.1', 'seg_lane'))
    if road is None or obj is None:
        raise ValueError('missing road/object mask for ' + stem)
    road = np.any(road != 0, axis=2) if road.ndim == 3 else road != 0
    obj = np.any(obj != 0, axis=2) if obj.ndim == 3 else obj != 0
    mask = cv2.resize(road.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    obj = cv2.resize(obj.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    mask[cv2.dilate(obj, np.ones((15, 15), np.uint8)) != 0] = 0
    mask[:int(height * .57)] = 0
    mask[int(height * .94):] = 0
    if lane is not None:
        lane = np.any(lane != 0, axis=2) if lane.ndim == 3 else lane != 0
        lane = cv2.resize(lane.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return gray, mask * 255, lane, (w, h)


def lane_score(previous, current, transform, mask0, mask1):
    if previous is None or current is None:
        return None
    h, w = previous.shape
    warped = cv2.warpPerspective(previous, transform, (w, h), flags=cv2.INTER_NEAREST) > 0
    valid = cv2.warpPerspective(mask0, transform, (w, h), flags=cv2.INTER_NEAREST) > 0
    valid &= mask1 > 0
    a, b = warped & valid, (current > 0) & valid
    if a.sum() < 100 or b.sum() < 100:
        return None
    distance = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)[a]
    return {'support': int(a.sum()), 'within_8px': float(np.mean(distance <= 4)),
            'median_px': float(np.median(distance) * 2)}


def estimate(prev, curr, mask0, mask1, ground_to_image):
    points = cv2.goodFeaturesToTrack(prev, 1600, .005, 5, mask=mask0, blockSize=5)
    if points is None or len(points) < 50:
        return None, {'reason': 'too_few_features'}
    target, ok1, _ = cv2.calcOpticalFlowPyrLK(prev, curr, points, None, winSize=(21, 21), maxLevel=3)
    if target is None:
        return None, {'reason': 'optical_flow_failed'}
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, target, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        return None, {'reason': 'backward_flow_failed'}
    ok = (ok1[:, 0] > 0) & (ok2[:, 0] > 0)
    ok &= np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 1.0
    xy = np.rint(target.reshape(-1, 2)).astype(int)
    h, w = prev.shape
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    in_road = np.zeros(len(points), dtype=bool)
    in_road[inside] = mask1[xy[inside, 1], xy[inside, 0]] > 0
    ok &= in_road
    a, b = points.reshape(-1, 2)[ok], target.reshape(-1, 2)[ok]
    if len(a) < 50:
        return None, {'reason': 'too_few_tracks', 'tracks': int(len(a))}
    inv = np.linalg.inv(ground_to_image)
    ag = cv2.perspectiveTransform(a.reshape(-1, 1, 2), inv).reshape(-1, 2)
    bg = cv2.perspectiveTransform(b.reshape(-1, 1, 2), inv).reshape(-1, 2)
    valid = np.isfinite(ag).all(axis=1) & np.isfinite(bg).all(axis=1)
    for p in (ag, bg):
        valid &= (p[:, 0] > 3) & (p[:, 0] < 60) & (np.abs(p[:, 1]) < 15)
    ag, bg = ag[valid], bg[valid]
    if len(ag) < 50:
        return None, {'reason': 'too_few_ground_tracks', 'tracks': int(len(ag))}
    affine, inliers = cv2.estimateAffinePartial2D(ag, bg, method=cv2.RANSAC,
        ransacReprojThreshold=.20, maxIters=3000, confidence=.995)
    if affine is None or inliers is None:
        return None, {'reason': 'ground_fit_failed'}
    keep = inliers[:, 0] > 0
    if keep.sum() < 40 or keep.mean() < .35:
        return None, {'reason': 'weak_ground_fit', 'inliers': int(keep.sum()), 'tracks': int(len(ag))}
    x, y = ag[keep], bg[keep]
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    u, _, vt = np.linalg.svd((x-xc).T @ (y-yc))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    translation = yc - rot @ xc
    residual = np.linalg.norm((x @ rot.T + translation) - y, axis=1)
    A = np.eye(3); A[:2, :2] = rot; A[:2, 2] = translation
    H = ground_to_image @ A @ inv
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-10:
        return None, {'reason': 'invalid_transform'}
    H /= H[2, 2]
    info = {'tracks': int(len(ag)), 'inliers': int(keep.sum()),
            'inlier_ratio': float(keep.mean()), 'median_residual_m': float(np.median(residual)),
            'translation_m': translation.tolist(),
            'yaw_deg': float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))}
    if np.linalg.norm(translation) > 5 or abs(info['yaw_deg']) > 10:
        info['reason'] = 'implausible_motion'
        return None, info
    return H, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True); ap.add_argument('--clip', required=True)
    ap.add_argument('--output', required=True); args = ap.parse_args()
    sys.path.insert(0, '/opt/ue5_synth/tools')
    from gnd_calib import load_gnd_calib
    base = os.path.join('/data/test_data/gnd_train', args.batch, args.clip)
    files = sorted(glob.glob(os.path.join(base, 'img_ori', '*.jpg')))
    if not files:
        raise SystemExit('no frames')
    cal = load_gnd_calib(args.batch, args.clip)
    if cal is None:
        raise SystemExit('missing calibration')
    first = load_frame(files[0], base)
    K = np.asarray(cal['K'], dtype=float).copy()
    K[0, :] *= first[0].shape[1] / first[3][0]
    K[1, :] *= first[0].shape[0] / first[3][1]
    R = np.asarray(cal['R_cam2body'], dtype=float).T
    t = np.asarray(cal['t_cam2body'], dtype=float)
    G = K @ np.column_stack((R[:, 0], R[:, 1], -R @ t))
    if not np.isfinite(G).all() or abs(np.linalg.det(G)) < 1e-9:
        raise SystemExit('invalid ground projection')
    records = []; previous = first
    cumulative = np.eye(3)
    cumulative_valid = True
    for index, file in enumerate(files[1:], 1):
        try:
            current = load_frame(file, base)
            H, info = estimate(previous[0], current[0], previous[1], current[1], G)
            if H is not None:
                base_score = lane_score(previous[2], current[2], np.eye(3), previous[1], current[1])
                after = lane_score(previous[2], current[2], H, previous[1], current[1])
                info.update({'baseline_lane': base_score, 'estimated_lane': after})
                # Segmentation is only a fail-closed plausibility gate, not ground truth.
                if after is None:
                    info['reason'] = 'unverifiable_lane_support'
                elif after['within_8px'] < .60:
                    info['reason'] = 'lane_alignment_low'
                elif base_score and after['within_8px'] + .05 < base_score['within_8px']:
                    info['reason'] = 'worse_than_static'
                else:
                    info['reason'] = 'preview_candidate_only'
                    info['homography_960'] = H.tolist()
                    if cumulative_valid:
                        cumulative = H @ cumulative
                        cumulative /= cumulative[2, 2]
                        anchor = lane_score(first[2], current[2], cumulative,
                                            first[1], current[1])
                        info['anchor_lane'] = anchor
                        if anchor is None or anchor['within_8px'] < .60:
                            info['reason'] = 'cumulative_alignment_unverified'
                            cumulative_valid = False
                    else:
                        info['reason'] = 'cumulative_chain_broken'
            if info['reason'] != 'preview_candidate_only':
                cumulative_valid = False
            records.append({'frame': index, 'stem': os.path.basename(file), **info})
            previous = current
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            records.append({'frame': index, 'stem': os.path.basename(file), 'reason': str(exc)})
            # Do not silently bridge over missing or corrupt frames.
            break
    result = {'batch': args.batch, 'clip': args.clip, 'frames': len(files),
              'method': 'road_lk_calibrated_se2', 'training_eligible': False,
              'limitations': ['planar road approximation', 'lens distortion ignored',
                              'segmentation is not independent ground truth',
                              'pairwise estimates are not world poses'], 'pairs': records}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'clip': args.clip, 'pairs': len(records), 'candidates':
                      sum(x['reason'] == 'preview_candidate_only' for x in records),
                      'output': args.output}))


if __name__ == '__main__':
    main()
