"""Conservative OBJ sprite placement in a calibrated road image.

Road masks may be estimated. The returned polygons are screening geometry,
not verified BEV labels or proof that a vehicle is physically unobstructed.
"""
import cv2
import numpy as np


def crop_sprite_alpha(sprite, threshold=8, margin=2):
    if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
        raise ValueError('sprite must be BGRA')
    ys, xs = np.where(sprite[:, :, 3] > threshold)
    if not len(xs):
        raise ValueError('empty sprite alpha')
    h, w = sprite.shape[:2]
    x0, x1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + margin + 1)
    return sprite[y0:y1, x0:x1].copy()


def footprint_mask(cam, x, y, length, width, shape, projector):
    """Project a longitudinally aligned nominal footprint into the image."""
    h, w = shape[:2]
    points = [[x + length/2, y + width/2, 0],
              [x - length/2, y + width/2, 0],
              [x - length/2, y - width/2, 0],
              [x + length/2, y - width/2, 0]]
    uv, _, ok = projector(cam, points)
    if not np.all(ok) or not np.isfinite(uv).all():
        return None
    if not ((uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h)).all():
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(uv).astype(np.int32), 1)
    return mask if int(mask.sum()) >= 20 else None


def safe_footprint(footprint, road, blocked, min_road=.90, max_blocked=.01):
    if footprint is None:
        return False
    selected = footprint.astype(bool)
    if not selected.any() or road.shape != selected.shape or blocked.shape != selected.shape:
        return False
    return (float(np.mean(road[selected] > .5)) >= min_road and
            float(np.mean(blocked[selected] > .5)) <= max_blocked)


def sample_poses(cam, road, blocked, count, dmin, dmax, lateral, length, width,
                 rng, projector, attempts_per_object=80):
    if dmin <= 0 or dmax < dmin or lateral < 0 or length <= 0 or width <= 0:
        raise ValueError('invalid sampling parameters')
    poses = []
    for _ in range(count):
        for _ in range(attempts_per_object):
            x = rng.uniform(dmin, dmax)
            y = rng.uniform(-lateral, lateral)
            if any(abs(x-a) < length and abs(y-b) < width for a, b in poses):
                continue
            f = footprint_mask(cam, x, y, length, width, road.shape, projector)
            if safe_footprint(f, road, blocked):
                poses.append((x, y))
                break
    return sorted(poses, key=lambda p: -p[0])
