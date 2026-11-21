"""跨帧一致性：一个 clip 内合成物体的世界坐标保持不变。

原理：
  真实数据是连续帧，相机在动、世界不动。
  合成物体也应该在世界里固定，而不是每帧重新找空隙。

做法：
  1) 用 clip 的中间帧找空隙
  2) 把选中的世界坐标固定下来
  3) 所有帧都用这组坐标，只做可见性检查
"""
import numpy as np

from done_data_labels import project_body_to_pixel


def pick_stable_positions(clip_frames, loader, margin_m=1.2,
                          n_max=4, sample_every=5):
    """在整个 clip 上找「始终不与真实物体冲突」的世界坐标。

    clip_frames: [(img_path, xml_name), ...]
    loader:  callable(clip_dir, xml_name) -> labels
    返回 [(x, y), ...]（世界坐标，米）
    """
    from batch_real_labels import find_gaps

    # 抽若干帧
    samples = clip_frames[::max(sample_every, 1)][:8]
    if not samples:
        return []

    # 每帧的可用位置集合
    per_frame = []
    for img_p, xml_n in samples:
        labels = loader(xml_n)
        if not labels:
            continue
        cam = labels["camera"]
        # 用大图尺寸（从第一帧读）
        cands, _ = find_gaps(labels, cam, (1080, 1920), margin_m=margin_m)
        if not cands:
            per_frame.append(set())
            continue
        keys = set()
        for x, y, _ in cands:
            keys.add((round(x, 1), round(y, 1)))
        per_frame.append(keys)

    if not per_frame:
        return []

    # 取所有帧的交集
    common = per_frame[0]
    for s in per_frame[1:]:
        common &= s

    if not common:
        # 退一步：用出现次数最多的位置
        from collections import Counter
        cnt = Counter()
        for s in per_frame:
            for k in s:
                cnt[k] += 1
        need = max(1, len(per_frame) * 2 // 3)
        common = set(k for k, v in cnt.items() if v >= need)

    if not common:
        return []

    # 按距离排序，取最近的 n_max 个
    items = sorted(common, key=lambda k: (k[0], abs(k[1])))
    return items[:n_max]


def place_stable(clip_dir, clip_frames, all_positions):
    """生成每帧的合成物体（世界坐标固定）。"""
    out = []
    for img_p, xml_n in clip_frames:
        out.append({"img": img_p, "xml": xml_n, "positions": all_positions})
    return out


def verify_temporal(placed_list, cam_getter, img_size=(1920, 1080)):
    """验证跨帧位置是否稳定（应该完全稳定）。"""
    w, h = img_size
    prev = None
    jumps = []
    for item in placed_list:
        cam = cam_getter(item["xml"])
        if cam is None:
            continue
        cur = []
        for (x, y) in item["positions"]:
            uv, z, ok = project_body_to_pixel([[x, y, 0.0]], cam)
            if ok[0] and np.isfinite(uv[0]).all():
                cur.append((float(uv[0][0]), float(uv[0][1])))
            else:
                cur.append(None)
        if prev is not None:
            for a, b in zip(prev, cur):
                if a is None or b is None:
                    continue
                jumps.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
        prev = cur

    if not jumps:
        return {"max_jump_px": None, "n": 0}
    return {"max_jump_px": round(max(jumps), 2),
            "median_jump_px": round(float(np.median(jumps)), 2),
            "n": len(jumps)}
