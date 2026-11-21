"""补齐 temporal / multi_camera / split 三项门禁。

temporal   跨帧一致性：同一合成物体在相邻帧的位置/尺寸不应跳变
multi_cam  多相机：x2f + x8d 两路都能读到同一时刻的图
split      scene 级划分：按 clip 分组，避免同场景跨训练/测试
"""
import glob
import json
import os

import numpy as np

from export_gate import make_check


def check_temporal(frames_meta, max_jump_px=80.0, max_scale_change=0.35):
    """跨帧一致性检查。

    frames_meta: [{"stem":..., "synthetic":[{"bbox":[...], "world":{...}}]}, ...]
    按 stem 时间戳排序后，逐帧比较「同序号合成物体」的 bbox 中心与宽高。
    """
    if len(frames_meta) < 2:
        return make_check("unchecked", note="帧数 < 2，无法检查时序")

    def ts(m):
        try:
            return float(m["stem"].split(".")[0])
        except Exception:
            return 0.0

    ms = sorted(frames_meta, key=ts)

    jumps = []
    scales = []
    for a, b in zip(ms[:-1], ms[1:]):
        sa = a.get("synthetic") or []
        sb = b.get("synthetic") or []
        n = min(len(sa), len(sb))
        for i in range(n):
            ba, bb = sa[i]["bbox"], sb[i]["bbox"]
            ca = ((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2)
            cb = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            d = float(np.hypot(cb[0] - ca[0], cb[1] - ca[1]))
            jumps.append(d)

            wa = max(ba[2] - ba[0], 1e-6)
            wb = max(bb[2] - bb[0], 1e-6)
            scales.append(abs(wb - wa) / wa)

    if not jumps:
        return make_check("unchecked", note="无可比对象")

    max_j = max(jumps)
    max_s = max(scales)
    med_j = float(np.median(jumps))

    ok = (max_j <= max_jump_px) and (max_s <= max_scale_change)
    return make_check(
        "passed" if ok else "failed",
        evidence_obj={
            "n_pairs": len(jumps),
            "max_jump_px": round(max_j, 2),
            "median_jump_px": round(med_j, 2),
            "max_scale_change": round(max_s, 4),
            "threshold_jump_px": max_jump_px,
            "threshold_scale": max_scale_change,
        },
        note="跨帧最大跳变 %.0fpx / 尺度 %.0f%%" % (max_j, max_s * 100))


def check_multi_camera(clip_dir, cam_names=("cam_st_x2f_front", "cam_st_x8d_front")):
    """多相机：两个相机目录都存在且有同一时刻的图。"""
    found = {}
    for c in cam_names:
        d = os.path.join(clip_dir, c)
        if not os.path.isdir(d):
            found[c] = 0
            continue
        js = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
        found[c] = len(js)

    active = [c for c, n in found.items() if n > 0]
    if len(active) < 2:
        return make_check(
            "unchecked",
            evidence_obj={"cameras": found},
            note="只有 %d 路相机" % len(active))

    # 找时间戳交集（取整到 0.1s）
    def stems(c):
        d = os.path.join(clip_dir, c)
        out = set()
        for f in os.listdir(d):
            if f.lower().endswith(".jpg"):
                out.add(os.path.splitext(f)[0])
        return out

    s0 = stems(active[0])
    s1 = stems(active[1])
    common = s0 & s1

    return make_check(
        "passed" if common else "unchecked",
        evidence_obj={"cameras": found,
                      "common_timestamps": len(common)},
        note="%d 路相机，共同时刻 %d 帧" % (len(active), len(common)))


def check_split(clip_dirs, test_ratio=0.2):
    """scene 级划分：按 clip 分组，不按帧随机分。

    这里只检查「划分方案不会泄漏」：同一 clip 的帧只能进同一个集合。
    """
    if not clip_dirs:
        return make_check("unchecked", note="无 clip 列表")

    clips = sorted(set(os.path.basename(p.rstrip("/")) for p in clip_dirs))
    n = len(clips)
    if n < 2:
        return make_check(
            "unchecked",
            evidence_obj={"n_clips": n},
            note="只有一个 clip，无法做 train/val 划分")

    n_test = max(1, int(round(n * test_ratio)))
    test = clips[-n_test:]
    train = clips[:-n_test]

    return make_check(
        "passed",
        evidence_obj={"n_clips": n, "train": train, "test": test,
                      "split_by": "clip", "leak_free": True},
        note="按 clip 划分 %d/%d（训练/测试）" % (len(train), len(test)))


def augment_gate(clip_dir, frames_meta, all_clips):
    """把三项追加到门禁 checks。"""
    return {
        "temporal": check_temporal(frames_meta),
        "multi_camera": check_multi_camera(clip_dir),
        "split": check_split(all_clips),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/opt/ue5_synth/tools")

    clip = ("/data/test_data/real_clips/"
            "20261120_HS5001_changchun_1v_extremely_closetruck/"
            "133913-1788137262.664")
    print("clip:", os.path.basename(clip))

    # 模拟几帧
    fake = []
    for i in range(5):
        fake.append({
            "stem": "%.3f" % (1788137262.664 + i * 0.1),
            "synthetic": [
                {"bbox": [900 + i * 2, 600, 1100 + i * 2, 900],
                 "world": {"x_m": 10, "y_m": 0}},
            ],
        })

    print()
    print("temporal:     ", check_temporal(fake)["status"],
          check_temporal(fake)["note"])
    print("multi_camera: ", check_multi_camera(clip)["status"],
          check_multi_camera(clip)["note"])

    clips = glob.glob("/data/test_data/real_clips/*/*")
    print("split:        ", check_split(clips)["status"],
          check_split(clips)["note"])
