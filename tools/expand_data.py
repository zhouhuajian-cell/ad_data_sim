#!/usr/bin/env python3
"""从网盘扩充训练数据（只读源，写本地）。

护栏：data_guard.guard_all_mnt_ops()
"""
import os
import shutil
import sys

sys.path.insert(0, "/opt/ue5_synth/tools")
import data_guard
from data_guard import assert_writable, guard_all_mnt_ops, is_mnt

guard_all_mnt_ops()

REMOTE = "/mnt/done_data_new/18_bev-obj"
LOCAL = "/data/test_data/real_clips_extra"


def find_approved_batches(root, limit=200):
    """找已终审的批次。"""
    out = []
    for veh in sorted(os.listdir(root)):
        vp = os.path.join(root, veh)
        if not os.path.isdir(vp):
            continue
        for region in sorted(os.listdir(vp)):
            rp = os.path.join(vp, region)
            if not os.path.isdir(rp):
                continue
            ap = os.path.join(rp, "已终审")
            if not os.path.isdir(ap):
                continue
            for car in sorted(os.listdir(ap)):
                cp = os.path.join(ap, car)
                if not os.path.isdir(cp):
                    continue
                for b in sorted(os.listdir(cp)):
                    bp = os.path.join(cp, b)
                    if not os.path.isdir(bp):
                        continue
                    # 检查有 clip
                    has = False
                    for c in os.listdir(bp):
                        if os.path.isdir(os.path.join(bp, c, "cam_st_x2f_front")):
                            has = True
                            break
                    if has:
                        out.append({
                            "batch": b, "path": bp,
                            "vehicle": veh, "region": region, "car": car,
                        })
                    if len(out) >= limit:
                        return out
    return out


def estimate_size(batch_path, max_clips=2, max_frames=30):
    """估算复制大小。"""
    total = 0
    n_jpg = 0
    for c in sorted(os.listdir(batch_path))[:max_clips]:
        cd = os.path.join(batch_path, c, "cam_st_x2f_front")
        if not os.path.isdir(cd):
            continue
        for f in sorted(os.listdir(cd))[:max_frames]:
            if f.lower().endswith(".jpg"):
                try:
                    total += os.path.getsize(os.path.join(cd, f))
                    n_jpg += 1
                except OSError:
                    pass
    return total, n_jpg


def copy_batch(batch_path, name, dst_root, max_clips=2, max_frames=30):
    """复制一个批次的子集（只读源）。"""
    assert_writable(dst_root)
    dst = os.path.join(dst_root, name)
    if os.path.isdir(dst):
        return False, "已存在"

    n_img = 0
    for c in sorted(os.listdir(batch_path))[:max_clips]:
        src_cd = os.path.join(batch_path, c, "cam_st_x2f_front")
        if not os.path.isdir(src_cd):
            continue

        # 只复制前 N 帧
        jpgs = sorted(f for f in os.listdir(src_cd) if f.lower().endswith(".jpg"))
        jpgs = jpgs[:max_frames]

        dst_cd = os.path.join(dst, c, "cam_st_x2f_front")
        os.makedirs(dst_cd, exist_ok=True)
        for f in jpgs:
            shutil.copy2(os.path.join(src_cd, f), os.path.join(dst_cd, f))
            n_img += 1

        # 对应的 xml
        for lab in os.listdir(batch_path) if False else []:
            pass
        for lab_name in os.listdir(os.path.join(batch_path, c)):
            if lab_name.startswith("label_lidar") and "align" not in lab_name:
                src_ld = os.path.join(batch_path, c, lab_name)
                if not os.path.isdir(src_ld):
                    continue
                dst_ld = os.path.join(dst, c, lab_name)
                os.makedirs(dst_ld, exist_ok=True)
                for f in jpgs:
                    x = os.path.splitext(f)[0] + ".xml"
                    sp = os.path.join(src_ld, x)
                    if os.path.isfile(sp):
                        shutil.copy2(sp, os.path.join(dst_ld, x))
                break

    return True, "%d 帧" % n_img


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max-gb", type=float, default=20.0)
    a = ap.parse_args()

    print("=== 扫描网盘批次（只读）===")
    batches = find_approved_batches(REMOTE, limit=200)
    print("已终审批次: %d" % len(batches))

    if not batches:
        raise SystemExit("无批次")

    # 去重（batch 名）
    seen = set()
    uniq = []
    for b in batches:
        if b["batch"] in seen:
            continue
        seen.add(b["batch"])
        uniq.append(b)
    print("去重后: %d" % len(uniq))

    # 按大小排序，挑小的先复制
    print("\n=== 估算大小 ===")
    sized = []
    for b in uniq[:a.limit]:
        sz, nj = estimate_size(b["path"])
        sized.append((sz, nj, b))
    sized.sort(key=lambda x: x[0])

    total_bytes = 0
    picked = []
    for sz, nj, b in sized:
        if total_bytes + sz > a.max_gb * 1e9:
            break
        picked.append(b)
        total_bytes += sz

    print("选中: %d 个批次，约 %.1f GB / %.0f 帧"
          % (len(picked), total_bytes / 1e9,
             sum(estimate_size(b["path"])[1] for b in picked)))

    print("\n=== 复制（只读源）===")
    os.makedirs(LOCAL, exist_ok=True)
    ok = 0
    for i, b in enumerate(picked):
        try:
            done, msg = copy_batch(b["path"], b["batch"], LOCAL,
                                   max_clips=2, max_frames=30)
            if done:
                ok += 1
                if (i + 1) % 5 == 0:
                    print("  [%d/%d] %s %s" % (i + 1, len(picked),
                                               b["batch"][:40], msg))
        except Exception as e:
            print("  失败 %s: %s" % (b["batch"][:30], str(e)[:60]))

    print("\n完成: %d/%d 批次 → %s" % (ok, len(picked), LOCAL))

    # 统计
    tot = 0
    for dp, _, fs in os.walk(LOCAL):
        tot += len([f for f in fs if f.endswith(".jpg")])
    print("本地新增帧: %d" % tot)


if __name__ == "__main__":
    main()
