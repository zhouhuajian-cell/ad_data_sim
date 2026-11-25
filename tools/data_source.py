"""数据源切换：网盘只读 / 本地副本。

默认优先用本地副本（/data/test_data/real_clips），
找不到再退回网盘（只读）。

网盘路径永远不写。
"""
import os

# 网盘源（只读）
REMOTE_ROOT = ("/mnt/done_data_new/18_bev-obj/00_M57/"
               "00_daizhongshen_changchun/已终审/hs5001")

# 本地副本
LOCAL_ROOT = "/data/test_data/real_clips"

# 已知的测试包（按"适合放合成物体"排序）
TEST_CLIPS = [
    "20261120_HS5001_changchun_1v_extremely_closetruck",  # 前方 22.7m 才有一辆车
    "20261120_HS5001_changchun_1v_ground_text",           # 地面文字场景
    "20261120_HS5001_changchun_1v_closecar",              # 密集场景（压力测试）
    "20261120_HS5001_changchun_1v_night_specialcar",      # 夜间
]


def resolve_root(prefer_local=True):
    """返回可用的数据根目录。"""
    if prefer_local and os.path.isdir(LOCAL_ROOT):
        return LOCAL_ROOT, "local"
    if os.path.isdir(REMOTE_ROOT):
        return REMOTE_ROOT, "remote"
    return None, None


def resolve_batch(name, prefer_local=True):
    """返回某个批次的绝对路径。"""
    root, kind = resolve_root(prefer_local)
    if root is None:
        return None, None
    p = os.path.join(root, name)
    if os.path.isdir(p):
        return p, kind
    # 本地没有就回退网盘
    p2 = os.path.join(REMOTE_ROOT, name)
    if prefer_local and os.path.isdir(p2):
        return p2, "remote"
    return None, None


def list_clips(prefer_local=True):
    """DYNAMIC-CLIPS: 动态扫描所有批次（含 extra）。"""
    out = []
    seen = set()
    roots = ["/data/test_data/real_clips",
             "/data/test_data/real_clips_extra"]
    for r in roots:
        if not os.path.isdir(r):
            continue
        kind = "local"
        for n in sorted(os.listdir(r)):
            p = os.path.join(r, n)
            if not os.path.isdir(p) or n in seen:
                continue
            # 必须至少有一个含 cam_st_x2f_front 的子目录
            try:
                subs = os.listdir(p)
            except OSError:
                continue
            has = any(os.path.isdir(os.path.join(p, sd, cam))
                      for sd in subs for cam in ("cam_st_x2f_front", "cam_st_x8d_front", "cam_ld_x8d_front", "cam_hk_x8b_front"))
            if has:
                out.append((n, p, kind))
                seen.add(n)
    return out


def assert_writable(path):
    """确认路径可写（防止误写网盘）。"""
    p = os.path.abspath(path)
    if p.startswith("/mnt/"):
        raise PermissionError("禁止写入网盘: %s" % p)
    return True


if __name__ == "__main__":
    root, kind = resolve_root()
    print("数据根:", root, "(%s)" % kind)
    print()
    for n, p, k in list_clips():
        sz = 0
        for dp, _, fs in os.walk(p):
            for f in fs:
                try:
                    sz += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        print("  %-52s %6.0f MB  [%s]" % (n[:52], sz / 1e6, k))
    print()
    try:
        assert_writable("/data/render/x")
        print("写入检查: /data OK")
    except Exception as e:
        print("写入检查失败:", e)
    try:
        assert_writable("/mnt/done_data_new/x")
        print("!! 网盘写保护失效")
    except PermissionError:
        print("网盘写保护: 生效 ✅")
