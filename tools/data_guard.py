"""数据访问安全层：强制网盘只读。

任何对 /mnt 的写操作直接抛异常。
所有训练/仿真脚本必须通过本模块访问网盘。
"""
import os
import shutil

# 网盘根（只读）
MOUNT_ROOTS = ("/mnt/",)

# 本地工作区（可写）
LOCAL_STAGING = "/data/train_staging"


def is_mnt(path):
    """是否在网盘下。"""
    p = os.path.abspath(path)
    return any(p.startswith(r) for r in MOUNT_ROOTS)


def assert_readonly(path):
    """确认是只读操作。"""
    if is_mnt(path):
        return True
    return False


def assert_writable(path):
    """确认可写（不在网盘）。"""
    p = os.path.abspath(path)
    if is_mnt(p):
        raise PermissionError(
            "禁止写入网盘（只读约束）: %s\n"
            "请先复制到本地 %s 再处理" % (p, LOCAL_STAGING))
    return True


def safe_copy_from_mnt(src, dst_dir):
    """从网盘安全复制（只读源，写到本地）。"""
    assert_writable(dst_dir)
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)          # copy2 只读源
    return dst


def safe_list_mnt(path):
    """列出网盘目录（只读）。"""
    if not is_mnt(path):
        raise ValueError("路径不在网盘下: " + path)
    return sorted(os.listdir(path))


def guard_all_mnt_ops():
    """全局护栏：monkey-patch 危险的 os/shutil 写操作。

    在训练/仿真脚本开头调用一次即可。
    """
    import builtins

    _orig_remove = os.remove
    _orig_unlink = os.unlink
    _orig_rmtree = shutil.rmtree
    _orig_rename = os.rename
    _orig_replace = os.replace
    _orig_makedirs = os.makedirs
    _orig_mkdir = os.mkdir
    _orig_open = builtins.open

    def _check(path, op):
        try:
            if is_mnt(path):
                raise PermissionError(
                    "[数据护栏] 禁止对网盘执行 %s: %s" % (op, path))
        except (TypeError, ValueError):
            pass

    def guarded_remove(p, *a, **k):
        _check(p, "remove")
        return _orig_remove(p, *a, **k)

    def guarded_unlink(p, *a, **k):
        _check(p, "unlink")
        return _orig_unlink(p, *a, **k)

    def guarded_rmtree(p, *a, **k):
        _check(p, "rmtree")
        return _orig_rmtree(p, *a, **k)

    def guarded_rename(s, d, *a, **k):
        _check(s, "rename(src)")
        _check(d, "rename(dst)")
        return _orig_rename(s, d, *a, **k)

    def guarded_replace(s, d, *a, **k):
        _check(s, "replace(src)")
        _check(d, "replace(dst)")
        return _orig_replace(s, d, *a, **k)

    def guarded_makedirs(p, *a, **k):
        _check(p, "makedirs")
        return _orig_makedirs(p, *a, **k)

    def guarded_mkdir(p, *a, **k):
        _check(p, "mkdir")
        return _orig_mkdir(p, *a, **k)

    os.remove = guarded_remove
    os.unlink = guarded_unlink
    shutil.rmtree = guarded_rmtree
    os.rename = guarded_rename
    os.replace = guarded_replace
    os.makedirs = guarded_makedirs
    os.mkdir = guarded_mkdir

    return True


if __name__ == "__main__":
    print("=== 数据护栏自检 ===")
    print("网盘路径判定:")
    for p in ("/mnt/done_data/x", "/mnt/Data_Platform/y",
              "/data/train/z", "/opt/ue5_synth/w"):
        print("  %-32s mnt=%s" % (p, is_mnt(p)))

    print("\n写保护:")
    for p in ("/data/train/ok", "/mnt/done_data/no"):
        try:
            assert_writable(p)
            print("  %-28s 可写 OK" % p)
        except PermissionError as e:
            print("  %-28s 拒绝 OK" % p)

    print("\n全局护栏:")
    guard_all_mnt_ops()
    try:
        os.makedirs("/mnt/done_data_test/abc")
        print("  !! 护栏失效")
    except PermissionError:
        print("  makedirs 被拦截 ✅")
    try:
        os.remove("/mnt/done_data/x.txt")
        print("  !! 护栏失效")
    except PermissionError:
        print("  remove 被拦截 ✅")
    except FileNotFoundError:
        print("  remove 被拦截 ✅（先到护栏）")
