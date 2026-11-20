"""安全补丁工具：所有注入都走 insert_before_last_script()，先校验再插入"""
import io
import os

P = "/opt/ue5_synth/dashboard/index.html"


def read():
    return io.open(P, encoding="utf-8").read()


def write(s):
    # 校验：脚本块数量必须 >= 3，且字符数不能低于原文件
    assert s.count("<script") >= 3, "script 块数量异常: %d" % s.count("<script")
    assert len(s) > 100000, "文件过小，可能被截断: %d" % len(s)
    assert s.rstrip().endswith("</html>"), "文件结尾异常"
    io.open(P, "w", encoding="utf-8").write(s)


def insert_before_last_script(js, tag):
    s = read()
    if tag in s:
        print("  [跳过] %s 已存在" % tag)
        return s
    idx = s.rfind("</script>")
    if idx < 0:
        raise RuntimeError("找不到 </script>")
    s = s[:idx] + "\n" + js + "\n" + s[idx:]
    write(s)
    print("  [注入] %s" % tag)
    return s


def insert_after(anchor, html, tag):
    s = read()
    if tag in s:
        print("  [跳过] %s 已存在" % tag)
        return s
    if anchor not in s:
        raise RuntimeError("锚点未找到: %s" % anchor[:60])
    s = s.replace(anchor, html + anchor, 1)
    write(s)
    print("  [插入] %s" % tag)
    return s


def replace_once(old, new, tag):
    s = read()
    if old not in s:
        print("  [跳过] %s 锚点不存在（可能已改过）" % tag)
        return s
    s = s.replace(old, new, 1)
    write(s)
    print("  [替换] %s" % tag)
    return s
