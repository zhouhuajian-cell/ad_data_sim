# -*- coding: utf-8 -*-
# case_check.py — 跨平台两校验（方案 4.2 下发前必跑）
# 干什么：① uproject 声明的模块名 == Source/ 下实际模块目录名；
#         ② Content 内被 uasset/umap 引用的 /Game/ 路径大小写 == 磁盘实际文件名大小写。
# 怎么跑：python case_check.py <工程根目录>   （Windows/Linux 都能跑）
# 成功判据：退出码 0 且输出 "ALL PASS"；有不一致会逐条列出（大小写是 Ubuntu 加载失败的最高频原因）。
import json
import os
import re
import sys

# uasset 是二进制，引用路径以 ASCII 和 UTF-16LE 两种编码藏在字符串区，两种都扫
_RE_ASCII = re.compile(rb"/Game/[A-Za-z0-9_/\-]{2,200}")
_RE_UTF16 = re.compile(rb"(?:/\x00G\x00a\x00m\x00e\x00(?:[\x20-\x7e]\x00){2,200})")
_SKIP_DIRS = ("Intermediate", "Saved", "Binaries", "DerivedDataCache", ".vs", "Build")


def real_path_map(content_dir):
    """磁盘实际大小写 → 小写键。Windows 不区分大小写，必须用目录枚举拿到真实大小写。"""
    m = {}
    for root, dirs, files in os.walk(content_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, content_dir).replace("\\", "/")
            m[rel.lower()] = rel
    return m


def check_uproject_modules(root):
    errs = []
    up = [f for f in os.listdir(root) if f.endswith(".uproject")]
    if not up:
        return ["找不到 .uproject"]
    with open(os.path.join(root, up[0]), "r", encoding="utf-8") as f:
        mods = [m["Name"] for m in (json.load(f).get("Modules") or [])]
    src = os.path.join(root, "Source")
    dirs = [d for d in (os.listdir(src) if os.path.isdir(src) else [])
            if os.path.isdir(os.path.join(src, d))]
    for m in mods:
        if m not in dirs:
            errs.append("uproject 声明模块 %r 不在 Source/ 下（实际: %s）" % (m, dirs))
    for d in dirs:
        if d not in mods and not any(d.startswith(m) for m in mods):
            errs.append("Source/%s 未在 uproject Modules 声明（Editor 模块可忽略）" % d)
    return errs


def check_content_case(root):
    content = os.path.join(root, "Content")
    if not os.path.isdir(content):
        return ["找不到 Content/"]
    rmap = real_path_map(content)
    errs, seen = [], set()
    for sub, _dirs, files in os.walk(content):
        for fn in files:
            if not fn.endswith((".uasset", ".umap")):
                continue
            path = os.path.join(sub, fn)
            try:
                blob = open(path, "rb").read()
            except OSError:
                continue
            refs = set(m.decode("ascii", "ignore") for m in _RE_ASCII.findall(blob))
            for m in _RE_UTF16.findall(blob):
                refs.add(m.decode("utf-16-le", "ignore"))
            for ref in refs:
                ref = ref.rstrip("\x00")
                if ref in seen:
                    continue
                seen.add(ref)
                # /Game/Foo/Bar → Content/Foo/Bar（uasset 不带扩展名，两种都试）
                rel = ref[len("/Game/"):]
                if not rel or rel.startswith("."):
                    continue
                hit = rmap.get(rel.lower()) or rmap.get((rel + ".uasset").lower()) \
                    or rmap.get((rel + ".umap").lower())
                if hit is None:
                    continue        # 引用了不存在的资产 → 加载时报缺，属另一类问题，这里只管大小写
                want = rel if rel.endswith((".uasset", ".umap")) else rel + \
                    os.path.splitext(hit)[1]
                if want != hit:
                    errs.append("大小写不一致: 引用 %s  实际 %s" % (want, hit))
    return errs


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "用法: python case_check.py <工程根目录>")
        return 2
    root = sys.argv[1]
    errs = check_uproject_modules(root) + check_content_case(root)
    for e in errs:
        print("FAIL %s" % e)
    if errs:
        print("%d 处不一致（修完再下发）" % len(errs))
        return 1
    print("ALL PASS：模块名与 Content 大小写均一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
