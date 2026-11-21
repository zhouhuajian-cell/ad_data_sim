#!/usr/bin/env python3
"""下载 Kenney CC0 3D 模型（Car Kit / 交通锥等）。

Kenney 素材是 CC0，可从 GitHub 镜像或官方 CDN 获取。
"""
import json
import os
import subprocess
import sys
import zipfile

OUT = "/data/asset_src"
os.makedirs(OUT, exist_ok=True)

# 候选源（按优先级）
SOURCES = [
    # GitHub 上的 Kenney 镜像
    ("kenney_car", "https://github.com/KenneyNL/Vehicle-Pack/archive/refs/heads/master.zip"),
    ("kenney_car2", "https://codeload.github.com/KenneyNL/Vehicle-Pack/zip/refs/heads/master"),
    ("kenney_kit", "https://codeload.github.com/KenneyNL/Car-Kit/zip/refs/heads/main"),
    # 备选：开源车辆模型
    ("gltf_samples", "https://github.com/KhronosGroup/glTF-Sample-Models/archive/refs/heads/master.zip"),
]


def try_download(name, url):
    z = os.path.join(OUT, name + ".zip")
    if os.path.isfile(z) and os.path.getsize(z) > 10000:
        print("  已有: %s (%.1f MB)" % (name, os.path.getsize(z) / 1e6))
        return z
    print("  下载 %s ..." % url)
    r = subprocess.run(
        ["curl", "-sL", "--max-time", "180", "-o", z, url],
        capture_output=True, text=True)
    if os.path.isfile(z) and os.path.getsize(z) > 10000:
        print("    OK %.1f MB" % (os.path.getsize(z) / 1e6))
        return z
    if os.path.isfile(z):
        os.remove(z)
    print("    失败:", (r.stderr or r.stdout or "")[:150])
    return None


def extract(z, name):
    d = os.path.join(OUT, name)
    if os.path.isdir(d):
        n = sum(len(f) for _, _, f in os.walk(d))
        if n > 0:
            print("  已解压: %s (%d 文件)" % (name, n))
            return d
    try:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
        n = sum(len(f) for _, _, f in os.walk(d))
        print("  解压: %s (%d 文件)" % (name, n))
        return d
    except Exception as e:
        print("  解压失败:", e)
        return None


def find_models(root):
    """找所有 3D 模型文件。"""
    out = {"fbx": [], "obj": [], "glb": [], "gltf": []}
    for dp, _, fs in os.walk(root):
        for f in fs:
            ext = os.path.splitext(f)[1].lower().lstrip(".")
            if ext in out:
                out[ext].append(os.path.join(dp, f))
    return out


if __name__ == "__main__":
    print("=== 下载 Kenney 模型 ===")
    for name, url in SOURCES:
        z = try_download(name, url)
        if z:
            d = extract(z, name)
            if d:
                m = find_models(d)
                tot = sum(len(v) for v in m.values())
                print("    模型: fbx=%d obj=%d glb=%d gltf=%d (共%d)" %
                      (len(m["fbx"]), len(m["obj"]), len(m["glb"]),
                       len(m["gltf"]), tot))
                if tot > 0:
                    for k, v in m.items():
                        if v:
                            print("      例: " + os.path.relpath(v[0], d))

    print("\n=== 汇总 ===")
    m = find_models(OUT)
    print("  FBX:  %d" % len(m["fbx"]))
    print("  OBJ:  %d" % len(m["obj"]))
    print("  GLB:  %d" % len(m["glb"]))
    print("  glTF: %d" % len(m["gltf"]))
