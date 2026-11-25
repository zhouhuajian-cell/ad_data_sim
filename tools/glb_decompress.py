#!/usr/bin/env python3
"""把 EXT_meshopt_compression 压缩的 GLB 解压成普通 GLB。

用法:
    python glb_decompress.py <in.glb> <out.glb>
    python glb_decompress.py --batch <src_dir> <dst_dir>
"""
import json
import os
import struct
import sys

import numpy as np

try:
    import meshoptimizer as mo
    HAS_MO = True
except Exception:
    HAS_MO = False

CT = {
    5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
    5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4),
}
TYPE_N = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16,
}


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, ver, total = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not glb")
    off = 12
    js = None
    binc = b""
    while off + 8 <= len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + clen]
        off += clen
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binc = chunk
    return js, binc


def write_glb(path, js, binc):
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    while len(jb) % 4:
        jb += b" "
    b = binc
    while len(b) % 4:
        b += b"\x00"
    total = 12 + 8 + len(jb) + 8 + len(b)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A))
        f.write(jb)
        f.write(struct.pack("<II", len(b), 0x004E4942))
        f.write(b)


def decompress(js, binc):
    if not HAS_MO:
        raise RuntimeError("meshoptimizer 不可用")
    views = js.get("bufferViews", [])
    accs = js.get("accessors", [])

    # bufferView -> 引用它的 accessor
    by_view = {}
    for ai, a in enumerate(accs):
        v = a.get("bufferView")
        if v is not None:
            by_view.setdefault(v, []).append(ai)

    new_views = []
    out = bytearray()
    shared_views = {}

    for vi, bv in enumerate(views):
        ext = (bv.get("extensions") or {}).get("EXT_meshopt_compression")
        if ext is None:
            o = bv.get("byteOffset", 0)
            ln = bv.get("byteLength", 0)
            data = binc[o:o + ln]
        else:
            src_off = ext.get("byteOffset", 0)
            comp_len = ext.get("byteLength", 0)
            src = binc[src_off:src_off + comp_len]
            stride = ext.get("byteStride") or 0
            mode = ext.get("mode", "ATTRIBUTES")
            # COUNT-FROM-ACC: bufferView 在 TRIANGLES 模式下没有 count，
            # 元素个数必须从 accessor 取（每个 accessor 独立 count）
            cnt = bv.get("count")
            if not cnt and by_view.get(vi):
                a0 = accs[by_view[vi][0]]
                cnt = a0.get("count")
            cnt = int(cnt or 0)

            if mode == "ATTRIBUTES":
                # UINT8-VIEW-FIX: 该库 dtype=uint8 分支分配内存有 bug
                # （只分配 count 字节，实际要写 count*stride）
                # 走默认 float32 分支，再 view 回 uint8 取原始字节
                arr = mo.decode_vertex_buffer(cnt, stride, src)
                data = arr.view(np.uint8).tobytes()
                f = ext.get("filter")
                if f == "OCTAHEDRAL":
                    arr2 = mo.decode_filter_oct(
                        np.frombuffer(data, np.uint8), cnt, stride)
                    data = arr2.tobytes()
                elif f == "QUATERNION":
                    arr2 = mo.decode_filter_quat(
                        np.frombuffer(data, np.uint8), cnt, stride)
                    data = arr2.tobytes()
                elif f == "EXPONENTIAL":
                    arr2 = mo.decode_filter_exp(
                        np.frombuffer(data, np.uint8), cnt, stride)
                    data = arr2.tobytes()
            else:
                # SPLIT-SHARED-VIEW: TRIANGLES 模式一个 bv 打包多个索引流
                # 整体 count 在 ext['count']，元素大小在 ext['byteStride']
                total_cnt = int(ext.get("count") or cnt)
                isz = int(stride or 2)
                arr = mo.decode_index_buffer(total_cnt, isz, src)
                data = arr.tobytes()
                # 标记：稍后按 accessor.byteOffset 切分
                shared_views[vi] = True

        while len(out) % 4:
            out += b"\x00"
        nv = {"buffer": 0, "byteOffset": len(out),
              "byteLength": len(data)}
        if "target" in bv:
            nv["target"] = bv["target"]
        # STRIDE-KEEP-FIX: 顶点属性可能带 padding，必须保留 byteStride
        #   （如 POSITION VEC3 short 实际占 8 字节 = 6 + 2 padding）
        _stride = bv.get("byteStride")
        if not _stride and ext and ext.get("mode") == "ATTRIBUTES":
            _stride = ext.get("byteStride")
        if _stride:
            nv["byteStride"] = _stride
        new_views.append(nv)
        out += data

    # SPLIT-SHARED-VIEW: 把共享 bv 拆成每 accessor 一个 bv
    if shared_views:
        remap = {}
        for vi in shared_views:
            refs = by_view.get(vi, [])
            base_view = new_views[vi]
            base_data = out[base_view["byteOffset"]:
                            base_view["byteOffset"] + base_view["byteLength"]]
            for ai in refs:
                a = accs[ai]
                csz = CT.get(a.get("componentType"), ("H", 2))[1]
                cnt = int(a.get("count", 0))
                a_off = int(a.get("byteOffset") or 0)
                seg = base_data[a_off:a_off + cnt * csz]
                while len(out) % 4:
                    out += b"\x00"
                nv = {"buffer": 0, "byteOffset": len(out),
                      "byteLength": len(seg)}
                if "target" in base_view:
                    nv["target"] = base_view["target"]
                new_views.append(nv)
                out += seg
                a["bufferView"] = len(new_views) - 1
                a.pop("byteOffset", None)
        # 重新计算 bufferByteLength

    js["bufferViews"] = new_views
    js["buffers"] = [{"byteLength": len(out)}]

    used = [e for e in (js.get("extensionsUsed") or [])
            if e != "EXT_meshopt_compression"]
    req = [e for e in (js.get("extensionsRequired") or [])
           if e != "EXT_meshopt_compression"]
    if used:
        js["extensionsUsed"] = used
    else:
        js.pop("extensionsUsed", None)
    if req:
        js["extensionsRequired"] = req
    else:
        js.pop("extensionsRequired", None)
    return js, bytes(out)


def one(src, dst):
    js, binc = read_glb(src)
    used = js.get("extensionsUsed") or []
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if "EXT_meshopt_compression" not in used:
        with open(src, "rb") as f:
            data = f.read()
        with open(dst, "wb") as f:
            f.write(data)
        return "copy"
    js2, b2 = decompress(js, binc)
    write_glb(dst, js2, b2)
    return "decomp %.1fMB" % (len(b2) / 1e6)


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--batch":
        s, d = sys.argv[2], sys.argv[3]
        n_ok = n_fail = 0
        for dp, _, fn in os.walk(s):
            for f in fn:
                if not f.endswith(".glb"):
                    continue
                src = os.path.join(dp, f)
                rel = os.path.relpath(src, s)
                dst = os.path.join(d, rel)
                try:
                    r = one(src, dst)
                    print("OK  %-50s %s" % (rel, r))
                    n_ok += 1
                except Exception as e:
                    print("ERR %-50s %s" % (rel, str(e)[:70]))
                    n_fail += 1
        print()
        print("成功 %d / 失败 %d" % (n_ok, n_fail))
        return
    if len(sys.argv) < 3:
        print("usage: glb_decompress.py <in> <out> | --batch <src> <dst>")
        return
    print(one(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
