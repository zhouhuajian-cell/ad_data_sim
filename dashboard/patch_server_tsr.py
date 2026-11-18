#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_server_tsr.py — 给面板补 TSR 仿真接口

补的接口：
  GET  /api/tsr/library        素材库概况（国家、类别、数量）
  GET  /api/tsr/library/thumb  某素材缩略图 (?country=BR&class=speed_limit_60)
  GET  /api/tsr/library/class  某国家某类别的全部素材（小图网格用）
  GET  /api/tsr/images         目录下图片清单（分页）
  GET  /api/tsr/image          单张实拍图（缩放后返回）
  POST /api/tsr/detect         单图 YOLO 检测
  POST /api/tsr/replace        单图替换预览（写文件，返回 URL）
  POST /api/tsr/batch          批量替换（后台任务）

说明：只用 server.py 已有的 _send/_body，不依赖新的辅助方法。
像素级操作全部交给 tools/ 下的脚本（在 ad_mining 的 venv 里跑），面板本身零依赖。
"""
import io
import os
import sys

MARK = "# [TSR-API]"


def _slug(s):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(s)).strip("_")[:96]


GET_BLOCK = '''
        # [TSR-API] 素材库概况
        if u.path == "/api/tsr/library":
            root = cfg.get("tsr_library") or DEFAULT_TSR_LIBRARY
            items = []
            if os.path.isdir(root):
                for name in sorted(os.listdir(root)):
                    d = os.path.join(root, name)
                    if not os.path.isdir(d) or name.startswith("_"):
                        continue
                    files = [f for f in os.listdir(d)
                             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    if not files:
                        continue
                    classes = sorted(set(os.path.splitext(f)[0].split("_v")[0]
                                         for f in files))
                    items.append({"country": name, "count": len(files),
                                  "class_count": len(classes),
                                  "classes": classes[:80]})
            return self._send(200, {"root": root, "countries": items,
                                    "total_countries": len(items),
                                    "total_assets": sum(i["count"] for i in items)})

        # [TSR-API] 素材缩略图
        if u.path == "/api/tsr/library/thumb":
            country = (q.get("country") or [""])[0]
            cls = (q.get("class") or [""])[0]
            root = cfg.get("tsr_library") or DEFAULT_TSR_LIBRARY
            d = os.path.join(root, _tsr_slug(country))
            target = None
            if os.path.isdir(d) and cls:
                for f in sorted(os.listdir(d)):
                    if os.path.splitext(f)[0].split("_v")[0] == cls and \\
                            f.lower().endswith((".png", ".jpg", ".jpeg")):
                        target = os.path.join(d, f)
                        break
            if not target or not os.path.isfile(target):
                return self._send(404, {"error": "asset not found"})
            return self._send_image(max_w=140)(target)

        # [TSR-API] 某类别的全部素材（小图列表用）
        if u.path == "/api/tsr/library/class":
            country = (q.get("country") or [""])[0]
            cls = (q.get("class") or [""])[0]
            root = cfg.get("tsr_library") or DEFAULT_TSR_LIBRARY
            d = os.path.join(root, _tsr_slug(country))
            names = []
            if os.path.isdir(d) and cls:
                names = sorted(f for f in os.listdir(d)
                               if os.path.splitext(f)[0].split("_v")[0] == cls
                               and f.lower().endswith((".png", ".jpg", ".jpeg")))
            return self._send(200, {"country": country, "class": cls, "files": names})

        # [TSR-API] 实拍图清单
        if u.path == "/api/tsr/images":
            d = (q.get("dir") or [""])[0]
            limit = int((q.get("limit") or ["30"])[0])
            offset = int((q.get("offset") or ["0"])[0])
            if not d or not os.path.isdir(d):
                return self._send(400, {"error": "目录不存在", "dir": d})
            files = sorted(f for f in os.listdir(d)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            return self._send(200, {"dir": d, "total": len(files), "offset": offset,
                                    "images": files[offset:offset + limit]})

        # [TSR-API] 单张实拍图
        if u.path == "/api/tsr/image":
            d = (q.get("dir") or [""])[0]
            f = (q.get("file") or [""])[0]
            if not d or not f or "/" in f or "\\\\" in f or ".." in f:
                return self._send(400, {"error": "参数错误"})
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                return self._send(404, {"error": "not found"})
            maxw = int((q.get("max") or ["900"])[0])
            return self._send_image(max_w=maxw)(p)

        # [TSR-API] 替换结果图
        if u.path == "/api/tsr/result":
            f = (q.get("file") or [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9_.\\-]+\\.(?:jpg|png)", f, re.I):
                return self._send(400, {"error": "invalid"})
            p = os.path.join(TSR_OUT_DIR, f)
            if not os.path.isfile(p):
                return self._send(404, {"error": "not found"})
            with open(p, "rb") as fh:
                blob = fh.read()
            return self._send(200, blob, "image/png" if p.endswith(".png") else "image/jpeg")
'''


POST_BLOCK = '''
        # [TSR-API] 单图检测
        if u.path == "/api/tsr/detect":
            body = self._body()
            d = str(body.get("dir") or "")
            f = str(body.get("file") or "")
            conf = float(body.get("conf") or 0.30)
            if not d or not f:
                return self._send(400, {"error": "缺少 dir/file"})
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                return self._send(400, {"error": "图片不存在"})
            return self._send(200, _run_algo([os.path.join(TOOLS_DIR, "tsr_detect_api.py"),
                                              p, str(conf)], timeout=180))

        # [TSR-API] 单图替换预览
        if u.path == "/api/tsr/replace":
            body = self._body()
            d = str(body.get("dir") or "")
            f = str(body.get("file") or "")
            country = str(body.get("country") or "")
            cls = str(body.get("target_class") or "")
            bbox = body.get("bbox") or []
            if not d or not f:
                return self._send(400, {"error": "缺少 dir/file"})
            os.makedirs(TSR_OUT_DIR, exist_ok=True)
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                return self._send(400, {"error": "图片不存在"})
            out = os.path.join(TSR_OUT_DIR, _tsr_slug(os.path.splitext(f)[0]) + "_preview.jpg")
            args = [os.path.join(TOOLS_DIR, "tsr_preview_api.py"), p, out,
                    "--library", cfg.get("tsr_library") or DEFAULT_TSR_LIBRARY,
                    "--country", country]
            if cls:
                args += ["--target-class", cls]
            if bbox and len(bbox) == 4:
                args += ["--bbox", ",".join(str(float(v)) for v in bbox)]
            res = _run_algo(args, timeout=300)
            if res.get("ok"):
                res["url"] = "/api/tsr/result?file=" + os.path.basename(out)
            return self._send(200, res)

        # [TSR-API] 批量替换
        if u.path == "/api/tsr/batch":
            body = self._body()
            d = str(body.get("dir") or "")
            country = str(body.get("country") or "")
            cls = str(body.get("target_class") or "")
            limit = int(body.get("limit") or 20)
            if not d or not os.path.isdir(d):
                return self._send(400, {"error": "实拍目录不存在"})
            os.makedirs(TSR_OUT_DIR, exist_ok=True)
            cmd = [os.path.join(TOOLS_DIR, "batch_replace.py"),
                   "--images", d, "--signs-dir", cfg.get("tsr_library") or DEFAULT_TSR_LIBRARY,
                   "--out", TSR_OUT_DIR, "--country", country, "--limit", str(limit)]
            if cls:
                cmd += ["--target-class", cls]
            rec, err = start_job("tsr_batch", {"cmd": cmd, "out": TSR_OUT_DIR})
            if err:
                return self._send(400, {"error": err})
            return self._send(self._json_code if hasattr(self, "_json_code") else 200,
                              {"id": rec["id"], "out": TSR_OUT_DIR})
'''


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/opt/ue5_synth/dashboard/server.py"
    with io.open(path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARK in src:
        print("补丁已存在，跳过")
        return 0

    # 1) 顶部常量与辅助函数（放在 start_job 定义之前）
    anchor = "def start_job("
    idx = src.find(anchor)
    if idx < 0:
        print("ERROR: 找不到 start_job 锚点")
        return 2
    prelude = '''# [TSR-API] TSR 仿真相关常量与辅助
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
TOOLS_DIR = os.path.abspath(TOOLS_DIR)
DEFAULT_TSR_LIBRARY = "/data/tsr_library"
TSR_OUT_DIR = "/data/render/tsr_batch"
# 像素/AI 相关脚本必须用 ad_mining 的 venv（有 ultralytics/opencv），面板本体零依赖
ALGO_PYTHON = "/opt/ad_mining/venv/bin/python"


def _tsr_slug(s):
    """把国家码/文件名变成安全的路径片段，防目录穿越。"""
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(s)).strip("_")[:96]


def _run_algo(args, timeout=300):
    """在 ad_mining 的 venv 里跑一个 tools 脚本，返回它 stdout 的 JSON。"""
    import subprocess
    exe = ALGO_PYTHON if os.path.isfile(ALGO_PYTHON) else "python3"
    try:
        p = subprocess.run([exe] + [str(a) for a in args],
                           capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "").strip()
        if not out:
            return {"ok": False, "error": (p.stderr or "脚本无输出")[:300]}
        return json.loads(out.splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


'''
    src = src[:idx] + prelude + src[idx:]

    # 2) 发送缩放图片的辅助（挂到 Handler 上：用模块级函数 + 闭包返回）
    helper_anchor = "    def _body(self):"
    hidx = src.find(helper_anchor)
    if hidx < 0:
        print("ERROR: 找不到 _body 锚点")
        return 2
    helper = '''    def _send_image(self, max_w=900):
        """返回一个「读文件→按需缩放→发出」的闭包。缩放用 PIL，没有就原样发。"""
        def _go(path):
            try:
                with open(path, "rb") as f:
                    blob = f.read()
            except OSError:
                return self._send(404, {"error": "image not found"})
            try:
                from PIL import Image
                import io as _io
                im = Image.open(path)
                if im.width > max_w:
                    h = int(im.height * max_w / float(im.width))
                    im = im.resize((max_w, h), Image.LANCZOS)
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                buf = _io.BytesIO()
                im.save(buf, "JPEG", quality=88)
                blob = buf.getvalue()
                return self._send(200, blob, "image/jpeg")
            except Exception:
                ctype = "image/png" if path.lower().endswith(".png") else "image/jpeg"
                return self._send(200, blob, ctype)
        return _go

'''
    src = src[:hidx] + helper + src[hidx:]

    # 3) GET 路由：插到 /api/asset/image 之后
    idx = src.find('if u.path == "/api/asset/image":')
    if idx < 0:
        print("ERROR: 找不到 /api/asset/image")
        return 2
    tail = src[idx:]
    nxt = tail.find("\n        if u.path ==", 1)
    if nxt < 0:
        print("ERROR: 找不到下一个 GET 路由")
        return 2
    at = idx + nxt
    src = src[:at] + "\n" + GET_BLOCK.rstrip() + src[at:]

    # 4) POST 路由：插到 /api/assets/upload 之后
    idx = src.find('if u.path == "/api/assets/upload":')
    if idx < 0:
        print("ERROR: 找不到 /api/assets/upload")
        return 2
    tail = src[idx:]
    nxt = tail.find("\n        if u.path ==", 1)
    if nxt < 0:
        print("ERROR: 找不到下一个 POST 路由")
        return 2
    at = idx + nxt
    src = src[:at] + "\n" + POST_BLOCK.rstrip() + src[at:]

    with io.open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("已插入 TSR 接口：library/thumb/class/images/image/result + detect/replace/batch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
