#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_server_assets.py — 给 server.py 补「素材 → 路标生成 → 渲染」三段接口

补什么（前端要用、后端还没有的）：
  POST /api/assets/import     触发 UE 导入 AssetInbox → /Game/Assets/Textures（新增）
  POST /api/signs/spawn       用某张素材生成路标到指定关卡（新增）
  GET  /api/signs             列出关卡里已有的路标（新增）
  GET  /api/assets/thumbs     素材缩略图（已有 /api/asset/image，这里补一个批量列表）

做法：以幂等方式往 server.py 里插入路由分支与 action 定义；重复执行不会重复插入
（每条插入都带唯一标记注释）。

用法：python3 patch_server_assets.py /opt/ue5_synth/dashboard/server.py
"""
import io
import os
import re
import sys

MARK_IMPORT = "# [PATCH-ASSETS] 素材导入接口"
MARK_SPAWN = "# [PATCH-ASSETS] 路标生成接口"
MARK_LIST = "# [PATCH-ASSETS] 路标列表接口"

NEW_GET = '''
        {MARK_LIST}
        if u.path == "/api/signs":
            # 列出关卡里的路标 Actor（读 UE 不方便，这里读最近一次生成记录）
            rec_path = os.path.join(load_config()["logs_dir"], "signs.json")
            items = []
            if os.path.exists(rec_path):
                try:
                    with open(rec_path, "r", encoding="utf-8") as f:
                        items = json.load(f)
                except Exception as e:
                    self._json({{"error": str(e)}}, 500)
                    return
            self._json({{"signs": items}})
            return
'''.format(MARK_LIST=MARK_LIST)

NEW_POST = '''
        {MARK_IMPORT}
        if u.path == "/api/assets/import":
            # 触发 UE 导入：AssetInbox 里的图片 → /Game/Assets/Textures
            # 用独立进程跑，避免阻塞 HTTP 线程；前端轮询 /api/job
            rec, err = start_job("import_assets", {{}})
            if err:
                self._json({{"error": err}}, 400)
                return
            self._json({{"id": rec["id"]}})
            return
        {MARK_SPAWN}
        if u.path == "/api/signs/spawn":
            body = self._body()
            mat = str(body.get("material") or "").strip()
            map_path = str(body.get("map") or "").strip()
            if not mat:
                self._json({{"error": "缺少 material（素材材质路径）"}}, 400)
                return
            params = {{
                "material": mat,
                "map": map_path,
                "pos": str(body.get("pos") or "2000,300,220"),
                "yaw": str(body.get("yaw") or "180"),
                "scale": str(body.get("scale") or "1.0"),
            }}
            rec, err = start_job("spawn_sign", params)
            if err:
                self._json({{"error": err}}, 400)
                return
            # 记一笔到 signs.json，供 /api/signs 列出
            try:
                cfg = load_config()
                rec_path = os.path.join(cfg["logs_dir"], "signs.json")
                items = []
                if os.path.exists(rec_path):
                    with open(rec_path, "r", encoding="utf-8") as f:
                        items = json.load(f)
                items.append({{
                    "material": mat,
                    "map": map_path,
                    "pos": params["pos"],
                    "yaw": params["yaw"],
                    "scale": params["scale"],
                    "job": rec["id"],
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }})
                with open(rec_path, "w", encoding="utf-8") as f:
                    json.dump(items, f, indent=1, ensure_ascii=False)
            except Exception:
                pass
            self._json({{"id": rec["id"]}})
            return
'''.format(MARK_IMPORT=MARK_IMPORT, MARK_SPAWN=MARK_SPAWN)

# 新 action → 命令行
NEW_ACTIONS = '''
    # [PATCH-ACTIONS] 素材导入与路标生成
    if action == "import_assets":
        return [py, os.path.join(s, "engine/Content/Python/import_texture.py")]
    if action == "spawn_sign":
        env_script = os.path.join(s, "linux/run_spawn_sign.sh")
        return ["bash", env_script,
                p.get("material", ""), p.get("map", "") or cfg["uproject"],
                p.get("pos", "2000,300,220"), p.get("yaw", "180"), p.get("scale", "1.0")]
'''


def main(path):
    with io.open(path, "r", encoding="utf-8") as f:
        src = f.read()

    changed = []

    # 1) GET 路由：在 /api/assets 之后插入 /api/signs
    if MARK_LIST not in src:
        anchor = 'if u.path == "/api/assets":'
        idx = src.find(anchor)
        if idx < 0:
            print("ERROR: 找不到 /api/assets 路由锚点")
            return 2
        # 找到该 if 块的结束（下一个同级 if 或 return 之后）
        m = re.search(r'(\n\s*if u\.path == "/api/assets":.*?\n(?:.*?\n)*?)\s*\n(\s*)if u\.path == "',
                      src[idx - 20:], re.S)
        # 简化：直接在 /api/assets 块后的第一个 "\n        if u.path ==" 前插入
        tail = src[idx:]
        nxt = tail.find('\n        if u.path ==', 1)
        if nxt < 0:
            print("ERROR: 找不到下一个路由，无法插入")
            return 2
        insert_at = idx + nxt
        src = src[:insert_at] + "\n" + NEW_GET.rstrip() + src[insert_at:]
        changed.append("GET /api/signs")

    # 2) POST 路由：在 /api/assets/upload 之后插入
    if MARK_IMPORT not in src:
        anchor = 'if u.path == "/api/assets/upload":'
        idx = src.find(anchor)
        if idx < 0:
            print("ERROR: 找不到 /api/assets/upload 锚点")
            return 2
        tail = src[idx:]
        nxt = tail.find('\n        if u.path ==', 1)
        if nxt < 0:
            print("ERROR: POST 路由找不到下一条")
            return 2
        insert_at = idx + nxt
        src = src[:insert_at] + "\n" + NEW_POST.rstrip() + src[insert_at:]
        changed.append("POST /api/assets/import + /api/signs/spawn")

    # 3) action → 命令行
    if "[PATCH-ACTIONS]" not in src:
        anchor = '    if action == "preview":'
        idx = src.find(anchor)
        if idx < 0:
            print("ERROR: 找不到 preview action 锚点")
            return 2
        src = src[:idx] + NEW_ACTIONS.lstrip("\n") + "\n" + src[idx:]
        changed.append("actions: import_assets / spawn_sign")

    if not changed:
        print("无需修改（补丁已存在）")
        return 0

    with io.open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("已插入：" + "、".join(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/opt/ue5_synth/dashboard/server.py"))
