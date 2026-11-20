import io
import re

s = io.open("/opt/ue5_synth/dashboard/index.html", encoding="utf-8").read()

# 提取所有 script 块，逐个检查
blocks = list(re.finditer(r"<script(?![^>]*src)[^>]*>([\s\S]*?)</script>", s))
print("script 块数:", len(blocks))
for k, m in enumerate(blocks):
    body = m.group(1)
    if not body.strip():
        continue
    # 简单平衡检查（跳过注释和字符串的粗略版）
    dp = 0
    db = 0
    i = 0
    n = len(body)
    in_str = None
    in_lc = False
    in_bc = False
    esc = False
    while i < n:
        c = body[i]
        if c == "\n":
            in_lc = False
            i += 1
            continue
        if in_lc:
            i += 1
            continue
        if in_bc:
            if c == "*" and i + 1 < n and body[i + 1] == "/":
                in_bc = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
            i += 1
            continue
        if c == "/" and i + 1 < n and body[i + 1] == "/":
            in_lc = True
            i += 2
            continue
        if c == "/" and i + 1 < n and body[i + 1] == "*":
            in_bc = True
            i += 2
            continue
        if c in "\"'`":
            in_str = c
            i += 1
            continue
        if c == "(":
            dp += 1
        elif c == ")":
            dp -= 1
        elif c == "{":
            db += 1
        elif c == "}":
            db -= 1
        i += 1
    status = "OK" if (dp == 0 and db == 0) else "!! dp=%d db=%d" % (dp, db)
    print("  块 %d: %d 字符  %s  头: %s" % (k, len(body), status, body.strip()[:50].replace("\n", " ")))
