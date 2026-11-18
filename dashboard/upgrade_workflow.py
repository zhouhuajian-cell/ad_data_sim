#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""upgrade_workflow.py — TSR 工作台改造（幂等、只改 HTML/JS 文本）

改动（按实际使用习惯）：
  1) 导航 4 步 → 3 步：数据源 / 工作台 / 批量生成
     原来「选图」和「预览」要来回切右栏标签，步骤太碎
  2) 数据源条常驻：目录 + 国家码 + 一键加载
  3) 工作台加图片导航：上一张/下一张 + 缩略图条 + 键盘 ←→ 和空格

⚠️ 教训：之前用 Python 往 JS 里插代码时，用了 "\\n" 把字面反斜杠 n 写进了 JS，
   导致整个 script 语法错误、页面全白。这里所有换行都用真实换行拼接。
"""
import io
import re
import sys

P = "/opt/ue5_synth/dashboard/tsr.html"
MARK = "[WORKFLOW-V2]"


def main():
    s = io.open(P, encoding="utf-8").read()
    if MARK in s:
        print("已改造过，跳过")
        return 0

    done = []

    # ---------- 1) 导航三步 ----------
    m = re.search(r'<nav id="navTsr">.*?</nav>', s, re.S)
    if not m:
        print("ERROR: navTsr 未找到")
        return 2
    nav = (
        '<nav id="navTsr">' + MARK + '\n'
        '      <button data-step="lib" class="on"><span class="idx">01</span>数据源</button>\n'
        '      <button data-step="work"><span class="idx">02</span>工作台</button>\n'
        '      <button data-step="batch"><span class="idx">03</span>批量生成</button>\n'
        '    </nav>'
    )
    s = s[:m.start()] + nav + s[m.end():]
    done.append("导航三步")

    # ---------- 2) 数据源条 ----------
    srcbar = (
        '<div id="sourceBar" style="display:none;padding:12px;border-bottom:1px solid var(--line);'
        'background:#151920">' + MARK + '\n'
        '      <div class="hint" style="margin-bottom:7px">实拍图目录</div>\n'
        '      <input type="text" id="fDir2" placeholder="/mnt/Data_Platform/.../images"\n'
        '        style="width:100%;padding:7px 9px;border-radius:2px;background:var(--inputBg);'
        'border:1px solid var(--line);outline:0;font-size:12.5px;margin-bottom:7px">\n'
        '      <div style="display:flex;gap:6px;margin-bottom:7px">\n'
        '        <input type="text" id="fCountryCode" value="BR" placeholder="国家码"\n'
        '          style="width:74px;padding:7px 9px;border-radius:2px;background:var(--inputBg);'
        'border:1px solid var(--line);outline:0;font-size:12.5px;text-align:center">\n'
        '        <button class="bigbtn ghost" id="btnLoadImg2" '
        'style="flex:1;margin:0;padding:7px;font-size:13px">加载图片</button>\n'
        '      </div>\n'
        '      <div class="hint" id="loadStat" style="font-size:12px;color:var(--txtMute)">未加载</div>\n'
        '    </div>\n'
        '    <div class="propbody">'
    )
    s = s.replace('<div class="propbody">', srcbar, 1)
    done.append("数据源条")

    # ---------- 3) 工具栏 + 缩略图条 ----------
    i = s.find('<div class="toolbar">')
    j = s.find('<div class="viewport"')
    if i < 0 or j < 0:
        print("ERROR: toolbar/viewport 未找到")
        return 2
    toolbar = (
        '<div class="toolbar">' + MARK + '\n'
        '      <div class="seg" id="viewSeg">\n'
        '        <button data-v="side" class="on">左右对比</button>\n'
        '        <button data-v="single">仅仿真后</button>\n'
        '        <button data-v="orig">仅原图</button>\n'
        '      </div>\n'
        '      <div class="vsep"></div>\n'
        '      <label class="chk" style="padding:0"><input type="checkbox" id="cbBox" checked> 叠加检测框</label>\n'
        '      <div class="vsep"></div>\n'
        '      <button class="btn" id="btnPrevImg">◀ 上一张</button>\n'
        '      <span id="imgIdx" style="min-width:64px;text-align:center;color:var(--txtDim);'
        'font-size:12.5px">— / —</span>\n'
        '      <button class="btn" id="btnNextImg">下一张 ▶</button>\n'
        '      <div class="vsep"></div>\n'
        '      <button class="btn" id="btnRun">生成预览</button>\n'
        '      <span class="right" id="tbInfo">—</span>\n'
        '    </div>\n'
        '    <div class="thumbstrip" id="thumbStrip"><span class="empty">'
        '加载图片后在此快速切换</span></div>\n'
        '    '
    )
    s = s[:i] + toolbar + s[j:]
    done.append("工具栏+缩略图条")

    # ---------- 4) 缩略图条样式 ----------
    css = (
        '.thumbstrip{height:70px;flex:0 0 70px;background:#13171D;'
        'border-top:1px solid var(--line);display:flex;gap:4px;padding:6px 8px;'
        'overflow-x:auto;align-items:center}\n'
        '.thumbstrip::-webkit-scrollbar{height:5px}\n'
        '.thumbstrip::-webkit-scrollbar-thumb{background:#3A424E;border-radius:3px}\n'
        '.thumbstrip button{flex:0 0 74px;height:54px;border-radius:2px;overflow:hidden;'
        'background:#0D1014;padding:0;border:1px solid transparent;position:relative}\n'
        '.thumbstrip button img{width:100%;height:100%;object-fit:cover;display:block;opacity:.7}\n'
        '.thumbstrip button.on{border-color:var(--primary)}\n'
        '.thumbstrip button.on img{opacity:1}\n'
        '.thumbstrip button .badge{position:absolute;right:2px;bottom:2px;font-size:10px;'
        'background:rgba(0,0,0,.75);color:#9FB0C2;padding:0 3px;border-radius:2px}\n'
        '.thumbstrip .empty{color:var(--txtMute);font-size:12px;padding:0 6px}\n'
    )
    s = s.replace('.viewport{flex:1;position:relative;overflow:hidden;min-height:0;',
                  css + '.viewport{flex:1;position:relative;overflow:hidden;min-height:0;', 1)
    done.append("缩略图样式")

    # ---------- 5) 状态字段 ----------
    s = s.replace("lib:null, dets:[], prevUrl:'', curImgUrl:'', origUrl:'',",
                  "lib:null, dets:[], prevUrl:'', curImgUrl:'', origUrl:'', imgs:[], idx:-1,",
                  1)

    # ---------- 6) 导航映射 ----------
    s = s.replace("const map = {lib:'country', pick:'class', prev:'class', batch:'output',",
                  "const map = {lib:'country', work:'class', batch:'output',", 1)

    # ---------- 7) 图片导航函数 ----------
    JSN = "".join([
        "/* " + MARK + " 图片导航 */\n",
        "function renderStrip(){\n",
        "  const strip = $('#thumbStrip');\n",
        "  if (!strip) return;\n",
        "  if (!S.imgs.length) {\n",
        "    strip.innerHTML = '<span class=\"empty\">加载图片后在此快速切换</span>';\n",
        "    return;\n",
        "  }\n",
        "  strip.innerHTML = S.imgs.map((f, i) =>\n",
        "    '<button data-i=\"' + i + '\" class=\"' + (i===S.idx?'on':'') + '\">' +\n",
        "    '<img loading=\"lazy\" src=\"/api/tsr/image?dir=' + encodeURIComponent(S.dir) +\n",
        "    '&file=' + encodeURIComponent(f) + '&max=110\">' +\n",
        "    '<span class=\"badge\">' + (i+1) + '</span></button>').join('');\n",
        "  strip.querySelectorAll('button').forEach(b => b.onclick = () => pickIndex(+b.dataset.i));\n",
        "  const cur = strip.querySelector('button.on');\n",
        "  if (cur) cur.scrollIntoView({block:'nearest', inline:'center'});\n",
        "}\n",
        "\n",
        "function pickIndex(i){\n",
        "  if (!S.imgs.length) return;\n",
        "  i = Math.max(0, Math.min(S.imgs.length - 1, i));\n",
        "  S.idx = i;\n",
        "  S.file = S.imgs[i];\n",
        "  S.prevUrl = ''; S.dets = [];\n",
        "  S.origUrl = '/api/tsr/image?dir=' + encodeURIComponent(S.dir) +\n",
        "              '&file=' + encodeURIComponent(S.file) + '&max=1400';\n",
        "  const el = $('#imgIdx'); if (el) el.textContent = (i+1) + ' / ' + S.imgs.length;\n",
        "  const st = $('#loadStat');\n",
        "  if (st) st.textContent = '当前 ' + (i+1) + '/' + S.imgs.length + '：' + S.file;\n",
        "  $$('#thumbStrip button').forEach(b => b.classList.toggle('on', +b.dataset.i === i));\n",
        "  renderView();\n",
        "  autoDetect();\n",
        "}\n",
        "\n",
        "function stepImage(d){ pickIndex(S.idx < 0 ? 0 : S.idx + d); }\n",
        "\n",
        "/* 数据源：目录 + 国家码 + 一键加载 */\n",
        "async function loadFromSource(){\n",
        "  const dir = $('#fDir2').value.trim();\n",
        "  const cc = ($('#fCountryCode').value || '').trim().toUpperCase();\n",
        "  if (!dir) { alert('请填写实拍图目录'); return; }\n",
        "  S.dir = dir;\n",
        "  if (cc && cc !== S.country) { S.country = cc; loadClasses(); }\n",
        "  $('#loadStat').textContent = '加载中…';\n",
        "  const r = await api('/api/tsr/images?dir=' + encodeURIComponent(dir) + '&limit=60');\n",
        "  if (!r.ok) {\n",
        "    $('#loadStat').textContent = r.d.error || '读取失败';\n",
        "    return;\n",
        "  }\n",
        "  S.imgs = r.d.images || []; S.idx = -1;\n",
        "  renderStrip();\n",
        "  $('#loadStat').textContent = '共 ' + r.d.total + ' 张，已载入前 ' + S.imgs.length + ' 张';\n",
        "  logline('[INFO] 目录共 ' + r.d.total + ' 张');\n",
        "  if (S.imgs.length) pickIndex(0);\n",
        "}\n",
        "\n",
        "function bindWorkflowBtns(){\n",
        "  const p = $('#btnPrevImg'); if (p) p.onclick = () => stepImage(-1);\n",
        "  const n = $('#btnNextImg'); if (n) n.onclick = () => stepImage(1);\n",
        "  const b = $('#btnLoadImg2'); if (b) b.onclick = loadFromSource;\n",
        "}\n",
        "\n",
        "/* 键盘：←/→ 切图，空格生成预览 */\n",
        "document.addEventListener('keydown', e => {\n",
        "  const tn = (e.target.tagName || '').toUpperCase();\n",
        "  if (tn === 'INPUT' || tn === 'TEXTAREA') return;\n",
        "  if (e.key === 'ArrowLeft') { e.preventDefault(); stepImage(-1); }\n",
        "  else if (e.key === 'ArrowRight') { e.preventDefault(); stepImage(1); }\n",
        "  else if (e.key === ' ' && S.file) { e.preventDefault(); runPreview(); }\n",
        "});\n",
        "\n",
    ])
    k = s.find("async function autoDetect(){")
    if k < 0:
        print("ERROR: autoDetect 未找到")
        return 2
    s = s[:k] + JSN + s[k:]
    done.append("导航函数")

    # ---------- 8) 加载后填 imgs ----------
    s = s.replace(
        "  pct('就绪', 0);\n  logline(`[INFO] 目录共 ${r.d.total} 张`);",
        "  pct('就绪', 0);\n  logline(`[INFO] 目录共 ${r.d.total} 张`);\n"
        "  S.imgs = r.d.images || []; S.idx = -1;\n  renderStrip();", 1)

    # ---------- 9) 步骤切换显示数据源条 ----------
    s = s.replace(
        "  switchProp(map[S.step] || 'country');",
        "  switchProp(map[S.step] || 'country');\n"
        "  const _sb = $('#sourceBar');\n"
        "  if (_sb) _sb.style.display = (b.dataset.step === 'lib') ? '' : 'none';", 1)

    # ---------- 10) 初始化时绑定 ----------
    s = s.replace("(async () => {\n  await loadLib();",
                  "bindWorkflowBtns();\n(async () => {\n  await loadLib();", 1)

    io.open(P, "w", encoding="utf-8").write(s)
    print("完成：" + "、".join(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
