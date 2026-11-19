#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_report.py — 把批量产出打包成单文件 HTML 报告

用法：
  python3 make_report.py --out /data/render/tsr_100 --title "TSR 仿真 100 张"
                         --html /data/render/tsr_100_report.html [--limit 0]
"""
import argparse
import base64
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


CSS = """
:root{--bg:#0E1116;--panel:#151B23;--line:#232C38;--txt:#E6EBF2;--dim:#8C99AA;
--primary:#3E7BFA;--ok:#36CFC9;--warn:#F5A623}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{padding:26px 32px 22px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:20px;font-weight:600;letter-spacing:.3px}
.sub{color:var(--dim);font-size:13px;margin-top:5px}
.stats{display:flex;gap:32px;margin-top:16px;flex-wrap:wrap}
.stat{font-size:12.5px;color:var(--dim)}
.stat b{color:var(--ok);font-size:15px;margin-left:6px}
.note{margin:20px 32px;padding:16px 18px;background:#12181F;border-left:3px solid var(--primary);
border-radius:4px;color:var(--dim);font-size:12.5px;line-height:1.9}
.note b{color:var(--txt)}
.note .warn{color:var(--warn)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px;
padding:6px 32px 50px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
figure{position:relative;aspect-ratio:1.6;background:#000;overflow:hidden}
figure img{width:100%;height:100%;object-fit:cover;display:block}
figcaption{position:absolute;top:8px;left:8px;font-size:11px;padding:2px 9px;border-radius:3px;
background:rgba(8,16,26,.88);color:#9FB0C4}
figure:last-child figcaption{color:var(--ok)}
.info{padding:10px 13px;display:flex;justify-content:space-between;align-items:center;gap:10px}
.cls{color:var(--primary);font-weight:600;font-size:12.5px}
.meta{color:var(--dim);font-size:11px;font-family:ui-monospace,monospace}
footer{padding:24px 32px 40px;color:#4A5568;font-size:11.5px;border-top:1px solid var(--line)}
"""


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="批量产出目录")
    ap.add_argument("--html", default="", help="输出 html 路径")
    ap.add_argument("--title", default="TSR 标牌仿真 · 样张对比")
    ap.add_argument("--limit", type=int, default=0, help="最多包含多少组（0=全部）")
    ap.add_argument("--crop", action="store_true",
                    help="用标牌特写而不是整图（整图看融合，特写看细节）")
    a = ap.parse_args()

    anns = sorted(glob.glob(os.path.join(a.out, "annotations", "*.json")))
    if a.limit:
        anns = anns[:a.limit]
    summary = {}
    sp = os.path.join(a.out, "summary.json")
    if os.path.isfile(sp):
        with open(sp, encoding="utf-8") as f:
            summary = json.load(f)

    cards = []
    for ap_ in anns:
        try:
            with open(ap_, encoding="utf-8") as f:
                d = json.load(f)
            objs = d.get("objects", [])
            if not objs:
                continue
            src = d.get("source_path", "")
            comp = os.path.join(a.out, d.get("image", ""))
            if not (os.path.isfile(src) and os.path.isfile(comp)):
                continue
            orig = cv2.imread(src)
            sim = cv2.imread(comp)
            if orig is None or sim is None:
                continue

            if a.crop:
                bb = objs[0]["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bb]
                pad = max(10, int((x2 - x1) * 0.5))
                H, W = orig.shape[:2]
                a0, b0 = max(0, y1 - pad), max(0, x1 - pad)
                a1, b1 = min(H, y2 + pad), min(W, x2 + pad)
                S = 320
                ca = cv2.resize(orig[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)
                cb = cv2.resize(sim[a0:a1, b0:b1], (S, S), interpolation=cv2.INTER_NEAREST)
            else:
                tw = 640
                th = int(orig.shape[0] * tw / orig.shape[1])
                ca = cv2.resize(orig, (tw, th), interpolation=cv2.INTER_AREA)
                cb = cv2.resize(sim, (tw, th), interpolation=cv2.INTER_AREA)

            _, ba = cv2.imencode(".jpg", ca, [cv2.IMWRITE_JPEG_QUALITY, 92])
            _, bb_ = cv2.imencode(".jpg", cb, [cv2.IMWRITE_JPEG_QUALITY, 92])
            o = objs[0]
            cards.append({
                "orig": base64.b64encode(ba).decode(),
                "sim": base64.b64encode(bb_).decode(),
                "cls": o.get("class", ""),
                "mat": o.get("material", ""),
                "size": "%dx%d" % (int(o["bbox"][2] - o["bbox"][0]),
                                   int(o["bbox"][3] - o["bbox"][1])),
            })
        except Exception:
            continue

    print("卡片数:", len(cards))

    stats = [
        ("样张", str(len(cards))),
        ("扫描帧数", str(summary.get("scanned_files", "-"))),
        ("寻优救回", str(summary.get("tune_saved", "-"))),
        ("Gate4 剔除", str(summary.get("gate4_reject", "-"))),
        ("用时", "%.0f 分钟" % (summary.get("elapsed_sec", 0) / 60)
         if summary.get("elapsed_sec") else "-"),
    ]

    html = [
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">',
        '<title>%s</title><style>%s</style></head><body>' % (a.title, CSS),
        '<header><h1>%s</h1>' % a.title,
        '<div class="sub">左：实拍原图　右：仿真替换后%s</div>'
        % ("（标牌特写）" if a.crop else "（完整场景）"),
        '<div class="stats">',
    ]
    for k, v in stats:
        html.append('<div class="stat">%s<b>%s</b></div>' % (k, v))
    html.append('</div></header>')

    html.append(
        '<div class="note">'
        '<b>处理流程：</b>YOLO 检测 → 形状严格匹配（圆↔圆/方↔方/三角↔三角）→ '
        '遮挡保留 → 光度/模糊/噪点对齐（P5~P95 黑白点映射 + 迭代模糊匹配）→ '
        '逐帧寻优（grow × brightness）→ 四道质检门控。<br>'
        '<b>四道质检：</b>Gate1 几何完整性、Gate2 边缘跳变、Gate3 亮度一致性、'
        'Gate4 模型自检（合成图喂回 YOLO，认不出就丢弃）。<br>'
        '<span class="warn">已知局限：</span>椭圆牌与「方形底板上的圆牌」形状无法完全贴合；'
        '内容按大类随机替换，语义可能与原牌不同。'
        '</div>'
    )

    html.append('<div class="grid">')
    for c in cards:
        html.append(
            '<div class="card"><div class="pair">'
            '<figure><img src="data:image/jpeg;base64,%s"><figcaption>原图</figcaption></figure>'
            '<figure><img src="data:image/jpeg;base64,%s"><figcaption>仿真后</figcaption></figure>'
            '</div><div class="info"><span class="cls">%s</span>'
            '<span class="meta">%s · %s</span></div></div>'
            % (c["orig"], c["sim"], c["cls"], c["size"], c["mat"])
        )
    html.append('</div>')
    html.append('<footer>PerceptionSim · TSR 标牌仿真平台　|　'
                '单文件报告，图片已内嵌，可直接转发</footer></body></html>')

    out = a.html or os.path.join(a.out, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print("HTML:", out, "%.2f MB" % (os.path.getsize(out) / 1024 / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
