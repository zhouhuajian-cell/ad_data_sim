#!/usr/bin/env python3
"""GND 单 clip 按参数重绘（供 /api/gnd/render 调用）。"""
import argparse, glob, json, os, sys

sys.path.insert(0, "/opt/ue5_synth/tools")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    batch, clip = cfg["batch"], cfg["clip"]
    params = cfg.get("params") or {}
    out = cfg["out"]

    # NEW-PIPELINE: 走 pipeline_sim（净面 + 素材贴图 + 物理做旧）
    cp = os.path.join("/data/test_data/gnd_train", batch, clip)
    if not os.path.isdir(cp):
        print("clip not found:", cp)
        return 1

    from pipeline_sim import process_clip as sim_process
    # 强制 clean 模式 + 注入素材参数
    p2 = dict(params or {})
    p2.setdefault("gnd_mode", "clean")
    # 前端字段名映射
    if p2.get("gnd_spec_on") and p2.get("gnd_items"):
        p2["gnd_items"] = p2["gnd_items"]
    # BATCH-VS-INTERACTIVE:
    #   cfg["batch_mode"]=True  → 整 clip 全处理（nframes=0 表示全部）
    #   否则                    → 只处理当前 1 帧（前端交互）
    _raw_nf = cfg.get("frames", None)
    if cfg.get("batch_mode"):
        nf = int(_raw_nf) if _raw_nf not in (None, "") else 0
        use_sess = False
    else:
        # FULL-CLIP-GND: 交互也渲染整个 clip（0 = 全部）
        nf = int(_raw_nf) if _raw_nf not in (None, "", 0) else 0
        use_sess = False
    print("    参数:", json.dumps(p2, ensure_ascii=False)[:300])
    n = sim_process(batch, clip, "gnd", p2, out, nframes=nf,
                    use_session=use_sess)
    print("rendered %d frames" % n)
    return 0 if n and n > 0 else 2

if __name__ == "__main__":
    sys.exit(main())
