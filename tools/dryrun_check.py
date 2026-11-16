# -*- coding: utf-8 -*-
# dryrun_check.py — 数据链路自检（不依赖 UE / 不依赖 N 卡）
# 干什么：用 mock_frames.py 造合成帧，按真实命令顺序跑 queue_gen → postprocess → pack_dataset，
#         再对 capture_math 投影公式与后处理纯函数做断言；任何一步不符立即 FAIL。
# 怎么跑：python tools/dryrun_check.py [--frames 12] [--keep] [--out DIR]
# 成功判据：末行「自检通过：PASS n / FAIL 0」，退出码 0。产物默认跑完清理，--keep 保留便于排查。
# 用途：改完 post/、queue_gen、capture_math 后在本地做回归；dashboard 的「链路自检」按钮跑的就是它。
# ⚠️ 自检通过只证明脚本契约与数值链路自洽，**不证明**引擎内捕获可用（阶段三仍需编辑器验收）。
import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

# 依赖检查放在最前面：线上服务器（系统 python3）可能没装 numpy/Pillow，
# 直接 import 会甩一大段 traceback，看不出该怎么办。这里给出可执行的一句话。
_MISSING = []
try:
    import numpy as np
except ImportError:          # noqa: N816
    np = None
    _MISSING.append("numpy")
try:
    from PIL import Image
except ImportError:          # noqa: N816
    Image = None
    _MISSING.append("Pillow")
if _MISSING:
    sys.stderr.write(
        "自检需要 %s，当前 Python（%s）没有装。\n"
        "  Ubuntu：sudo apt install python3-numpy python3-pil  （或 pip3 install numpy pillow）\n"
        "  Windows：pip install numpy pillow\n"
        "装好后重新点「脚本自检」即可。\n" % ("、".join(_MISSING), sys.executable))
    sys.exit(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PASS, FAIL, SKIP = [], [], []


def _env():
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"     # 子进程输出统一 utf-8，避免 Windows 本地调试乱码
    e["PYTHONUTF8"] = "1"
    return e


def run(args, cwd=ROOT, expect=0):
    p = subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=_env())
    if p.returncode != expect:
        raise AssertionError("命令退出码 %d（预期 %d）\n  cmd: %s\n  stdout:\n%s\n  stderr:\n%s"
                             % (p.returncode, expect, " ".join(args),
                                p.stdout.strip()[-1500:], p.stderr.strip()[-1500:]))
    return p


def run_fail(args, cwd=ROOT):
    """预期失败的命令：退出码必须非 0。"""
    p = subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=_env())
    if p.returncode == 0:
        raise AssertionError("命令本应失败却退出 0: %s\nstdout:\n%s" % (" ".join(args), p.stdout))
    return p


def check(name, fn):
    try:
        fn()
    except Exception as e:
        FAIL.append(name)
        msg = str(e) or repr(e)
        print("FAIL %s\n     %s" % (name, msg.replace("\n", "\n     ")))
    else:
        PASS.append(name)
        print("PASS %s" % name)


def skip(name, why):
    SKIP.append(name)
    print("SKIP %s（%s）" % (name, why))


def load_module(name, rel_path):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_files(d, suffix):
    return sum(1 for f in os.listdir(d) if f.endswith(suffix))


def expected_boxes(meta_dir):
    """独立于 pack_dataset 的期望值：类别合法、框 4 个数字、裁切后不小于 1px。"""
    classes = {"road", "vehicle", "pedestrian", "traffic_sign", "cone", "vegetation"}
    n = 0
    for fn in sorted(os.listdir(meta_dir)):
        if not fn.endswith(".json"):
            continue
        fr = read_json(os.path.join(meta_dir, fn))
        w, h = int(fr.get("width") or 1920), int(fr.get("height") or 1080)
        for ob in fr.get("objects") or []:
            b = ob.get("bbox_2d")
            if ob.get("class") not in classes or not isinstance(b, list) or len(b) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
            if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--keep", action="store_true", help="保留自检目录")
    ap.add_argument("--out", default="", help="自检目录（默认临时目录）")
    a = ap.parse_args()
    base = os.path.abspath(a.out) if a.out else tempfile.mkdtemp(prefix="ue5_synth_dryrun_")
    os.makedirs(base, exist_ok=True)
    print("== 自检目录 %s" % base)

    scenes = read_json(os.path.join(ROOT, "config", "scenes.json"))
    lights = read_json(os.path.join(ROOT, "config", "lights.json"))
    targets = read_json(os.path.join(ROOT, "config", "targets.json"))
    cameras = read_json(os.path.join(ROOT, "config", "cameras.json"))
    n_tasks = len(scenes) * len(lights) * len(targets) * len(cameras)

    # ---------- 1. queue_gen：条数、幂等、非法输入拦截 ----------
    q1 = os.path.join(base, "q1.jsonl")
    q2 = os.path.join(base, "q2.jsonl")

    def t_queue_gen():
        p = run(["common/queue_gen.py", "--scenes", "config/scenes.json",
                 "--lights", "config/lights.json", "--targets", "config/targets.json",
                 "--cameras", "config/cameras.json", "-o", q1])
        lines = open(q1, encoding="utf-8").read().strip().splitlines()
        assert len(lines) == n_tasks, "行数 %d != 组合数 %d" % (len(lines), n_tasks)
        assert "已写" in p.stdout
    check("queue_gen 展开条数 = %d" % n_tasks, t_queue_gen)

    def t_queue_idempotent():
        run(["common/queue_gen.py", "--scenes", "config/scenes.json",
             "--lights", "config/lights.json", "--targets", "config/targets.json",
             "--cameras", "config/cameras.json", "-o", q2])
        b1, b2 = open(q1, "rb").read(), open(q2, "rb").read()
        assert b1 == b2, "同参数两次输出不一致（不可复现）"
    check("queue_gen 幂等（逐字节一致）", t_queue_idempotent)

    def t_queue_reject():
        bad = os.path.join(base, "bad_targets.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump([[{"class": "bogus", "actor_class": "/Game/X.X",
                        "location": [0, 0, 0], "rotation": [0, 0, 0]}]], f)
        out = os.path.join(base, "bad.jsonl")
        p = run_fail(["common/queue_gen.py", "--scenes", "config/scenes.json",
                      "--lights", "config/lights.json", "--targets", bad,
                      "--cameras", "config/cameras.json", "-o", out])
        assert "FAIL" in p.stderr and not os.path.exists(out), "非法输入应报错且不写文件"
    check("queue_gen 非法输入报错且不写文件", t_queue_reject)

    # ---------- 2. mock 假帧 ----------
    d8 = os.path.join(base, "data8")
    run(["tools/mock_frames.py", "--out", d8, "--frames", str(a.frames), "--bit", "8"])
    meta8 = os.path.join(d8, "meta")
    raw8 = os.path.join(d8, "raw")

    def t_mock():
        assert count_files(meta8, ".json") == a.frames, "meta 帧数不对"
        for suf in ("_rgb.png", "_depth.png", "_sem.png"):
            assert count_files(raw8, suf) == a.frames, "raw %s 数量不对" % suf
        fr = read_json(os.path.join(meta8, "av_000000.json"))
        for key in ("frame_id", "width", "height", "speed", "camera", "lighting",
                    "objects", "image_files"):
            assert key in fr, "真值缺字段 %s" % key
        assert fr["image_files"]["rgb"] == "av_000000_rgb.png"
    check("mock_frames 四通道 + 真值字段齐全", t_mock)

    # ---------- 3. postprocess：命名、条数、参数日志、工况映射、续跑、可复现 ----------
    final8 = os.path.join(d8, "final")

    def t_post():
        p = run(["post/postprocess.py", "--raw", raw8, "--meta", meta8, "--out", final8])
        assert "new=%d" % a.frames in p.stdout, p.stdout
        assert count_files(final8, ".png") == a.frames, "成品图数量不对"
        logs = [json.loads(ln) for ln in
                open(os.path.join(final8, "params_log.jsonl"), encoding="utf-8").read().splitlines()]
        assert len(logs) == a.frames, "参数日志条数 %d != %d" % (len(logs), a.frames)
        for rec in logs:
            assert "error" not in rec, rec
            assert rec["bit_depth"] == 8, rec
            assert rec["params"]["domain_align"] is False, "没给 --real 却启用了域对齐"
            assert rec["dst"] == rec["frame"] + ".png", "成品图应和 meta 同主文件名"
        # 工况映射：真值 lighting.condition 必须落到参数上
        by_frame = {r["frame"]: r for r in logs}
        for fn in sorted(os.listdir(meta8)):
            fr = read_json(os.path.join(meta8, fn))
            cond = fr["lighting"]["condition"]
            rec = by_frame[fr["frame_id"]]
            assert rec["condition"] == cond
            if cond in pp.CONDITION_PARAMS:
                want = pp.CONDITION_PARAMS[cond]["exposure_ev"]
                assert abs(rec["params"]["exposure_ev"] - want) < 1e-9, \
                    "%s 曝光未按工况映射" % fr["frame_id"]
            want_blur = min(13.0, float(fr["speed"]) * pp.BLUR_PX_PER_MPS)
            assert abs(rec["params"]["blur_px"] - want_blur) < 1e-6, "模糊核长未按车速换算"
        # 成品图尺寸与真值一致
        fr = read_json(os.path.join(meta8, "av_000000.json"))
        im = Image.open(os.path.join(final8, "av_000000.png"))
        assert im.size == (fr["width"], fr["height"]), "成品尺寸 %s != %s" % (im.size, (fr["width"], fr["height"]))
    check("postprocess 成品/日志/工况映射/尺寸", t_post)

    def t_post_resume():
        p = run(["post/postprocess.py", "--raw", raw8, "--meta", meta8, "--out", final8])
        assert "skip=%d" % a.frames in p.stdout, "续跑未跳过已有成品: %s" % p.stdout
    check("postprocess 续跑跳过已完成帧", t_post_resume)

    def t_post_repro():
        victim = os.path.join(final8, "av_000001.png")
        h1 = open(victim, "rb").read()
        os.remove(victim)
        run(["post/postprocess.py", "--raw", raw8, "--meta", meta8, "--out", final8])
        h2 = open(victim, "rb").read()
        assert h1 == h2, "同 seed 重跑结果不一致（随机源不确定）"
    check("postprocess 同 seed 重跑逐字节一致", t_post_repro)

    def t_post_params():
        out = os.path.join(base, "final_params")
        pj = os.path.join(base, "params.json")
        with open(pj, "w", encoding="utf-8") as f:
            json.dump({"exposure_ev": 0.5, "read_noise": 0.0,
                       "domain_align": False, "bogus_key": 1}, f)
        p = run(["post/postprocess.py", "--raw", raw8, "--meta", meta8,
                 "--out", out, "--params", pj, "--limit", "1"])
        assert "忽略未知参数键" in p.stderr, "未知参数键应告警"
        rec = json.loads(open(os.path.join(out, "params_log.jsonl"),
                              encoding="utf-8").read().splitlines()[0])
        assert abs(rec["params"]["exposure_ev"] - 0.5) < 1e-9, "params.json 显式值未优先"
        assert rec["params"]["read_noise"] == 0.0
    check("postprocess params.json 显式键优先 + 未知键告警", t_post_params)

    # ---------- 4. case_check：大小写与模块一致性 ----------
    def t_case_check():
        proj = os.path.join(base, "proj")
        os.makedirs(os.path.join(proj, "Source", "avsynth"), exist_ok=True)
        os.makedirs(os.path.join(proj, "Content", "props"), exist_ok=True)
        up = os.path.join(proj, "proj.uproject")
        with open(up, "w", encoding="utf-8") as f:
            json.dump({"Modules": [{"Name": "avsynth"}]}, f)
        with open(os.path.join(proj, "Source", "avsynth", "avsynth.Build.cs"), "w") as f:
            f.write("// mock\n")
        asset = os.path.join(proj, "Content", "props", "Sign.uasset")
        with open(asset, "wb") as f:
            f.write(b"UE5MOCK\x00\x00\x00\x00" + b"/Game/props/sign")
        p = run_fail(["common/case_check.py", proj])
        assert "大小写不一致" in p.stdout, "小写引用未报不一致: %s" % p.stdout
        with open(asset, "wb") as f:                 # 引用改成与磁盘一致的大小写
            f.write(b"UE5MOCK\x00\x00\x00\x00" + b"/Game/props/Sign")
        p = run(["common/case_check.py", proj])
        assert "ALL PASS" in p.stdout, p.stdout
        with open(up, "w", encoding="utf-8") as f:   # 声明一个不存在的模块
            json.dump({"Modules": [{"Name": "avsynth"}, {"Name": "Ghost"}]}, f)
        p = run_fail(["common/case_check.py", proj])
        assert "Ghost" in p.stdout, "未报模块缺失: %s" % p.stdout
    check("case_check 大小写/模块一致性拦截", t_case_check)

    # ---------- 5. pack_dataset：COCO ----------
    coco = os.path.join(base, "coco")
    exp_boxes = expected_boxes(meta8)

    def t_coco():
        run(["post/pack_dataset.py", "--meta", meta8, "--images", final8,
             "--out", coco, "--format", "coco", "--val-ratio", "0.25"])
        allj = read_json(os.path.join(coco, "annotations", "instances_all.json"))
        assert len(allj["images"]) == a.frames, "COCO 帧数不对"
        assert len(allj["annotations"]) == exp_boxes, \
            "COCO 框数 %d != 期望 %d" % (len(allj["annotations"]), exp_boxes)
        assert [c["name"] for c in allj["categories"]] == \
            ["road", "vehicle", "pedestrian", "traffic_sign", "cone", "vegetation"]
        img_by_id = {im["id"]: im for im in allj["images"]}
        for an in allj["annotations"]:
            x, y, w, h = an["bbox"]
            im = img_by_id[an["image_id"]]
            assert w > 0 and h > 0, "退化框"
            assert x >= 0 and y >= 0 and x + w <= im["width"] + 1e-6 \
                and y + h <= im["height"] + 1e-6, "框越界: %s" % an
            assert abs(an["area"] - w * h) < 1e-6
        assert img_by_id[1]["file_name"] == "av_000000.png", "COCO 文件名应同 meta 主名"
        tr = read_json(os.path.join(coco, "annotations", "instances_train.json"))
        va = read_json(os.path.join(coco, "annotations", "instances_val.json"))
        tri = {im["id"] for im in tr["images"]}
        vai = {im["id"] for im in va["images"]}
        assert len(tri) + len(vai) == a.frames and not (tri & vai), "COCO 切分重叠或漏帧"
        assert len(vai) == max(1, round(a.frames * 0.25)), "val 比例未生效"
        assert len(tr["annotations"]) + len(va["annotations"]) == len(allj["annotations"])
    check("pack_dataset COCO（计数/越界/切分/文件名）", t_coco)

    # ---------- 6. pack_dataset：YOLO ----------
    yolo = os.path.join(base, "yolo")

    def t_yolo():
        run(["post/pack_dataset.py", "--meta", meta8, "--images", final8,
             "--out", yolo, "--format", "yolo", "--val-ratio", "0.25"])
        labels = os.path.join(yolo, "labels")
        assert count_files(labels, ".txt") == a.frames, "YOLO 标签数不对"
        cls_line = 0
        for fn in os.listdir(labels):
            for ln in open(os.path.join(labels, fn), encoding="utf-8").read().splitlines():
                parts = ln.split()
                assert len(parts) == 5, "YOLO 行应为 cls cx cy w h: %r" % ln
                c, cx, cy, w, h = parts[0], *map(float, parts[1:])
                assert 0 <= int(c) < 6 and 0 <= cx <= 1 and 0 <= cy <= 1
                assert 0 < w <= 1 and 0 < h <= 1
                cls_line += 1
        assert cls_line == exp_boxes, "YOLO 框数 %d != 期望 %d" % (cls_line, exp_boxes)
        yml = open(os.path.join(yolo, "data.yaml"), encoding="utf-8").read()
        assert "nc: 6" in yml and "traffic_sign" in yml, "data.yaml 类别不全"
        trl = open(os.path.join(yolo, "train.txt"), encoding="utf-8").read().splitlines()
        val = open(os.path.join(yolo, "val.txt"), encoding="utf-8").read().splitlines()
        assert len(trl) + len(val) == a.frames, "YOLO 切分条数不对"
        for p in trl + val:
            assert os.path.exists(p), "YOLO 列表里的图像不存在: %s" % p
        assert len(val) == max(1, round(a.frames * 0.25)), "YOLO val 比例未生效"
    check("pack_dataset YOLO（标签/切分/data.yaml/路径）", t_yolo)

    # ---------- 7. 16bit 路径：OpenCV 解码 + 缺依赖时明确报错 ----------
    d16 = os.path.join(base, "data16")
    run(["tools/mock_frames.py", "--out", d16, "--frames", "3", "--bit", "16"])

    def t_16bit():
        final16 = os.path.join(d16, "final")
        run(["post/postprocess.py", "--raw", os.path.join(d16, "raw"),
             "--meta", os.path.join(d16, "meta"), "--out", final16])
        logs = [json.loads(ln) for ln in
                open(os.path.join(final16, "params_log.jsonl"), encoding="utf-8").read().splitlines()]
        assert len(logs) == 3 and all(r["bit_depth"] == 16 for r in logs), \
            "16bit 未被识别: %s" % [r.get("bit_depth") for r in logs]
        w, h, depth, ctype = pp.png_info(os.path.join(d16, "raw", "av_000000_rgb.png"))
        assert (depth, ctype) == (16, 2), "mock 16bit PNG 头不对: %s %s" % (depth, ctype)
    if pp.cv2 is None:
        skip("postprocess 16bit 解码", "当前环境无 OpenCV")
    else:
        check("postprocess 16bit 解码（OpenCV）", t_16bit)

    def t_16bit_strict():
        saved = pp.cv2
        try:
            pp.cv2 = None
            try:
                pp.load_linear_rgb(os.path.join(d16, "raw", "av_000000_rgb.png"))
            except pp.RawFormatError as e:
                assert "OpenCV" in str(e), "报错信息应给出可执行的安装提示"
            else:
                raise AssertionError("缺 OpenCV 时应报错，不能静默截断成 8bit")
        finally:
            pp.cv2 = saved
        # 同函数读 8bit 不受影响
        arr, depth = pp.load_linear_rgb(os.path.join(raw8, "av_000000_rgb.png"))
        assert depth == 8 and arr.shape[2] == 3 and arr.max() <= 1.0
    check("postprocess 16bit 缺依赖时明确报错（不静默降级）", t_16bit_strict)

    # ---------- 8. 纯函数：运动模糊形状/守恒 ----------
    def t_blur():
        const = np.full((4, 10, 3), 0.5, dtype=np.float32)
        out = pp.motion_blur(const, 5)
        assert out.shape == const.shape, "模糊后尺寸变了: %s != %s" % (out.shape, const.shape)
        assert np.allclose(out, 0.5, atol=1e-9), "常数图经过归一化模糊后应不变"
        imp = np.zeros((3, 9, 3), dtype=np.float32)
        imp[:, 4, :] = 1.0
        got = pp.motion_blur(imp, 3)
        assert got.shape == imp.shape
        assert abs(got.sum() - imp.sum()) < 1e-6, \
            "冲激响应应守恒: sum=%s != %s" % (got.sum(), imp.sum())
        assert np.allclose(pp.motion_blur(imp, 0), imp), "核长 0 应原样返回"
    check("motion_blur 尺寸/守恒/零核", t_blur)

    # ---------- 9. capture_math：投影公式 ----------
    def t_math():
        f, r, u = cm.camera_basis(0, 0)
        assert np.allclose(f, (1, 0, 0)) and np.allclose(r, (0, 1, 0)) and np.allclose(u, (0, 0, 1))
        u0, v0, d0 = cm.world_to_screen((10, 0, 0), (0, 0, 0), 0, 0, 90, 1920, 1080)
        assert abs(u0 - 960) < 1e-6 and abs(v0 - 540) < 1e-6 and abs(d0 - 10) < 1e-9
        ur, vr, _ = cm.world_to_screen((10, 1, 0), (0, 0, 0), 0, 0, 90, 1920, 1080)
        assert ur > u0 and abs(vr - v0) < 1e-6, "右侧点应投到画面右侧"
        ua, va, _ = cm.world_to_screen((10, 0, 1), (0, 0, 0), 0, 0, 90, 1920, 1080)
        assert va < v0, "上方点应投到画面上方（y 向下）"
        _, _, dback = cm.world_to_screen((-10, 0, 0), (0, 0, 0), 0, 0, 90, 1920, 1080)
        assert dback < 0, "相机后方的深度应为负"
        box = cm.project_aabb((10, 0, 0), (0.5, 0.5, 0.5), 0, (0, 0, 0), 0, 0, 90, 1920, 1080)
        assert box is not None and box[2] > box[0] and box[3] > box[1]
        assert 0 <= box[0] and box[2] <= 1920 and 0 <= box[1] and box[3] <= 1080
        assert box[4] > 0 and box[5] is False
        assert cm.project_aabb((-10, 0, 0), (0.5, 0.5, 0.5), 0, (0, 0, 0), 0, 0, 90,
                               1920, 1080) is None, "相机后方整体不可见应返回 None"
        assert cm.normalize_depth(100) == 32768
        assert cm.normalize_depth(200) == 65535 and cm.normalize_depth(300) == 65535
        clipped = cm.clip_bbox((-50, -10, 100, 60), 1920, 1080)
        assert clipped == (0.0, 0.0, 100, 60), clipped
        assert abs(cm.hfov_from_focal(25, 36) - 71.5126) < 0.01
        assert abs(cm.focal_from_hfov(cm.hfov_from_focal(25, 36), 36) - 25) < 1e-6
        ov = cm.boxes_overlap_ratio((0, 0, 10, 10), 10.0, [((5, 5, 15, 15), 5.0)])
        assert abs(ov - 0.25) < 1e-9, ov
        assert cm.boxes_overlap_ratio((0, 0, 10, 10), 10.0, [((5, 5, 15, 15), 20.0)]) == 0.0
    check("capture_math 投影/深度/裁切/遮挡", t_math)

    # ---------- 10. batch_capture：纯逻辑/契约/RT 复用（假 unreal 注入，引擎外可测） ----------
    def _fake_unreal():
        fake = types.ModuleType("unreal")
        fake.TextureRenderTargetFormat = types.SimpleNamespace(
            RTF_RGBA16F="RGBA16F", RTF_RG16F="RG16F", RTF_R8="R8")
        fake.ImageWriteBlueprintLibrary = None
        return fake

    fake_unreal = _fake_unreal()
    sys.modules["unreal"] = fake_unreal
    bc = load_module("dryrun_batch_capture", "engine/Content/Python/batch_capture.py")

    def t_batch_capture():
        src = open(os.path.join(ROOT, "engine/Content/Python/batch_capture.py"),
                   encoding="utf-8").read()
        assert "import numpy" not in src and "import cv2" not in src, "引擎内脚本禁止 numpy/cv2"
        row = {"id": "av_000001", "frame_id": "av_000001", "seed": 20261113,
               "map": scenes[0]["map"], "lighting": lights[1], "targets": targets[1],
               "camera": cameras[1]}
        p1 = bc.plan_targets(row["targets"], bc.task_rng(row))
        p2 = bc.plan_targets(row["targets"], bc.task_rng(row))
        assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True), \
            "同 seed 两次计划不一致"
        assert len(p1) == len(targets[1]), "目标数不等于组合里的 item 数"
        assert len({p["instance_id"] for p in p1}) == len(p1), "实例号重复"
        p3 = bc.plan_targets(row["targets"], bc.task_rng(dict(row, seed=20261114)))
        assert json.dumps(p1, sort_keys=True) != json.dumps(p3, sort_keys=True), \
            "不同 seed 计划却相同"
        state = bc.parse_camera_spec(row["camera"], row["lighting"]["condition"])
        want_hfov = cm.hfov_from_focal(row["camera"]["focal_length"],
                                       row["camera"]["film_back_width"])
        assert abs(state["hfov_deg"] - want_hfov) < 1e-9, "hfov 未由焦距+背板反算"
        m = bc.build_manifest(row, state, bc.object_specs_from_plans(p1), bit_depth=8)
        for k in ("frame_id", "timestamp", "width", "height", "speed", "camera",
                  "lighting", "objects", "image_files"):
            assert k in m, "manifest 缺字段 %s" % k
        assert m["frame_id"] == "av_000001"
        assert abs(m["speed"] - row["camera"]["speed_mps"]) < 1e-9, "speed 未取 camera.speed_mps"
        assert m["image_files"] == {"rgb": "av_000001_rgb.png", "depth": "av_000001_depth.png",
                                    "semantic": "av_000001_sem.png", "bit_depth": 8}
        assert m["objects"], "投影后没有任何目标"
        for ob in m["objects"]:
            x1, y1, x2, y2 = ob["bbox_2d"]
            assert 0 <= x1 < x2 <= m["width"] and 0 <= y1 < y2 <= m["height"], "bbox 越界"
            assert len(ob["bbox_3d"]) == 7 and 0 <= ob["occlusion"] <= 1 and ob["distance"] > 0
        assert bc.png_header_bit_depth(os.path.join(raw8, "av_000000_rgb.png")) == 8
        assert bc.png_header_bit_depth(os.path.join(d16, "raw", "av_000000_rgb.png")) == 16
        # RT 复用：每帧新建会把显存/UObject 表撑爆（1920×1080 RGBA16F 约 16MB/个）
        calls = []

        class _Lib:
            @staticmethod
            def create_render_target2d(world, w, h, fmt):
                calls.append((w, h, fmt))
                return {"rt": len(calls)}

        bc._render_lib = lambda: _Lib()
        bc._editor_world = lambda: None
        a1 = bc._create_render_target("rgb", 320, 180, "RTF_RGBA16F")
        a2 = bc._create_render_target("rgb", 320, 180, "RTF_RGBA16F")
        d1 = bc._create_render_target("depth", 320, 180, "RTF_RG16F")
        assert a1 is a2 and a1 is not d1 and len(calls) == 2, "RT 未按通道复用"
    check("batch_capture 计划/真值契约/位深/RT 复用", t_batch_capture)

    # ---------- 11. batch_capture：落盘兜底 + 主循环（跳过/失败/续跑/退出码） ----------
    def t_batch_save():
        out = os.path.join(base, "rt_export.png")

        class _FallbackLib:
            @staticmethod
            def export_render_target(world, rt, path):
                with open(path, "wb") as f:
                    f.write(b"fake-rt")

        bc._render_lib = lambda: _FallbackLib()
        bc.unreal.ImageWriteBlueprintLibrary = None
        bc._save_render_target({"rt": 1}, out)
        assert open(out, "rb").read() == b"fake-rt", "兜底落盘内容不对"

        class _DeadLib:
            @staticmethod
            def export_render_target(world, rt, path):
                return None

        bc._render_lib = lambda: _DeadLib()
        try:
            bc._save_render_target({"rt": 1}, os.path.join(base, "never.png"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("两条落盘路径都没产出文件时应抛错")
    check("batch_capture RT 落盘兜底/失败抛错", t_batch_save)

    def t_batch_main():
        md = os.path.join(base, "mainloop")
        bc.QUEUE = os.path.join(md, "tasks.jsonl")
        bc.RAW = os.path.join(md, "raw")
        bc.META = os.path.join(md, "meta")
        bc.DONE = os.path.join(md, "done")
        bc.ERRLOG = os.path.join(md, "errors.log")
        os.makedirs(md, exist_ok=True)
        base_row = {"seed": 20261113, "map": scenes[0]["map"], "lighting": lights[1],
                    "targets": targets[1], "camera": cameras[1]}
        rows = [dict(base_row, id="tc_%04d" % i, frame_id="tc_%04d" % i) for i in range(4)]
        with open(bc.QUEUE, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        bc.mark_done("tc_0001")                     # 预置 done → 应跳过
        for p in bc.frame_paths("tc_0002"):          # 预置四件套 → 应跳过
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(b"x")
        bc.load_level = lambda m: True
        bc.spawn_targets = lambda plans: []
        bc.apply_semantic_stencil = lambda pairs: None
        bc.set_lighting = lambda light: light
        bc.place_camera = lambda cam, cond=None: (None, bc.parse_camera_spec(cam, cond))

        def ok_capture(row, cam_actor, cam_state, plans, pairs=None):
            for p in bc.frame_paths(bc.frame_id_of(row)):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "wb").write(b"x")
            return {}

        def mixed_capture(row, cam_actor, cam_state, plans, pairs=None):
            if bc.frame_id_of(row) == "tc_0003":
                raise RuntimeError("boom")
            return ok_capture(row, cam_actor, cam_state, plans, pairs)

        bc.capture_frame = mixed_capture
        with contextlib.redirect_stdout(io.StringIO()):
            rc = bc.main()
        assert rc == 1, "有失败帧时退出码应为 1（实际 %r）" % rc
        errlog = open(bc.ERRLOG, encoding="utf-8", errors="replace").read()
        assert "boom" in errlog, "单帧异常未写错误日志"
        assert os.path.exists(os.path.join(bc.DONE, "tc_0000.done")), "成功帧未写 done"
        assert not os.path.exists(os.path.join(bc.DONE, "tc_0003.done")), "失败帧不应写 done"
        # 重跑：失败帧补跑、已完成/已有输出跳过 → 收口且退出码 0（断点续跑）
        bc.capture_frame = ok_capture
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc2 = bc.main()
        out2 = buf.getvalue()
        assert rc2 == 0, "重跑应成功，实际 %r：%s" % (rc2, out2)
        assert "ok=1 skip=3 fail=0" in out2, "续跑计数不对：%s" % out2[-300:]
        assert os.path.exists(os.path.join(bc.DONE, "tc_0003.done")), "补跑帧未写 done"
    check("batch_capture 主循环（跳过/失败/续跑/退出码）", t_batch_main)

    # ---------- 12. 真值抽检拼图（overlay_boxes） ----------
    def t_overlay():
        out = os.path.join(base, "overlay")
        run(["tools/overlay_boxes.py", "--meta", meta8, "--images", final8,
             "--out", out, "--limit", "6", "--cols", "3"])
        assert os.path.exists(os.path.join(out, "contact_sheet.jpg")), "拼图未生成"
        assert count_files(out, ".jpg") == 7, "逐帧抽检图数量不对（6 帧 + 1 张拼图）"
    check("overlay_boxes 真值抽检拼图", t_overlay)

    print("-" * 60)
    if FAIL:
        print("自检失败：PASS %d / FAIL %d / SKIP %d" % (len(PASS), len(FAIL), len(SKIP)))
        print("失败项：" + "；".join(FAIL))
    else:
        print("自检通过：PASS %d / FAIL 0 / SKIP %d" % (len(PASS), len(SKIP)))
    if a.keep:
        print("保留产物：%s" % base)
    else:
        shutil.rmtree(base, ignore_errors=True)
    return 0 if not FAIL else 1


# 模块级导入放在断言之前，供 7/6/8 节直接调用
# 服务器上可能只同步了部分目录（例如没下发 engine/），缺文件时给一句人话而不是 traceback
for _rel in ("post/postprocess.py", "engine/Content/Python/capture_math.py"):
    if not os.path.exists(os.path.join(ROOT, _rel)):
        sys.stderr.write(
            "自检缺文件：%s\n"
            "说明这台的仓库不完整。把仓库根目录（含 engine/、post/、tools/、config/）整体同步过来再跑：\n"
            "  scp -r engine post tools config tools root@<服务器>:/opt/ue5_synth/\n" % _rel)
        sys.exit(2)
pp = load_module("dryrun_postprocess", "post/postprocess.py")
cm = load_module("dryrun_capture_math", "engine/Content/Python/capture_math.py")

if __name__ == "__main__":
    sys.exit(main())
