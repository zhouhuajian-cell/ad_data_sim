import io

P = "/opt/ue5_synth/dashboard/server.py"
s = io.open(P, encoding="utf-8").read()

if "/api/real/clip_play" in s:
    print("已存在")
    raise SystemExit(0)

anchor = '        # [CALIB] 列出底图（供标定面板选图）'
if anchor not in s:
    print("ERROR: 锚点")
    raise SystemExit(2)

NEW = '''        # [CLIP-PLAY] 列出某 clip 的可播放帧（原图 + 仿真图成对）
        if u.path == "/api/real/clip_play":
            batch = (q.get("batch") or [""])[0]
            clip = (q.get("clip") or [""])[0]
            module = (q.get("module") or ["obj"])[0]
            if not batch or "/" in batch or ".." in batch:
                return self._send(400, {"error": "bad batch"})
            REAL_ROOT = "/data/test_data/real_clips"
            VD = "/data/render/verify_batch"

            # 原图帧
            cd = os.path.join(REAL_ROOT, batch, clip, "cam_st_x2f_front")
            if clip and not os.path.isdir(cd):
                return self._send(404, {"error": "clip 不存在"})
            if not clip:
                # 取第一个 clip
                bp = os.path.join(REAL_ROOT, batch)
                subs = [d for d in os.listdir(bp)
                        if os.path.isdir(os.path.join(bp, d, "cam_st_x2f_front"))]
                if not subs:
                    return self._send(404, {"error": "无 clip"})
                clip = sorted(subs)[0]
                cd = os.path.join(bp, clip, "cam_st_x2f_front")

            stems = sorted(os.path.splitext(f)[0]
                           for f in os.listdir(cd) if f.lower().endswith(".jpg"))

            suffix = "_composed.jpg" if module == "obj" else "_gnd.jpg"
            sd = os.path.join(VD, batch)

            frames = []
            for st in stems:
                sim_p = os.path.join(sd, st + suffix)
                if not os.path.isfile(sim_p):
                    continue
                frames.append({
                    "stem": st,
                    "orig": "/api/real/image?batch=" + batch +
                            "&clip=" + clip + "&stem=" + st,
                    "sim": "/api/real/verify_image?path=" +
                           batch + "/" + st + suffix,
                })

            return self._send(200, {"batch": batch, "clip": clip,
                                    "module": module,
                                    "frames": frames,
                                    "n": len(frames)})

'''

s = s.replace(anchor, NEW + anchor, 1)
io.open(P, "w", encoding="utf-8").write(s)
print("已加 /api/real/clip_play")
