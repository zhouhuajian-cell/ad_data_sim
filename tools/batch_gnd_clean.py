import sys, os, glob, json, time, shutil, subprocess
sys.path.insert(0, '/opt/ue5_synth/tools')

OUT = "/data/render/gnd_clean"
PY = "/opt/ad_mining/venv/bin/python"
TOOLS = "/opt/ue5_synth/tools"

def list_gnd():
    root = "/data/test_data/gnd_train"
    out = []
    for vd in sorted(os.listdir(root)):
        vp = os.path.join(root, vd)
        if not os.path.isdir(vp): continue
        for cl in sorted(os.listdir(vp)):
            cp = os.path.join(vp, cl)
            if os.path.isdir(os.path.join(cp, "img_ori")):
                out.append((vd, cl))
    return out

def merge(src, dst):
    for sub in ("frames", "annotations"):
        s, d = os.path.join(src, sub), os.path.join(dst, sub)
        if not os.path.isdir(s): continue
        os.makedirs(d, exist_ok=True)
        for fn in os.listdir(s):
            sp = os.path.join(s, fn)
            if os.path.isfile(sp): shutil.copy2(sp, os.path.join(d, fn))

clips = list_gnd()
print("GND clip: %d" % len(clips))
os.makedirs(OUT, exist_ok=True)
tmp = "/tmp/gnd_one"
cfg = '{"gnd_mode":"clean","lanes":3,"dash":true,"edge":0.2,"width":0.15,"lane_type":"lane_dashed_white"}'
ok = fail = 0
t0 = time.time()

for i, (vd, cl) in enumerate(clips):
    if os.path.isdir(tmp): shutil.rmtree(tmp)
    cmd = [PY, os.path.join(TOOLS,"pipeline_sim.py"), "--batch", vd, "--clip", cl,
           "--module","gnd","--frames","0","--out",tmp,"--config",cfg]
    env = dict(os.environ); env["PYTHONPATH"]=TOOLS
    env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        n = len(glob.glob(os.path.join(tmp,"frames","*_sim.jpg")))
        if r.returncode==0 and n>0:
            merge(tmp, OUT); ok += 1
            print("  [%d/%d] %-26s %-22s %d帧" % (i+1,len(clips),vd[:26],cl[:22],n))
        else:
            fail += 1
            e = (r.stderr or r.stdout or "")[-100:]
            print("  [%d/%d] %-26s %-22s 失败 %s" % (i+1,len(clips),vd[:26],cl[:22],e.replace("\n"," ")[:50]))
    except subprocess.TimeoutExpired:
        fail += 1; print("  [%d/%d] %s 超时" % (i+1,len(clips),cl[:22]))
    except Exception as e:
        fail += 1; print("  [%d/%d] %s 异常 %s" % (i+1,len(clips),cl[:22],str(e)[:40]))

dt = time.time()-t0
nf = len(glob.glob(os.path.join(OUT,"frames","*_sim.jpg")))
na = len(glob.glob(os.path.join(OUT,"annotations","*.json")))
print("\n=== 完成 %.1f 分钟 ===" % (dt/60))
print("成功 %d / 失败 %d" % (ok, fail))
print("产出: %d 帧 / %d 标注" % (nf, na))
if os.path.isdir(tmp): shutil.rmtree(tmp)
