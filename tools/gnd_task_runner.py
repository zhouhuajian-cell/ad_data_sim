import fcntl, sys, subprocess, os

# GND-TASK-RUNNER: 与 OBJ 共用渲染队列锁，避免 GPU 抢占
with open('/data/render_hybrid/.objtask.lock', 'w') as lock:
    print('waiting for render queue', flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    print('render started', flush=True)
    r = subprocess.run(
        ['/opt/ad_mining/venv/bin/python', '-u',
         '/opt/ue5_synth/tools/gnd_clip_world.py', '--config', sys.argv[1]],
        env=os.environ)
    sys.exit(r.returncode)
