import fcntl,sys,subprocess,os
with open('/data/render_hybrid/.objtask.lock','w') as lock:
 print('waiting for render queue',flush=True)
 fcntl.flock(lock,fcntl.LOCK_EX)
 print('render started',flush=True)
 r=subprocess.run(['/opt/ad_mining/venv/bin/python','-u','/opt/ue5_synth/tools/render_obj_task.py','--config',sys.argv[1]],env=os.environ)
 sys.exit(r.returncode)
