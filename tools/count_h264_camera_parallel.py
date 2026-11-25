import os,time,json,queue,threading
root=r'\\192.168.20.90\logs\010-FT-001'
targets={'cam_st_x8d_front','cam_hk_x8b_front'}
q=queue.Queue();q.put((root,None));counts={k:0 for k in targets};folders={k:0 for k in targets};errors=[];state={'dirs':0};lock=threading.Lock()
def worker():
    while True:
        item=q.get()
        if item is None:q.task_done();return
        directory,camera=item; localcounts={k:0 for k in targets};localfolders={k:0 for k in targets};problems=[]
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            name=entry.name.lower();nextcamera=name if name in targets else camera
                            if name in targets:localfolders[name]+=1
                            q.put((entry.path,nextcamera))
                        elif camera and entry.name.lower().endswith('.h264') and entry.is_file(follow_symlinks=False):localcounts[camera]+=1
                    except OSError as e:problems.append({'path':entry.path,'error':str(e)})
        except OSError as e:problems.append({'path':directory,'error':str(e)})
        with lock:
            for k in targets:counts[k]+=localcounts[k];folders[k]+=localfolders[k]
            errors.extend(problems);state['dirs']+=1
        q.task_done()
for _ in range(8):threading.Thread(target=worker,daemon=True).start()
def progress():
    while True:
        time.sleep(25)
        with lock:print(json.dumps({'dirs':state['dirs'],'pending':q.qsize(),'counts':counts,'errors':len(errors)},ensure_ascii=False),flush=True)
threading.Thread(target=progress,daemon=True).start();q.join()
print(json.dumps({'root':root,'counts':counts,'camera_directories':folders,'total':sum(counts.values()),'visited_dirs':state['dirs'],'errors':errors},ensure_ascii=False),flush=True)
