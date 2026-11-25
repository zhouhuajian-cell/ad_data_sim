import os, sys, time, json

root = r'\\192.168.20.90\logs\010-FT-001'
target = {'cam_st_x8d_front', 'cam_hk_x8b_front'}
stack = [(root, None)]
counts = {k: 0 for k in target}
folders = {k: 0 for k in target}
errors = []
visited = 0
last = time.monotonic()
while stack:
    directory, camera = stack.pop()
    visited += 1
    try:
        with os.scandir(directory) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        name = entry.name.lower()
                        next_camera = name if name in target else camera
                        if name in target:
                            folders[name] += 1
                        stack.append((entry.path, next_camera))
                    elif camera and entry.name.lower().endswith('.h264') and entry.is_file(follow_symlinks=False):
                        counts[camera] += 1
                except OSError as exc:
                    errors.append({'path':entry.path,'error':str(exc)})
    except OSError as exc:
        errors.append({'path':directory,'error':str(exc)})
    if time.monotonic() - last >= 20:
        print(json.dumps({'visited_dirs':visited,'pending_dirs':len(stack),'counts':counts,'errors':len(errors)},ensure_ascii=False),flush=True)
        last=time.monotonic()
print(json.dumps({'root':root,'counts':counts,'camera_directories':folders,'total':sum(counts.values()),
                  'visited_dirs':visited,'errors':errors},ensure_ascii=False),flush=True)
