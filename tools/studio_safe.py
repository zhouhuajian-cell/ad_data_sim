"""Isolated preview/publish workflow for the OBJ asset studio."""
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid

ROOT='/data/assets_global/OBJ_3D'
STAGE=os.path.join(ROOT,'studio_staging')
INDEX=os.path.join(ROOT,'asset_index_obj3d.json')
CANON=os.path.join(ROOT,'canonical')
BLENDER='/data/blender-4.2.23-linux-x64/blender'
TOOL='/opt/ue5_synth/tools/external_asset_canonicalize_blender.py'
THUMB='/opt/ue5_synth/tools/render_thumb.py'
LOCK=os.path.join(ROOT,'.studio_index.lock')
CLASSES={'car','suv','pickup','van','truck','bus','tricycle','motorcycle',
         'bicycle','rickshaw','low_speed_ev','scooter','skateboard',
         'pedestrian','rider','child','camel','kangaroo','animal','cone','barrier'}


def _id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-f0-9]{32}',value):
        raise ValueError('invalid task ID')
    return value


def _read(path):
    with open(path,encoding='utf-8') as f:return json.load(f)


def upload(raw):
    if len(raw)<12 or raw[:4]!=b'glTF' or int.from_bytes(raw[8:12],'little')!=len(raw):
        raise ValueError('invalid GLB file header or size')
    task=uuid.uuid4().hex
    folder=os.path.join(STAGE,task)
    os.makedirs(folder,exist_ok=False)
    with open(os.path.join(folder,'source.glb'),'xb') as f:f.write(raw)
    return {'ok':True,'task':task,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def preview(body):
    task=_id(body.get('task'));folder=os.path.join(STAGE,task)
    source=os.path.join(folder,'source.glb')
    if not os.path.isfile(source):raise ValueError('upload not found')
    cls=str(body.get('cls',''))
    axis=str(body.get('axis','x'))
    if cls not in CLASSES:raise ValueError('请选择有效的素材类别')
    if axis not in ('x','y','z'):raise ValueError('尺寸基准轴无效')
    try: yaw=float(body.get('yaw',0))
    except (TypeError,ValueError):raise ValueError('请输入有效的朝向角度')
    if not math.isfinite(yaw):raise ValueError('请输入有效的朝向角度')
    mode=body.get('dimension_mode','proportional')
    values=None
    if mode == 'independent':
        dimensions=body.get('dimensions')
        if not isinstance(dimensions,list) or len(dimensions)!=3:
            raise ValueError('长宽高必须全部填写')
        try: values=[float(v) for v in dimensions]
        except (TypeError,ValueError):raise ValueError('长宽高必须全部填写数字')
        if any(not math.isfinite(v) or v<.05 or v>100 for v in values):
            raise ValueError('长宽高须在 0.05 至 100 米之间')
        size=values['xyz'.index(axis)]
    elif mode == 'proportional':
        try: size=float(body.get('size'))
        except (TypeError,ValueError):raise ValueError('请填写有效的等比例基准尺寸')
        if not math.isfinite(size) or not .05<=size<=100:
            raise ValueError('基准尺寸须在 0.05 至 100 米之间')
    else: raise ValueError('无效的尺寸调整方式')
    version=uuid.uuid4().hex; dest=os.path.join(folder,version);os.makedirs(dest,exist_ok=False)
    model=os.path.join(dest,'model.glb');meta=os.path.join(dest,'meta.json');thumb=os.path.join(dest,'thumb.png')
    cmd=[BLENDER,'-b','--python-exit-code','7','--python',TOOL,'--',source,model,meta,str(yaw),str(size),axis]
    if values is not None:cmd.extend(str(v) for v in values)
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
    if proc.returncode or not os.path.isfile(model) or not os.path.isfile(meta):raise RuntimeError('normalization failed: '+(proc.stderr or proc.stdout)[-700:])
    render=subprocess.run([BLENDER,'-b','--python-exit-code','7','--python',THUMB,'--',model,thumb,'512'],capture_output=True,text=True,timeout=300)
    if render.returncode or not os.path.isfile(thumb):raise RuntimeError('preview failed: '+(render.stderr or render.stdout)[-700:])
    data=_read(meta);data.update({'uid':'ext_'+task[:12], 'true_class':cls,
        'source_description':str(body.get('source',''))[:500],
        'license':str(body.get('license',''))[:500],
        'model_sha256':_sha(model),'task':task,'version':version})
    with open(meta,'w',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    return {'ok':True,'task':task,'version':version,'meta':data,
            'thumb_url':'/api/studio/v2/thumb?task='+task+'&version='+version}


def _sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def thumb(task,version):
    return os.path.join(STAGE,_id(task),_id(version),'thumb.png')


def model_path(task,version):
    return os.path.join(STAGE,_id(task),_id(version),'model.glb')


def publish(body):
    task=_id(body.get('task'));version=_id(body.get('version'))
    dest=os.path.join(STAGE,task,version);model=os.path.join(dest,'model.glb');meta=_read(os.path.join(dest,'meta.json'))
    if meta.get('model_sha256')!=_sha(model):raise ValueError('preview model changed')
    uid=meta['uid'];path=os.path.join(CANON,uid+'.glb');png=os.path.join(CANON,uid+'.png')
    os.makedirs(CANON,exist_ok=True)
    with open(LOCK,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        idx=_read(INDEX)
        if not isinstance(idx,dict) or not isinstance(idx.get('assets'),list):raise ValueError('invalid index format')
        if os.path.exists(path) or os.path.exists(png) or any(x.get('uid')==uid for x in idx['assets']):raise ValueError('asset UID already exists')
        entry={'uid':uid,'true_class':meta['true_class'],'yaw_fix_deg':meta['yaw_fix_deg'],
               'lwh':meta['lwh'],'path':path,'src':os.path.join(STAGE,task,'source.glb'),
               'source':meta.get('source_description',''),'license':meta.get('license',''),
               'model_sha256':meta['model_sha256'],'training_eligible':False}
        tmp=INDEX+'.studio.'+uuid.uuid4().hex+'.tmp'
        created=[]
        try:
            with open(path,'xb') as f,open(model,'rb') as src:shutil.copyfileobj(src,f)
            created.append(path)
            with open(png,'xb') as f,open(os.path.join(dest,'thumb.png'),'rb') as src:shutil.copyfileobj(src,f)
            created.append(png)
            idx['assets'].append(entry)
            with open(tmp,'x',encoding='utf-8') as f:json.dump(idx,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
            os.replace(tmp,INDEX)
        except Exception:
            for p in [tmp]+created:
                if os.path.isfile(p):os.unlink(p)
            raise
    return {'ok':True,'uid':uid,'entry':entry}
