"""Staged import for GND markings and TSR signs; never overwrites library files."""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid

ROOT='/data/assets_global/OBJ_3D/studio_staging'
WORKER='/opt/ue5_synth/tools/studio_2d_worker.py'
PY='/opt/ad_mining/venv/bin/python'
GND='/data/gnd_assets'
TSR='/data/tsr_library_v2'


def _id(x):
    if not isinstance(x,str) or not re.fullmatch(r'[a-f0-9]{32}',x):raise ValueError('invalid task ID')
    return x


def _sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def upload(raw,module):
    if module not in ('gnd','tsr'):raise ValueError('invalid module')
    if not raw:raise ValueError('empty file')
    task=uuid.uuid4().hex;folder=os.path.join(ROOT,task);os.makedirs(folder,exist_ok=False)
    source=os.path.join(folder,'source_2d');preview=os.path.join(folder,'preview.png');meta=os.path.join(folder,'inspection.json')
    with open(source,'xb') as f:f.write(raw)
    run=subprocess.run([PY,WORKER,source,preview,meta],capture_output=True,text=True,timeout=120)
    if run.returncode:raise ValueError('素材解析失败: '+run.stderr[-500:])
    info=json.load(open(meta,encoding='utf-8'));info.update(module=module,task=task,sha256=_sha(preview))
    if module=='gnd' and info['alpha_min']==255:raise ValueError('GND 标线必须包含透明背景；请使用带 Alpha 的 PNG/WebP 或透明背景 SVG')
    if info['alpha_max']==0:raise ValueError('素材完全透明')
    with open(meta,'w',encoding='utf-8') as f:json.dump(info,f,ensure_ascii=False,indent=2)
    return {'ok':True,'task':task,'inspection':info,'thumb_url':'/api/studio/v2/2d-thumb?task='+task}


def preview(task):return os.path.join(ROOT,_id(task),'preview.png')


def publish(body):
    task=_id(body.get('task'));folder=os.path.join(ROOT,task)
    with open(os.path.join(folder,'inspection.json'),encoding='utf-8') as f:info=json.load(f)
    img=preview(task)
    if _sha(img)!=info.get('sha256'):raise ValueError('preview changed')
    country=str(body.get('country') or '').upper().strip()
    name=str(body.get('name') or '').strip().lower()
    if not re.fullmatch(r'[A-Z]{2}|GLOBAL',country):raise ValueError('国家请填写两位代码或 GLOBAL')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,47}',name):raise ValueError('类别仅支持英文、数字、下划线和短横线')
    source=str(body.get('source') or '')[:500];license_=str(body.get('license') or '')[:500]
    if info['module']=='gnd':
        length=float(body.get('length_m') or 0);width=float(body.get('width_m') or 0)
        if not all(math.isfinite(v) and .01<=v<=200 for v in (length,width)):raise ValueError('请填写有效的实际长度和宽度（米）')
        key='ext_'+country.lower()+'_'+name+'_'+task[:8]
        dest=os.path.join(GND,key)
        os.makedirs(dest,exist_ok=False)
        try:
            with open(img,'rb') as src,open(os.path.join(dest,'tex.png'),'xb') as out:shutil.copyfileobj(src,out)
            meta={'key':key,'label':name,'country':country,'size':[width*1000,length*1000],
                  'size_m':[length,width],'horizontal':False,'source':source,'license':license_,
                  'source_task':task,'source_format':info['format'],'training_eligible':False,'files':['tex.png']}
            with open(os.path.join(dest,'meta.json'),'x',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,indent=2)
        except Exception:
            raise RuntimeError('素材写入失败，保留隔离目录供检查: '+dest)
        return {'ok':True,'module':'gnd','key':key,'path':dest}
    if info['module']!='tsr':raise ValueError('invalid module')
    if country=='GLOBAL':raise ValueError('TSR 请填写具体国家两位代码')
    dest=os.path.join(TSR,country);os.makedirs(dest,exist_ok=True)
    # 现有 TSR 库通过 _v 后缀聚合同类别变体。
    filename=name+'_vstudio_'+task[:8]+'.png';path=os.path.join(dest,filename)
    with open(img,'rb') as src,open(path,'xb') as out:shutil.copyfileobj(src,out)
    with open(os.path.join(folder,'published.json'),'x',encoding='utf-8') as f:
        json.dump({'module':'tsr','country':country,'name':name,'path':path,'source':source,
                   'license':license_,'source_format':info['format'],'training_eligible':False},f,ensure_ascii=False,indent=2)
    return {'ok':True,'module':'tsr','country':country,'file':filename}
