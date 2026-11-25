import os, re, json, hashlib, subprocess, time, uuid, sys, random
sys.path.insert(0, "/opt/ue5_synth/tools")
from urllib.parse import unquote
ROOT='/data/test_data/real_clips'
OUT='/data/render_hybrid'
ALLOWED={'192.168.20.206/done_data_new':'/mnt/done_data_new'}
CAMS=('cam_st_x2f_front','cam_st_x8d_front','cam_ld_x8d_front','cam_hk_x8b_front')
INDEX='/data/assets_global/OBJ_3D/asset_index_obj3d.json'

VERSION_FILE='/opt/ue5_synth/dashboard/release_versions.json'
COUNTRIES_FILE='/usr/share/iso-codes/json/iso_3166-1.json'
def versions():
    try:
        with open(VERSION_FILE,encoding='utf8') as f:return json.load(f).get('versions',{})
    except (OSError,ValueError):return {}
def country_code(value):
    code=str(value or '').strip().upper()
    try:
        with open(COUNTRIES_FILE,encoding='utf8') as f: valid={x['alpha_2'] for x in json.load(f)['3166-1']}
    except (OSError,KeyError,ValueError): raise ValueError('国家代码库不可用')
    if code not in valid:raise ValueError('请选择有效的国家或地区代码')
    return code

def resolve(raw):
    raw=str(raw or '').strip().strip('"').replace('\\','/')
    if raw.startswith('//'):
        parts=raw[2:].split('/')
        if len(parts)<3: raise ValueError('请填写到具体片段目录')
        key=(parts[0]+'/'+parts[1]).lower()
        if key not in ALLOWED: raise ValueError('暂不支持该网盘')
        base=ALLOWED[key]; p=os.path.join(base,*parts[2:])
    else:
        p=raw; bases=list(ALLOWED.values());base=next((x for x in bases if p==x or p.startswith(x+'/')),None)
        if not base: raise ValueError('仅支持已挂载的数据网盘路径')
    p=os.path.realpath(p);base=os.path.realpath(base)
    if os.path.commonpath([base,p])!=base or not os.path.isdir(p):raise ValueError('数据路径不存在或不在指定网盘')
    return p

def inspect(raw, module='obj'):
    if str(module).lower() == 'gnd':
        import glob, yaml, numpy as np
        p=resolve(raw); cams=[x for x in CAMS if os.path.isdir(os.path.join(p,x))]
        if len(cams)!=1: raise ValueError('需要且只能识别一个前向相机目录，当前：'+','.join(cams))
        frames=sorted(glob.glob(os.path.join(p,cams[0],'*.jpg'))+glob.glob(os.path.join(p,cams[0],'*.png')))
        if not frames: raise ValueError('相机目录没有图像')
        roots=glob.glob(os.path.join(p,'parameter_yaml','*','intrinsic','camera-1.yaml'))
        if len(roots)!=1: raise ValueError('缺少或存在多个相机标定，请检查 parameter_yaml')
        calroot=os.path.dirname(os.path.dirname(roots[0])); exts=glob.glob(os.path.join(calroot,'extrinsic','camera-1_to_body-*.yaml'))
        if len(exts)!=1: raise ValueError('缺少或存在多个相机外参')
        with open(roots[0],encoding='utf8') as f: K=np.asarray(yaml.safe_load(f)['intrinsic'],float)
        with open(exts[0],encoding='utf8') as f: T=np.asarray(yaml.safe_load(f)['transformation'],float)
        if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all(): raise ValueError('标定矩阵无效')
        from bisect import bisect_left
        clouds=sorted(glob.glob(os.path.join(p,'lidar_hs_pandar128_topmiddle','*.las')))
        if len(clouds)<2: raise ValueError('GND 验证需要逐帧 LAS 点云')
        ct=sorted(float(os.path.splitext(os.path.basename(x))[0]) for x in clouds)
        offsets=[]; used=set()
        for im in frames:
            t=float(os.path.splitext(os.path.basename(im))[0]); k=bisect_left(ct,t)
            choices=[j for j in (k-1,k) if 0<=j<len(ct)]; j=min(choices,key=lambda j:abs(ct[j]-t))
            if j in used: raise ValueError('图像与点云不能唯一匹配')
            used.add(j); offsets.append(abs(ct[j]-t))
        if max(offsets)>.05: raise ValueError('图像与点云的时间差超过 50 毫秒')
        return {'ok':True,'path':p,'camera':cams[0],'frames':len(frames),'first':os.path.splitext(os.path.basename(frames[0]))[0], 'calibration':os.path.basename(exts[0]),'max_time_offset_ms':round(max(offsets)*1000,2),'module':'gnd'}
    from label_gate import list_clip_frames,load_frame_labels
    p=resolve(raw);cams=[x for x in CAMS if os.path.isdir(os.path.join(p,x))]
    if len(cams)!=1:raise ValueError('需要且只能识别一个前向相机目录，当前：'+','.join(cams))
    frames=list_clip_frames(p);paired=[(im,x) for im,x in frames if x]
    if not frames or len(paired)!=len(frames):raise ValueError('图像与标注 XML 未完整匹配：%d/%d'%(len(paired),len(frames)))
    first=load_frame_labels(p,cam_name=cams[0],xml_path=paired[0][1])
    if not first or not first.get('camera'):raise ValueError('无法读取相机标定')
    return {'ok':True,'path':p,'camera':cams[0],'frames':len(frames),'first':os.path.basename(frames[0][0])[:-4]}

def assets():
    data=json.load(open(INDEX,encoding='utf-8'));out=[]
    for a in data.get('assets',[]):
        if a.get('true_class') not in ('car','suv','van','pickup'):continue
        if not os.path.isfile(a.get('path','')):continue
        out.append({'uid':a['uid'],'class':a['true_class'],'lwh':a.get('lwh')})
    return out

def start(body):
    body=dict(body);body['country']=country_code(body.get('country'))
    source=inspect(body.get('path'), body.get('module') or 'obj')
    # START-GND-BRANCH: GND 走独立分支
    if str(body.get('module') or 'obj').lower()=='gnd':
        return _start_gnd(body,source)
    available={a['uid']:a for a in assets()}
    if body.get('asset_mode')=='random':
        count=int(body.get('count') or 1)
        if not 1<=count<=3 or len(available)<count:raise ValueError('可用素材不足，数量须为 1 至 3')
        requested=random.SystemRandom().sample(list(available),count)
    else:
        requested=body.get('asset_ids') or ([body.get('asset_id')] if body.get('asset_id') else [])
    if not isinstance(requested,list) or not 1<=len(requested)<=3:raise ValueError('请选择 1 至 3 个 OBJ 素材')
    selected=[]
    for uid in requested:
        asset=available.get(str(uid or ''))
        if not asset:raise ValueError('存在无效的 OBJ 3D 素材')
        selected.append(asset)
    asset=selected[0];uid=asset['uid']
    # API-RIGHT-MODE: straight / turn(左转) / right(右转)
    mode=body.get('trajectory','random')
    if mode not in ('random','follow_road','straight','turn','right'):raise ValueError('轨迹类型无效')
    if len(selected)>1 and mode not in ('random','follow_road'):raise ValueError('多素材请选择随机轨迹或沿自车轨迹')
    motions=random.SystemRandom().sample(['straight','turn','right'],len(selected)) if mode=='random' else [mode if mode in ('turn','right') else 'straight']*len(selected)
    allow_oncoming=bool(body.get('allow_oncoming',False))
    if allow_oncoming and mode!='random':raise ValueError('对向行驶请使用随机轨迹模式')
    specs=[{'uid':a['uid'],'class':a['class'],'motion':m,'direction':random.SystemRandom().choice(['same','oncoming']) if allow_oncoming else 'same'} for a,m in zip(selected,motions)]
    source_path=source['path'];clip=os.path.basename(source_path)
    parent=os.path.dirname(source_path);batch='task_'+hashlib.sha256(parent.encode()).hexdigest()[:16]
    os.makedirs(ROOT,exist_ok=True);dest=os.path.join(ROOT,batch)
    if os.path.lexists(dest):
        if not os.path.islink(dest) or os.path.realpath(dest)!=parent:raise ValueError('数据批次冲突')
    else:os.symlink(parent,dest,target_is_directory=True)
    job='objtask_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=os.path.join(OUT,job);os.makedirs(out)
    cfg={'batch':batch,'clip':clip,'out':out,'params':{'cls':asset['class'],'asset_id':uid,'asset_specs':specs, 'count':len(selected),'dist_min':12,'dist_max':30,'lat':3.5,'lateral_candidates':([0.0,-3.5,3.5] if mode in ('random','follow_road') else [0.0]),'contact_sink_m':0,'occlusion':True,'world_speed_mps':8.0,'relative_speed_mps':1.0,'trajectory_mode':mode}}
    config=os.path.join(out,'config.json');json.dump(cfg,open(config,'w'),ensure_ascii=False)
    meta={'job':job,'source':str(body['path']),'country':str(body.get('country') or ''),'trajectory':mode,'asset_ids':[a['uid'] for a in selected],'asset_specs':specs,'frames':source['frames'],'created':time.time(),'versions':versions()}
    json.dump(meta,open(os.path.join(out,'task.json'),'w'),ensure_ascii=False)
    env=dict(os.environ,PYTHONPATH='/opt/ue5_synth/tools',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    log=open(os.path.join(out,'run.log'),'w');proc=subprocess.Popen(['/opt/ad_mining/venv/bin/python','-u','/opt/ue5_synth/tools/task_obj_runner.py',config],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    meta['pid']=proc.pid;json.dump(meta,open(os.path.join(out,'task.json'),'w'),ensure_ascii=False)
    return {'ok':True,'job':job,'frames':source['frames'],'out':out,'batch':batch,'clip':clip,'module':('gnd' if str(body.get('module') or '').lower()=='gnd' else 'obj')}

def _start_gnd(body,source):
    """GND 片段任务：标线材质 + 车道数 + 渲染模式"""
    mat=str(body.get('mat') or '')
    mode=str(body.get('marking_mode') or 'replace')
    if mode not in ('add','replace'):raise ValueError('标线处理方式无效')
    if int(body.get('lanes') or 3) not in range(1,6):raise ValueError('车道数应为 1 至 5')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',mat) or not os.path.isfile(os.path.join('/data/gnd_assets',mat,'tex.png')):raise ValueError('请选择有效的 GND 标线素材')
    if mat not in ('lane_solid_white','lane_solid_yellow','lane_dashed_white','lane_dashed_yellow'):raise ValueError('当前验证预览仅支持白色/黄色实线和虚线素材')
    source_path=source['path'];clip=os.path.basename(source_path)
    parent=os.path.dirname(source_path)
    batch='gndtask_'+hashlib.sha256(parent.encode()).hexdigest()[:16]
    os.makedirs(ROOT,exist_ok=True);dest=os.path.join(ROOT,batch)
    if os.path.lexists(dest):
        if not os.path.islink(dest) or os.path.realpath(dest)!=parent:raise ValueError('数据批次冲突')
    else:os.symlink(parent,dest,target_is_directory=True)
    job='gndtask_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=os.path.join(OUT,job);os.makedirs(out)
    cfg={'source':source_path,'batch':batch,'clip':clip,'out':out,'params':{
        'mat':mat,
        'marking_mode':mode,
        'lanes':int(body.get('lanes') or 3),
        'render_mode':str(body.get('render_mode') or 'inherit'),
        'dash':int(body.get('dash') or 0),
        'stop':int(body.get('stop') or 0),
        'module':'gnd'}}
    config=os.path.join(out,'config.json');json.dump(cfg,open(config,'w'),ensure_ascii=False)
    meta={'job':job,'module':'gnd','source':str(body['path']),
          'country':str(body.get('country') or ''),'mat':mat,'marking_mode':mode,
          'lanes':cfg['params']['lanes'],'frames':source['frames'],
          'created':time.time(),'versions':versions()}
    json.dump(meta,open(os.path.join(out,'task.json'),'w'),ensure_ascii=False)
    env=dict(os.environ,PYTHONPATH='/opt/ue5_synth/tools',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    log=open(os.path.join(out,'run.log'),'w')
    proc=subprocess.Popen(['/opt/ad_mining/venv/bin/python','-u',
                           '/opt/ue5_synth/tools/gnd_task_runner.py',config],
                          env=env,stdout=log,stderr=subprocess.STDOUT,
                          start_new_session=True)
    log.close()
    meta['pid']=proc.pid
    json.dump(meta,open(os.path.join(out,'task.json'),'w'),ensure_ascii=False)
    return {'ok':True,'job':job,'frames':source['frames'],'out':out,'batch':batch,'clip':clip,'module':('gnd' if str(body.get('module') or '').lower()=='gnd' else 'obj')}

def start_tsr(body, source_path):
    """TSR preview output is always local; input paths are read-only."""
    country=country_code(body.get('country'))
    if not source_path or not os.path.isdir(source_path):raise ValueError('数据路径不存在')
    library=os.path.join('/data/tsr_library_v2',country)
    if not os.path.isdir(library):raise ValueError('该国家暂无 TSR 标牌素材：'+country)
    export=str(body.get('export_dir') or '').strip()
    if export and not os.path.realpath(export).startswith('/data/render/'):
        raise ValueError('输出仅支持服务器本地 /data/render，网盘数据保持只读')
    conf=float(body.get('conf') or .15);min_sign=int(body.get('min_sign') or 16)
    if not .05 <= conf <= .95 or not 12 <= min_sign <= 500:
        raise ValueError('置信度需在 0.05 至 0.95，最小尺寸需在 12 至 500 像素')
    base='/data/render/tsr_v3_10';os.makedirs(base,exist_ok=True)
    job=country+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=os.path.join(base,'walk_'+job);os.makedirs(out)
    progress=os.path.join(base,'walk_'+job+'_progress.json')
    cfg={'images':source_path,'source':source_path,'country':country,'out':out,'progress':progress,
         'frames':0,'limit':int(body.get('limit') or 10000),'min_sign':min_sign,
         'conf':conf,'pin_class':str(body.get('pin_class') or ''),
         'pin_material':str(body.get('pin_material') or '')}
    config=os.path.join(out,'config.json');json.dump(cfg,open(config,'w'),ensure_ascii=False)
    json.dump({'phase':'queued','done':0,'total':0,'ok':0},open(progress,'w'))
    meta={'job':job,'module':'tsr','country':country,'source':source_path,'created':time.time(),
          'frames':0,'renderer':'pipeline6_legacy_preview','versions':versions()}
    mp=os.path.join(out,'task.json');json.dump(meta,open(mp,'w'),ensure_ascii=False)
    log=open(os.path.join(out,'run.log'),'w')
    try:
        proc=subprocess.Popen(['/opt/ad_mining/venv/bin/python','-u','/opt/ue5_synth/tools/tsr_queued_runner.py',config],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
    finally:log.close()
    meta['pid']=proc.pid;json.dump(meta,open(mp,'w'),ensure_ascii=False)
    return {'ok':True,'job':job,'out':out,'progress':progress,'renderer':meta['renderer']}

def _last_error(out):
    try:
        with open(os.path.join(out,'run.log'),encoding='utf8',errors='replace') as f:
            lines=f.read().splitlines()
        if not lines:return '执行日志为空，无法确定失败原因'
        for line in reversed(lines):
            text=line.strip()
            if re.match(r'^(?:[\w.]+(?:Error|Exception)|Exception):',text):
                if 'unable to determine local ground height' in text:return '点云地面拟合失败：无法确定局部地面高度（'+text+'）'
                if 'synthetic targets overlap excessively in image space' in text:return '虚拟目标画面重叠过大，已拦截（'+text+'）'
                if '没有通过质检的合成图' in text:return text
                return text[-700:]
        return '\n'.join(lines[-5:])[-700:]
    except OSError: return '渲染进程已退出，请检查服务日志'

def status(job):
    if not re.fullmatch(r'(obj|gnd)task_\d{8}_\d{6}_[0-9a-f]{6}',str(job or '')):raise ValueError('任务 ID 无效')
    out=os.path.join(OUT,job);meta=json.load(open(os.path.join(out,'task.json')))
    try:
        cfg=json.load(open(os.path.join(out,'config.json'),encoding='utf-8'))
    except Exception:
        cfg={}
    ad=os.path.join(out,'annotations');n=len([x for x in os.listdir(ad) if x.endswith('.json')]) if os.path.isdir(ad) else 0
    mf=os.path.join(out,'trajectory_manifest.json');done=os.path.isfile(mf)
    manifest=json.load(open(mf)) if done else None
    running=os.path.exists('/proc/%d'%meta['pid'])
    if running:
        try:
            with open('/proc/%d/stat'%meta['pid']) as f: running=f.read().split(') ',1)[1][:1]!='Z'
        except OSError: running=False
    empty_render=bool(done and meta.get('module','obj')=='obj' and manifest.get('frames_rendered')==0)
    state='failed' if empty_render else ('done' if done else 'running' if running else 'failed')
    error='本片段没有生成任何仿真目标，请检查轨迹与落点' if empty_render else (None if done or running else _last_error(out))
    return {'ok':True,'job':job,'state':state,'error':error,'frames':meta['frames'],'done':n,'synthetic':manifest.get('frames_rendered') if manifest else None,'out':out,'batch':cfg.get('batch') or meta.get('batch') or '', 'clip':cfg.get('clip') or meta.get('clip') or '', 'module':meta.get('module','obj')}


def list_gnd_tasks(module='gnd'):
    """最近的 GND 片段任务，供不同浏览器查看服务器已有结果。"""
    result=[]
    if not os.path.isdir(OUT): return result
    if module == 'tsr': return list_tsr_tasks()
    if module not in ('gnd','obj'): raise ValueError('无效模块')
    names=sorted((x for x in os.listdir(OUT) if re.fullmatch(module+r'task_\d{8}_\d{6}_[0-9a-f]{6}',x)),reverse=True)[:100]
    for job in names:
        try:
            with open(os.path.join(OUT,job,'task.json'),encoding='utf8') as f: meta=json.load(f)
            v=status(job)
            result.append({'job':job,'state':v['state'],'frames':v['frames'],'done':v['done'],
                'created':meta.get('created'),'country':meta.get('country',''),
                'mode':meta.get('marking_mode','add'),'source':meta.get('source',''), 'batch':v['batch'], 'clip':v['clip'], 'module':module,'versions':meta.get('versions'),'error':v.get('error')})
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return result

def list_tsr_tasks():
    import glob
    base='/data/render/tsr_v3_10'; result=[]
    for mp in sorted(glob.glob(os.path.join(base,'walk_*','task.json')),reverse=True)[:100]:
        try:
            with open(mp,encoding='utf8') as f: meta=json.load(f)
            job=meta['job']; prog=os.path.join(base,'walk_'+job+'_progress.json')
            try:
                with open(prog,encoding='utf8') as f: v=json.load(f)
            except (OSError,ValueError): v={}
            phase=v.get('phase','queued');pid=int(meta.get('pid') or 0)
            alive=False
            if pid:
                try:
                    with open('/proc/%d/stat'%pid) as f: alive=f.read().split(') ',1)[1][:1]!='Z'
                except OSError: pass
            empty=phase=='done' and int(v.get('ok') or 0)==0
            state='failed' if phase=='failed' or empty else ('done' if phase=='done' else ('running' if alive else 'failed'))
            error=None
            if state=='failed':
                if empty:error='扫描完成但没有生成合成图：无标牌 %s，过小或不匹配 %s，渲染失败 %s'%(v.get('skip_no_sign',0),v.get('skip_small',0),v.get('fail',0))
                else:error=v.get('error') or _last_error(os.path.dirname(mp))
            result.append({'job':job,'state':state,'frames':v.get('total') or meta.get('frames',0),
              'done':v.get('done',0),'created':meta.get('created'),'country':meta.get('country',''),
              'source':meta.get('source',''),'module':'tsr','mode':'pipeline6_preview','output_count':v.get('ok',0),'versions':meta.get('versions'),'error':error})
        except (OSError,ValueError,KeyError,TypeError): continue
    return result

def frame(job,index):
    status(job)
    import cv2
    out=os.path.join(OUT,job);ad=os.path.join(out,'annotations')
    names=sorted(x for x in os.listdir(ad) if x.endswith('.json')) if os.path.isdir(ad) else []
    if index < 0 or index >= len(names):raise ValueError('帧不存在')
    data=json.load(open(os.path.join(ad,names[index])));stem=data['frame']
    img=cv2.imread(os.path.join(out,stem+('_gnd.jpg' if status(job).get('module')=='gnd' else '_obj.jpg')))
    if img is None:raise ValueError('图片尚未生成')
    h,w=img.shape[:2];fac=min(1.0,1920.0/w)
    if fac<1:img=cv2.resize(img,(round(w*fac),round(h*fac)),interpolation=cv2.INTER_AREA)
    for obj in data.get('objects',[]):
        source=obj.get('source');col=(30,170,255) if source=='synthetic' else (70,220,70)
        corners=obj.get('corners') or [];bbox=obj.get('bbox2d_sim') or obj.get('bbox2d')
        if len(corners)==8:
            pts=[(round(x*fac),round(y*fac)) for x,y in corners]
            for i,j in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):
                cv2.line(img,pts[i],pts[j],col,2,cv2.LINE_AA)
        elif bbox and len(bbox)==4:
            x1,y1,x2,y2=[round(v*fac) for v in bbox];cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
    ok,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok:raise ValueError('图片编码失败')
    return buf.tobytes()
