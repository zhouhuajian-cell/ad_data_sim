#!/usr/bin/env python3
import fcntl,json,os,runpy,sys,traceback
with open(sys.argv[1],encoding='utf8') as f: cfg=json.load(f)
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
# Local model, no Hub access. All outputs are in the local render directory.
progress=cfg['progress']
with open('/data/render_hybrid/.objtask.lock','w') as lock:
 print('waiting for render queue',flush=True)
 fcntl.flock(lock,fcntl.LOCK_EX)
 print('render started',flush=True)
 # Prevent global incremental index from suppressing this explicit test clip.
 sys.modules['walk_index']=None
 # V4.11 dependencies are missing; explicitly use the existing pipeline6 path.
 import types
 m=types.ModuleType('pipeline_v411')
 class UnavailableRenderer:
  def render(self,*a,**k):raise RuntimeError('V4.11 renderer unavailable')
 class DoubleSignSkip(Exception):pass
 m.SignRealismV411Renderer=UnavailableRenderer;m.DoubleSignSkip=DoubleSignSkip;sys.modules['pipeline_v411']=m
 # Optional helpers are absent on this deployment. The renderer already falls
 # back to fixed grow; SD refinement is not enabled for these tasks.
 import importlib.util,types
 if importlib.util.find_spec('fix_grow') is None:
  m=types.ModuleType('fix_grow');m.grow_for_frame=lambda image,bbox,fallback=1.12:fallback;sys.modules['fix_grow']=m
 if importlib.util.find_spec('sd_refine') is None:
  m=types.ModuleType('sd_refine')
  def unavailable(*args,**kwargs):raise RuntimeError('optional SD refinement is unavailable')
  m.refine=unavailable;m.make_sign_mask=unavailable;m.strength_for_size=unavailable;sys.modules['sd_refine']=m
 sys.argv=['/opt/ue5_synth/tools/batch_walk_curated_preview.py','--images',cfg['images'],
   '--country',cfg['country'],'--out',cfg['out'],'--limit',str(cfg.get('limit') or 10000),
   '--min-sign',str(cfg.get('min_sign') or 20),'--conf',str(cfg.get('conf') or .2),'--signs-dir',cfg['signs_dir'],
   '--jobs-progress',progress,'--vary-class','--legacy-render']
 if cfg.get('pipe6_pad'):sys.argv+=['--pipe6-pad',str(cfg['pipe6_pad'])]
 if cfg.get('pin_class'):sys.argv+=['--pin-class',cfg['pin_class']]
 if cfg.get('pin_material'):sys.argv+=['--pin-material',cfg['pin_material']]
 try:
  runpy.run_path(sys.argv[0],run_name='__main__')
  import glob
  if not glob.glob(os.path.join(cfg['out'],'*_composed.jpg')):raise RuntimeError('没有通过质检的合成图，不能标记完成')
 except SystemExit as e:
  if e.code in (None,0):
   import glob
   if not glob.glob(os.path.join(cfg['out'],'*_composed.jpg')):
    try:
     with open(progress,encoding='utf8') as f:details=json.load(f)
    except (OSError,ValueError):details={'done':0,'total':cfg['frames']}
    details.update(phase='failed',error='没有通过质检的合成图；无标牌 %s，过小或不匹配 %s，渲染失败 %s'%(details.get('skip_no_sign',0),details.get('skip_small',0),details.get('fail',0)))
    with open(progress,'w',encoding='utf8') as f:json.dump(details,f,ensure_ascii=False)
    raise SystemExit(2)
  if e.code not in (None,0):
   with open(progress,'w') as f:json.dump({'phase':'failed','done':0,'total':cfg['frames'],'error':'batch_walk exit '+str(e.code)},f)
  raise
 except Exception as e:
  traceback.print_exc()
  with open(progress,'w') as f:json.dump({'phase':'failed','done':0,'total':cfg['frames'],'error':str(e)},f)
  raise
