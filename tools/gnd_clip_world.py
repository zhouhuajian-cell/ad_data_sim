#!/usr/bin/env python3
"""GND clip preview: LiDAR odometry, world-fixed markings, semantic occlusion.
Source data is read-only. Outputs are validation artifacts, never full GT.
"""
import argparse, glob, json, os, sys, time, re
from pathlib import Path
import cv2
import numpy as np
import yaml
import laspy
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
sys.path.insert(0, '/opt/ue5_synth/tools')
from gnd_asset_render import load_asset_tex, load_asset_meta
from gnd_world_style import prepare_texture, composite

CAMS=('cam_st_x2f_front','cam_st_x8d_front','cam_ld_x8d_front','cam_hk_x8b_front')

def timestamps(paths):
    return np.array([float(Path(p).stem) for p in paths], dtype=np.float64)

def files_for_clip(src):
    dirs=[os.path.join(src,c) for c in CAMS if os.path.isdir(os.path.join(src,c))]
    if len(dirs)!=1: raise ValueError('expected one front camera directory')
    imgs=sorted(glob.glob(os.path.join(dirs[0],'*.jpg'))+glob.glob(os.path.join(dirs[0],'*.png')))
    clouds=sorted(glob.glob(os.path.join(src,'lidar_hs_pandar128_topmiddle','*.las')))
    if len(imgs)<2 or len(clouds)<2: raise ValueError('need >=2 JPEG frames and timestamped LAS scans')
    ct=timestamps(clouds); it=timestamps(imgs); inds=np.abs(it[:,None]-ct[None,:]).argmin(axis=1)
    offsets=np.abs(it-ct[inds]);
    if offsets.max()>0.05 or len(set(inds))!=len(imgs): raise ValueError('camera/LiDAR synchronization exceeds 50 ms or scan pairing is ambiguous')
    return imgs,[clouds[i] for i in inds],offsets

def calibration(src):
    ips=glob.glob(os.path.join(src,'parameter_yaml','*','intrinsic','camera-1.yaml'))
    if len(ips)!=1: raise ValueError('expected one camera calibration')
    root=os.path.dirname(os.path.dirname(ips[0])); ex=glob.glob(os.path.join(root,'extrinsic','camera-1_to_body-*.yaml'))
    if len(ex)!=1: raise ValueError('expected one camera to body calibration')
    with open(ips[0]) as f: intr=yaml.safe_load(f)
    with open(ex[0]) as f: ext=yaml.safe_load(f)
    K=np.array(intr['intrinsic'],dtype=np.float64); T=np.array(ext['transformation'],dtype=np.float64)
    D=np.array(intr.get('distortion') or [],dtype=np.float64).ravel()
    if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all(): raise ValueError('invalid calibration')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=.001) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=.001): raise ValueError('invalid rotation')
    return K,D,T

def read_cloud(path):
    a=laspy.read(path); xyz=np.column_stack((a.x,a.y,a.z)).astype(np.float64)
    xyz=xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz)<5000: raise ValueError('insufficient LiDAR points: '+path)
    return xyz

def odometry(paths):
    conf=KISSConfig(); conf.data.deskew=False; conf.mapping.voxel_size=1.;conf.registration.max_num_threads=4
    icp=KissICP(conf); poses=[]; ground=None
    for i,p in enumerate(paths):
        xyz=read_cloud(p)
        if i==0:
            m=(xyz[:,0]>4)&(xyz[:,0]<25)&(np.abs(xyz[:,1])<3)&(xyz[:,2]>-1)&(xyz[:,2]<.25)
            if m.sum()<100: raise ValueError('unable to determine local ground height')
            ground=float(np.percentile(xyz[m,2],50))
        icp.register_frame(xyz,np.empty(0,dtype=np.float64))
        pose=np.array(icp.last_pose)
        if not np.isfinite(pose).all(): raise ValueError('nonfinite odometry')
        if poses and np.linalg.norm(pose[:3,3]-poses[-1][:3,3])>8: raise ValueError('odometry jump greater than 8m per frame')
        poses.append(pose)
        if i%20==0: print('ODOMETRY %d/%d'%(i+1,len(paths)),flush=True)
    return poses, ground

class Segmenter:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.torch=torch; self.dev='cuda' if torch.cuda.is_available() else 'cpu'
        p='/data/models/mask2former_mapillary'; self.proc=AutoImageProcessor.from_pretrained(p,local_files_only=True)
        self.model=Mask2FormerForUniversalSegmentation.from_pretrained(p,local_files_only=True).to(self.dev).eval()
        names=self.model.config.id2label
        self.road=[int(i) for i,n in names.items() if n.lower() in ('road','service lane','bike lane','lane marking - general','lane marking - crosswalk')]
        self.marking=[int(i) for i,n in names.items() if 'lane marking' in n.lower()]
        if not self.road or not self.marking: raise ValueError('semantic road or marking classes missing')
    def masks(self,img):
        t=self.torch
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w=img.shape[:2]
        with t.inference_mode():
            ins=self.proc(images=rgb,return_tensors='pt').to(self.dev)
            outputs=self.model(**ins)
            sem=self.proc.post_process_semantic_segmentation(outputs,target_sizes=[(h,w)])[0].cpu().numpy()
            class_probs=outputs.class_queries_logits.softmax(dim=-1)[...,:-1]
            mask_probs=outputs.masks_queries_logits.sigmoid()
            mark=(class_probs[:,:,self.marking].sum(dim=-1)[:,:,None,None]*mask_probs).sum(dim=1,keepdim=True)
            mark=t.nn.functional.interpolate(mark,size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        road=np.isin(sem,self.road).astype(np.uint8)
        erase=((mark>0.15)&(road>0)).astype(np.uint8)*255
        erase=cv2.dilate(erase,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        erase=cv2.bitwise_and(erase,road*255)
        return road,erase

class RoadInpainter:
    def __init__(self):
        os.environ['LAMA_MODEL']='/root/.cache/torch/hub/checkpoints/big-lama.pt'
        if not os.path.isfile(os.environ['LAMA_MODEL']): raise RuntimeError('local LaMa model is unavailable')
        from simple_lama_inpainting import SimpleLama
        self.lama=SimpleLama()
    def apply(self,img,mask):
        from PIL import Image
        if not np.any(mask):return img.copy()
        yy,xx=np.where(mask>0);h,w=mask.shape
        x0=max(0,int(xx.min())-48);x1=min(w,int(xx.max())+49)
        y0=max(0,int(yy.min())-48);y1=min(h,int(yy.max())+49)
        crop=img[y0:y1,x0:x1]; m=mask[y0:y1,x0:x1]
        clean=np.asarray(self.lama(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),Image.fromarray(m)))
        clean=cv2.cvtColor(clean,cv2.COLOR_RGB2BGR)[:crop.shape[0],:crop.shape[1]]
        out=img.copy();region=out[y0:y1,x0:x1]; region[m>0]=clean[m>0]
        return out

class Cv2Inpainter:
    def apply(self, img, mask):
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA) if np.any(mask) else img.copy()

def warp_quad(rgb,alpha,tex,pts):
    h,w=alpha.shape; th,tw=tex.shape[:2]
    src=np.float32([[0,th-1],[0,0],[tw-1,0],[tw-1,th-1]])
    H=cv2.getPerspectiveTransform(src,pts.astype(np.float32))
    color=tex[:,:,:3] if tex.ndim==3 else cv2.cvtColor(tex,cv2.COLOR_GRAY2BGR)
    a=tex[:,:,3] if tex.ndim==3 and tex.shape[2]==4 else np.full((th,tw),255,np.uint8)
    wa=cv2.warpPerspective(a,H,(w,h),flags=cv2.INTER_LINEAR)
    mask=wa>alpha
    if np.any(mask):
        wr=cv2.warpPerspective(color,H,(w,h),flags=cv2.INTER_LINEAR)
        alpha[mask]=wa[mask];rgb[mask]=wr[mask]

def build_reference_path(poses, ground):
    p=np.array([q[:3,3] for q in poses],dtype=np.float64)
    ds=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)
    steps=np.r_[0.,np.cumsum(ds)]
    keep=np.r_[True,np.diff(steps)>0.01]; p=p[keep];steps=steps[keep]
    if len(steps)<3 or steps[-1]<2: raise ValueError('not enough ego motion to construct a road-aligned preview')
    if np.any(ds>8.):raise ValueError('trajectory discontinuity')
    # Estimate tangent from a symmetric 10m baseline, reducing frame-to-frame yaw noise.
    def raw_at(s):
        return np.array([np.interp(s,steps,p[:,k]) for k in range(3)])
    def point(s):
        start=raw_at(max(0.,min(s,steps[-1])))
        d=raw_at(min(steps[-1],max(0.,s)+5.))-raw_at(max(0.,min(s,steps[-1])-5.))
        d[2]=0.; d/=max(np.linalg.norm(d),1e-8)
        if s>steps[-1]:start[:2]+=(s-steps[-1])*d[:2]
        start[2]+=ground
        return start,d
    return point, float(steps[-1]+65.)

def paint(img,K,Tcb,Twb,tex,meta,pp,path,smax):
    h,w=img.shape[:2]; alpha=np.zeros((h,w),np.uint8);rgb=np.zeros((h,w,3),np.uint8)
    # +X is forward in the validated body-coordinate LAS; the asset is fixed in frame-0 world.
    Tcw=np.linalg.inv(Twb@Tcb); lanes=int(pp.get('lanes',3)); half=3.5*lanes/2
    if not 1<=lanes<=5:raise ValueError('lane count must be 1..5')
    ys=[-half+3.5*i for i in range(lanes+1)]
    width=float(pp.get('width',.15));edge=float(pp.get('edge',.20))
    if not (.03<=width<=2 and .03<=edge<=2):raise ValueError('invalid marking width')
    wear=float(pp.get('wear',.35));dirt=float(pp.get('dirt',.25))
    placements=[]; items=pp.get('gnd_items') or []
    if pp.get('gnd_spec_on'):
        if not items:raise ValueError('指定素材已启用但列表为空')
        if len(items)>20:raise ValueError('指定素材最多20项')
        for item in items:
            key=str(item.get('asset') or '')
            if not re.fullmatch(r'[A-Za-z0-9_-]+',key):raise ValueError('invalid asset key')
            asset=load_asset_tex(key);asset_meta=load_asset_meta(key)
            if asset is None or asset_meta is None:raise ValueError('asset unavailable: '+key)
            x=float(item.get('x',10));y=float(item.get('y',0));length=float(item.get('len_m',3));wd=float(item.get('width_m',.2))
            if not np.isfinite([x,y,length,wd]).all() or not (0<=x<=smax and abs(y)<=30 and .05<=length<=100 and .03<=wd<=30):raise ValueError('指定素材位置或尺寸无效: '+key)
            asset=prepare_texture(asset,wear,dirt,key)
            placements.append((x,y,length,wd,asset,False))
    else:
        key=str(pp['mat']);is_horizontal=bool(meta.get('horizontal'));is_arrow='arrow' in key
        if is_horizontal or is_arrow:
            for x in np.arange(8.,smax,6. if is_arrow else 12.):
                placements.append((x,0.,3. if is_arrow else .45,1. if is_arrow else 3.5,tex,False))
        else:
            for i,y in enumerate(ys):
                outer=i in (0,len(ys)-1)
                placements.append((8.,y,max(0.,smax-8.),edge if outer else width,tex,bool(pp.get('dash')) and not outer))
    if pp.get('stop'):
        stop=load_asset_tex('stop_line')
        if stop is None:raise ValueError('stop line asset unavailable')
        placements.append((25.,0.,.45,3.5*lanes,prepare_texture(stop,wear,dirt,'stop_line'),False))
    for start,y,length,wd,asset,dashed in placements:
        # Split long surfaces in world coordinates; wear remains attached to the texture.
        for x in np.arange(start,min(start+length,smax),4.):
            if dashed and int((x-start)//4)%2:continue
            piece=min(4.,start+length-x,smax-x)
            if piece<=0:continue
            tile_tex=asset
            p0,d0=path(x);p1,d1=path(x+piece)
            n0=np.array([-d0[1],d0[0],0.]);n1=np.array([-d1[1],d1[0],0.])
            v0=p0+n0*(y+wd/2);v1=p1+n1*(y+wd/2)
            v2=p1+n1*(y-wd/2);v3=p0+n0*(y-wd/2)
            corners=np.c_[np.stack((v0,v1,v2,v3)),np.ones(4)]
            cam=(Tcw@corners.T).T[:,:3];z=cam[:,2]
            if np.any(z<1.): continue
            pr=(K@cam.T).T; uv=pr[:,:2]/pr[:,2:3]
            if not np.isfinite(uv).all() or np.max(np.abs(uv))>100000:continue
            if uv[:,0].max()<0 or uv[:,0].min()>w or uv[:,1].max()<0 or uv[:,1].min()>h:continue
            warp_quad(rgb,alpha,tile_tex,uv)
    return rgb,alpha

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    with open(args.config,encoding='utf8') as f:cfg=json.load(f)
    src=cfg.get('source') or os.path.join('/data/test_data/real_clips',cfg['batch'],cfg['clip']);out=cfg['out']; pp=cfg['params']
    imgs,clouds,dt=files_for_clip(src)
    if args.limit: imgs,clouds,dt=imgs[:args.limit],clouds[:args.limit],dt[:args.limit]
    if not re.fullmatch(r'[A-Za-z0-9_-]+',str(pp.get('mat') or '')):raise ValueError('invalid material key')
    K,D,Tcb=calibration(src); tex=load_asset_tex(pp['mat']);meta=load_asset_meta(pp['mat'])
    if tex is None or meta is None:raise ValueError('asset unavailable')
    # The checkbox controls dash gaps in world space, rather than layering gaps
    # on top of a pre-dashed texture.
    if pp['mat'] in ('lane_dashed_white','lane_dashed_yellow'):
        solid=load_asset_tex(pp['mat'].replace('lane_dashed_','lane_solid_'))
        if solid is None:raise ValueError('solid lane texture unavailable')
        tex=solid
    tex=prepare_texture(tex,float(pp.get('wear',.35)),float(pp.get('dirt',.25)),str(pp['mat']))
    os.makedirs(out,exist_ok=True); ad=os.path.join(out,'annotations');os.makedirs(ad,exist_ok=True)
    poses,ground=odometry(clouds); path,smax=build_reference_path(poses,ground)
    with open(os.path.join(out,'ego_trajectory.json'),'w') as f:json.dump({'source':'KISS-ICP on LAS; assumes LAS is body coordinates','poses':[p.tolist() for p in poses],'training_eligible':False},f)
    mode=str(pp.get('marking_mode') or 'add');
    if mode not in ('add','replace'):raise ValueError('invalid marking mode')
    seg=Segmenter(); inpainter=(Cv2Inpainter() if pp.get('gnd_mode')=='legacy' else RoadInpainter()) if mode=='replace' else None; n=0; painted=0
    for i,(im,Twb) in enumerate(zip(imgs,poses)):
        t=time.monotonic();raw=cv2.imread(im)
        if raw is None:raise ValueError('image unreadable: '+im)
        und=cv2.undistort(raw,K,D) if len(D) else raw
        road,erase=seg.masks(und)
        clean=inpainter.apply(und,erase) if inpainter else und
        rgb,a=paint(clean,K,Tcb,Twb,tex,meta,pp,path,smax)
        sim,a=composite(clean,rgb,a,road,opacity=pp.get('opacity',.82),reflectivity=pp.get('reflectivity',1.95),optical_blur=pp.get('optical_blur',1.2),occlusion=bool(pp.get('occlusion',True)))
        stem=Path(im).stem; mask=(a>0.05).astype(np.uint8)*255
        if not cv2.imwrite(os.path.join(out,stem+'_gnd.jpg'),sim):raise RuntimeError('failed writing output')
        os.makedirs(os.path.join(out,'synthetic_masks'),exist_ok=True)
        cv2.imwrite(os.path.join(out,'synthetic_masks',stem+'.png'),mask)
        if inpainter:
            os.makedirs(os.path.join(out,'erase_masks'),exist_ok=True)
            cv2.imwrite(os.path.join(out,'erase_masks',stem+'.png'),erase)
            os.makedirs(os.path.join(out,'clean_frames'),exist_ok=True)
            cv2.imwrite(os.path.join(out,'clean_frames',stem+'.jpg'),clean)
        px=int(np.count_nonzero(mask));painted+=bool(px)
        with open(os.path.join(ad,stem+'.json'),'w') as f:json.dump({'frame':stem,'index':i,'rendered':True,'synthetic_pixels':px,'erased_pixels':int(np.count_nonzero(erase)) if inpainter else 0,'marking_mode':mode,'mask':'synthetic_masks/'+stem+'.png','params':pp,'labels_complete':False,'training_eligible':False,'image_space':'undistorted'},f)
        n+=1;print('RENDER %d/%d paint=%d %.2fs'%(n,len(imgs),px,time.monotonic()-t),flush=True)
    with open(os.path.join(out,'trajectory_manifest.json'),'w') as f:json.dump({'schema_version':1,'module':'gnd','frames_total':len(imgs),'frames_rendered':n,'frames_with_paint':painted,'camera_lidar_max_offset_ms':float(dt.max()*1000),'ground_z':ground,'marking_mode':mode,'params':pp,'training_eligible':False,'labels_complete':False,'note':'validation preview; road layout approximated from ego trajectory; original labels not merged'},f)
    return 0 if painted else 2
if __name__=='__main__':sys.exit(main())
