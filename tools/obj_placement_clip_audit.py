#!/usr/bin/env python3
"""Compare current 2D gate and 3D collision gate, per actual clip frame."""
import collections,json,os,random,sys
import cv2,numpy as np
sys.path.insert(0,'/opt/ue5_synth/tools')
from obj_collision_3d_codex import candidate_box,check_collision
from batch_canonical_v2 import MaskChecker
from label_gate import list_clip_frames,load_frame_labels
from lidar_depth_core import load_points

clip=sys.argv[1];output=sys.argv[2];frames=list_clip_frames(clip)
first=load_frame_labels(clip,xml_path=frames[0][1]);cam=first['camera'];img=cv2.imread(frames[0][0])
K={'fx':cam['fx'],'fy':cam['fy'],'cx':cam['cx'],'cy':cam['cy']}
checker=MaskChecker(img,K,np.asarray(cam['R']),np.asarray(cam['t']))
results=[]
for index,(imgpath,xml) in enumerate(frames):
    stem=os.path.splitext(os.path.basename(imgpath))[0];d=load_frame_labels(clip,xml_path=xml)
    if not d:
        results.append({'stem':stem,'error':'missing_labels'});continue
    pcpath=os.path.join(clip,'lidar_hs_pandar128_topmiddle',stem+'.bin')
    if not os.path.isfile(pcpath):results.append({'stem':stem,'error':'missing_lidar'});continue
    xyz=load_points(pcpath)[:,:3];xyz=xyz[np.isfinite(xyz).all(1)&(xyz[:,0]>4)&(xyz[:,0]<40)&(np.abs(xyz[:,1])<10)&(xyz[:,2]>.4)&(xyz[:,2]<4)]
    rng=random.Random(20261123+index); old=phys=shared=0;why=collections.Counter()
    for _ in range(60):
        x=rng.uniform(10,28);y=rng.uniform(-4,4);yaw=180.;lwh=[4.6,2.,1.5];z=.18
        ok,_,_=checker.footprint_ok(x,y,lwh[0],lwh[1],yaw)
        result=check_collision(candidate_box(x,y,z,lwh,yaw),d['objects'],xyz,ground_z=z)
        old+=int(ok);phys+=int(result['ok']);shared+=int(ok and result['ok']);why[result['reason']]+=1
    results.append({'stem':stem,'old_2d_first_mask_accepted':old,'collision_only_accepted':phys,'both':shared,'reasons':dict(why)})
    print('%d/%d old=%d 3d=%d'%(index+1,len(frames),old,phys),flush=True)
report={'frames':len(frames),'candidates_per_frame':60,'old_gate':'first_frame_mask_reused_as_in_current_pipeline',
        'frames_with_old_candidates':sum(r.get('old_2d_first_mask_accepted',0)>0 for r in results),
        'frames_with_3d_candidates':sum(r.get('collision_only_accepted',0)>0 for r in results),
        'limitations':['3D non-collision does not prove drivable road','no occlusion compositing checked',
                       'ground z approximate','real labels and lidar coverage not guaranteed complete',
                       'candidates sampled independently, not a trajectory'],
        'training_eligible':False,'frames_data':results}
json.dump(report,open(output,'w'),ensure_ascii=False,indent=2);print(json.dumps({k:v for k,v in report.items() if k!='frames_data'},ensure_ascii=False),flush=True)
