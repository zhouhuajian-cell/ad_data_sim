#!/usr/bin/env python3
"""Read-only single-frame comparison; 3D acceptance alone is not permission to composite."""
import json,math,os,random,sys,collections
import cv2,numpy as np
sys.path.insert(0,'/opt/ue5_synth/tools')
from obj_collision_3d_codex import candidate_box,check_collision
from batch_canonical_v2 import MaskChecker
from label_gate import load_frame_labels
from lidar_depth_core import load_points

clip=sys.argv[1];stem=sys.argv[2];output=sys.argv[3]
img=cv2.imread(os.path.join(clip,'cam_st_x2f_front',stem+'.jpg'));d=load_frame_labels(clip,xml_path=stem+'.xml');cam=d['camera']
K={'fx':cam['fx'],'fy':cam['fy'],'cx':cam['cx'],'cy':cam['cy']}
checker=MaskChecker(img,K,np.asarray(cam['R']),np.asarray(cam['t']))
xyz=load_points(os.path.join(clip,'lidar_hs_pandar128_topmiddle',stem+'.bin'))[:,:3]
relevant=xyz[(xyz[:,0]>4)&(xyz[:,0]<40)&(np.abs(xyz[:,1])<10)&(xyz[:,2]>.4)&(xyz[:,2]<4)]
rng=random.Random(20261123);old_ok=new_ok=both=0;reasons=collections.Counter();examples=[]
for _ in range(250):
    x=rng.uniform(10,28);y=rng.uniform(-4,4);yaw=180.;lwh=[4.6,2,1.5];z=.18
    old,foot,why=checker.footprint_ok(x,y,*lwh[:2],yaw)
    box=candidate_box(x,y,z,lwh,yaw);result=check_collision(box,d['objects'],relevant,ground_z=z)
    old_ok+=bool(old);new_ok+=bool(result['ok']);both+=bool(old and result['ok']);reasons[result['reason']]+=1
    if not old and result['ok'] and len(examples)<5:examples.append({'x':round(x,2),'y':round(y,2),'old_reason':why,'new':result})
report={'candidates':250,'old_2d_accepted':old_ok,'3d_collision_accepted':new_ok,'both':both,'3d_reasons':dict(reasons),
        '3d_only_examples':examples,'warning':'3D free is not verified drivable road, occlusion, trajectory or training eligibility',
        'lidar_note':'same-stem raw scan, ground height heuristic, labels not independently verified','training_eligible':False}
json.dump(report,open(output,'w'),indent=2,ensure_ascii=False);print(json.dumps(report,ensure_ascii=False),flush=True)
