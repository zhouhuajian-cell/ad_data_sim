#!/usr/bin/env python3
"""Preview world-fixed water barriers against a calibrated LiDAR clip."""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, '/opt/ue5_synth/tools')
from done_data_labels import parse_xml
from lidar_depth_core import load_points, fit_ground_plane, ground_z
from obj_occlusion_3d import foreground_depth, radial_to_camera_z, composite
from gnd_mask_4a import MapillaryMarkingExtractor

BLENDER='/data/blender-4.2.23-linux-x64/blender'
RENDER='/opt/ue5_synth/tools/render_canonical_one.py'
GLB='/data/assets_global/OBJ_3D/_obs_dec/water_barrier/barrier_wall_plastic_orange_a03/mesh.glb'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clip',required=True)
    ap.add_argument('--poses',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--limit',type=int,default=50)
    args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)
    os.makedirs(os.path.join(args.out,'annotations'),exist_ok=True)
    records=json.load(open(args.poses,encoding='utf-8'))['poses'][:args.limit]
    first=records[0]['stem']
    cloud=load_points(os.path.join(args.clip,'lidar_hs_pandar128_topmiddle',first+'.bin'))[:,:3]
    plane=fit_ground_plane(cloud)
    if not plane or not plane.get('sane'):
        raise RuntimeError('first frame ground unavailable')
    # Same fixed first-LiDAR-world positions for all frames.
    gz=ground_z(plane,30.,-3.5)
    world=[np.array([28.+2.2*i,-6.0,gz,1.]) for i in range(4)]
    ext=MapillaryMarkingExtractor()
    summary=[]
    for index,rec in enumerate(records):
        stem=rec['stem'];T=np.asarray(rec['pose'],float);inv=np.linalg.inv(T)
        imgp=os.path.join(args.clip,'cam_st_x2f_front',stem+'.jpg')
        xmlp=os.path.join(args.clip,'label_lidar_hs_pandar128_topmiddle',stem+'.xml')
        cam=parse_xml(xmlp)['cameras']['cam_st_x2f_front'];K=np.asarray(cam['K'],float)
        R=np.asarray(cam['R'],float);t=np.asarray(cam['t'],float)
        raw=cv2.imread(imgp)
        if raw is None: raise RuntimeError('missing image '+imgp)
        h,w=raw.shape[:2];bg=cv2.undistort(raw,K,np.asarray(cam['distortion']),None,K)
        bgpath=os.path.join(args.out,'_bg_'+stem+'.png');cv2.imwrite(bgpath,bg)
        Tc=np.eye(4);Tc[:3,:3]=R;Tc[:3,3]=t
        vehicles=[];anns=[]
        for i,p in enumerate(world):
            local=inv@p
            if local[0]<2.: continue
            # A fixed world orientation, transformed into the current LiDAR frame.
            yaw=-np.degrees(np.arctan2(T[1,0],T[0,0]))
            x,y,z=map(float,local[:3]);L,W,H=2.0,.5,1.0
            vehicles.append({'uid':'barrier_%d'%i,'glb':GLB,'x':x,'y':y,'z':z,
                             'yaw':float(yaw),'size':[L,W,H],'class':'barrier'})
            yr=np.radians(yaw);c,s=np.cos(yr),np.sin(yr);corn=[]
            for dz in (0.,H):
                for dx,dy in ((-L/2,-W/2),(L/2,-W/2),(L/2,W/2),(-L/2,W/2)):
                    corn.append([x+dx*c-dy*s,y+dx*s+dy*c,z+dz])
            C=np.asarray(corn)@R.T+t;ok=C[:,2]>.05
            if not ok.all():continue
            uv=C[:,:2]/C[:,2:3];uv[:,0]=uv[:,0]*K[0,0]+K[0,2];uv[:,1]=uv[:,1]*K[1,1]+K[1,2]
            box=[float(max(0,uv[:,0].min())),float(max(0,uv[:,1].min())),
                 float(min(w-1,uv[:,0].max())),float(min(h-1,uv[:,1].max()))]
            if box[2]<=box[0] or box[3]<=box[1]:continue
            anns.append({'source':'synthetic','class':'water_barrier','track_id':'barrier_%d'%i,
                         'bbox2d':box,'corners':uv.tolist(),'world_position_m':p[:3].tolist(),
                         'local_position_m':[x,y,z],'geometry':'nominal_unverified'})
        outimg=os.path.join(args.out,stem+'_obj.jpg')
        if vehicles:
            fg=os.path.join(args.out,'_fg_'+stem+'.png');dep=os.path.join(args.out,'_depth_'+stem+'.npy')
            cfg={'bg':bgpath,'K':{'fx':K[0,0],'fy':K[1,1],'cx':K[0,2],'cy':K[1,2],'W':w,'H':h},
                 'T':Tc.tolist(),'vehicles':vehicles,'out':fg,'depth_npy':dep,'seed':42}
            cfgp=os.path.join(args.out,'_cfg_'+stem+'.json')
            with open(cfgp,'w') as f:json.dump(cfg,f)
            proc=subprocess.run([BLENDER,'--background','--python-exit-code','7','--python',RENDER,'--',cfgp],capture_output=True,timeout=900)
            if proc.returncode or not os.path.isfile(dep) or not os.path.isfile(fg):
                raise RuntimeError('render failed '+stem+' '+proc.stderr.decode('utf-8','replace')[-500:])
            rgba=cv2.imread(fg,cv2.IMREAD_UNCHANGED)
            xyz=load_points(os.path.join(args.clip,'lidar_hs_pandar128_topmiddle',stem+'.bin'))[:,:3]
            mask=ext.get_vehicle_mask(bg)>127
            rd,_=foreground_depth(xyz,K,R,t,mask,max_fill_px=6)
            sd=radial_to_camera_z(np.load(dep),K)
            out,stat=composite(bg,rgba,sd,rd,mask)
            cv2.imwrite(outimg,out,[cv2.IMWRITE_JPEG_QUALITY,94])
        else:
            cv2.imwrite(outimg,bg,[cv2.IMWRITE_JPEG_QUALITY,94]);stat={'visible_ratio':0}
        with open(os.path.join(args.out,'annotations',stem+'.json'),'w',encoding='utf-8') as f:
            json.dump({'frame':stem,'objects':anns,'training_eligible':False},f,ensure_ascii=False,allow_nan=False)
        summary.append({'stem':stem,'world_positions_m':[p[:3].tolist() for p in world],
                        'visible_bbox_count':len(anns),'composite_visible_ratio':stat['visible_ratio']})
        print('frame %d/%d %s objects=%d visible=%.3f'%(index+1,len(records),stem,len(anns),stat['visible_ratio']),flush=True)
    with open(os.path.join(args.out,'static_validation.json'),'w') as f:
        json.dump({'frames':summary,'world_anchored':True,'training_eligible':False,
                   'ground_z_extrapolated':True,'world_position_source':'first_lidar'},f,indent=2)

if __name__=='__main__':main()
