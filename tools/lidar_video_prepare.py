#!/usr/bin/env python3
"""Prepare undistorted frames, bounded sparse depth and KISS transforms for a preview."""
import argparse,glob,json,os,sys,time
import cv2,numpy as np,torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from transformers import AutoImageProcessor,Mask2FormerForUniversalSegmentation
sys.path.insert(0,'/opt/ue5_synth/tools')
from audit_lidar_projection_codex import read_bin
from done_data_labels import parse_xml

def main():
    p=argparse.ArgumentParser();p.add_argument('--clip-dir',required=True);p.add_argument('--trajectory',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    os.makedirs(a.out,exist_ok=True); os.makedirs(os.path.join(a.out,'bg'),exist_ok=True);os.makedirs(os.path.join(a.out,'depth'),exist_ok=True)
    poses=json.load(open(a.trajectory))['poses']; model='facebook/mask2former-swin-large-mapillary-vistas-semantic'
    proc=AutoImageProcessor.from_pretrained(model,local_files_only=True); net=Mask2FormerForUniversalSegmentation.from_pretrained(model,local_files_only=True).to('cuda').eval()
    labels=net.config.id2label; items=[];sample_stem=poses[0]['stem']
    pc0=read_bin(os.path.join(a.clip_dir,'lidar_hs_pandar128_topmiddle',sample_stem+'.bin'))
    ground=pc0[(pc0[:,0]>15)&(pc0[:,0]<35)&(np.abs(pc0[:,1])<6)&(pc0[:,2]<.5)]
    if len(ground)<100:raise ValueError('insufficient ground points')
    ground_z=float(np.percentile(ground[:,2],20));print('ground_z',ground_z,flush=True)
    for index,rec in enumerate(poses):
        stem=rec['stem'];img=cv2.imread(os.path.join(a.clip_dir,'cam_st_x2f_front',stem+'.jpg'))
        if img is None:raise ValueError('missing image '+stem)
        h,w=img.shape[:2]; cam=parse_xml(os.path.join(a.clip_dir,'label_lidar_hs_pandar128_topmiddle',stem+'.xml'))['cameras']['cam_st_x2f_front']
        K=np.asarray(cam['K'],dtype=np.float64); K[0,:]*=960/w;K[1,:]*=540/h; dist=np.asarray(cam['distortion'])
        # Keep the original scaled intrinsics; undistortion changes the pixels, not K.
        bg=cv2.undistort(cv2.resize(img,(960,540)),K,dist,None,K)
        cv2.imwrite(os.path.join(a.out,'bg',stem+'.jpg'),bg,[cv2.IMWRITE_JPEG_QUALITY,94])
        with torch.inference_mode():
            inputs=proc(images=Image.fromarray(cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)),return_tensors='pt').to('cuda')
            raw=net(**inputs); pred=proc.post_process_instance_segmentation(raw,threshold=.5,target_sizes=[(540,960)])[0]
        seg=pred['segmentation'].cpu().numpy();info=pred['segments_info']; del inputs,raw,pred
        pc=read_bin(os.path.join(a.clip_dir,'lidar_hs_pandar128_topmiddle',stem+'.bin'))[:,:3].astype(np.float64)
        C=pc@np.asarray(cam['R']).T+np.asarray(cam['t']);valid=(C[:,2]>.1)&(C[:,2]<150);C=C[valid]
        uv=C[:,:2]/C[:,2:3];uv[:,0]=uv[:,0]*K[0,0]+K[0,2];uv[:,1]=uv[:,1]*K[1,1]+K[1,2]
        uv=np.rint(uv).astype(int);valid=(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<540);uv=uv[valid];z=C[valid,2]
        rd=np.full((540,960),np.nan,np.float32); foreground=np.zeros((540,960),np.uint8); stats=[]
        for it in info:
            label=str(labels.get(int(it['label_id']),'unknown')); low=label.lower()
            if low=='ego vehicle' or not any(k in low for k in ('car','truck','bus','vehicle','person','rider','motorcycle','bicycle','trailer')):continue
            ident=int(it['id']);mask=(seg==ident); foreground[mask]=1; pm=seg[uv[:,1],uv[:,0]]==ident;xy=uv[pm];dz=z[pm]
            if len(dz)<10 or mask.sum()<50:continue
            sparse=np.full((540,960),np.inf,np.float32); np.minimum.at(sparse,(xy[:,1],xy[:,0]),dz.astype('float32'))
            filled=np.isfinite(sparse); distances,inds=distance_transform_edt(~filled,return_indices=True)
            supported=mask&(distances<=6.0);values=sparse[inds[0],inds[1]];rd[supported]=values[supported]
            stats.append({'id':ident,'label':label,'points':len(dz),'mask_pixels':int(mask.sum()),'supported_pixels':int(supported.sum())})
        np.savez_compressed(os.path.join(a.out,'depth',stem+'.npz'),real_z=rd,foreground=foreground)
        R=np.asarray(cam['R']);t=np.asarray(cam['t']);T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
        items.append({'stem':stem,'K':K.tolist(),'T_lidar2cam':T.tolist(),'T_lidar2first':rec['pose'],'instances':stats})
        print('prepared %d/%d %s instances=%d'%(index+1,len(poses),stem,len(stats)),flush=True)
    with open(os.path.join(a.out,'scene.json'),'w') as f:json.dump({'frames':items,'ground_z':ground_z,'width':960,'height':540,'training_eligible':False,
       'depth_support_max_px':6,'undistorted_intrinsics':'scaled_original_K'},f,indent=2)
if __name__=='__main__':main()
