#!/usr/bin/env python3
"""Single-frame Mask2Former instance + sparse lidar depth audit (no training export)."""
import argparse,collections,json,os,sys
import cv2,numpy as np,torch
from PIL import Image
from transformers import AutoImageProcessor,Mask2FormerForUniversalSegmentation
sys.path.insert(0,'/opt/ue5_synth/tools')
from done_data_labels import parse_xml
from audit_lidar_projection_codex import read_bin

def main():
    p=argparse.ArgumentParser();p.add_argument('--clip-dir',required=True);p.add_argument('--stem',required=True);p.add_argument('--out',required=True);a=p.parse_args();os.makedirs(a.out,exist_ok=True)
    img=cv2.imread(os.path.join(a.clip_dir,'cam_st_x2f_front',a.stem+'.jpg'));h,w=img.shape[:2]
    model='facebook/mask2former-swin-large-mapillary-vistas-semantic'
    proc=AutoImageProcessor.from_pretrained(model,local_files_only=True); net=Mask2FormerForUniversalSegmentation.from_pretrained(model,local_files_only=True).to('cuda').eval()
    with torch.inference_mode():
        inputs=proc(images=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),return_tensors='pt').to('cuda')
        raw=net(**inputs)
        pred=proc.post_process_instance_segmentation(raw,threshold=0.5,target_sizes=[(h,w)])[0]
    seg=pred['segmentation'].detach().cpu().numpy(); info=pred['segments_info']; labels=net.config.id2label
    cam=parse_xml(os.path.join(a.clip_dir,'label_lidar_hs_pandar128_topmiddle',a.stem+'.xml'))['cameras']['cam_st_x2f_front']
    points=read_bin(os.path.join(a.clip_dir,'lidar_hs_pandar128_topmiddle',a.stem+'.bin'))[:,:3].astype('float64')
    c=points@np.asarray(cam['R']).T+np.asarray(cam['t']); valid=(c[:,2]>.1)&(c[:,2]<150); xyz=points[valid]; z=c[valid,2]
    xy,_=cv2.projectPoints(xyz,np.asarray(cam['rvec']),np.asarray(cam['tvec']),np.asarray(cam['K']),np.asarray(cam['distortion']))
    uv=np.rint(xy.reshape(-1,2)).astype(int);valid=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);uv,z=uv[valid],z[valid]
    output=[];vis=img.copy();keywords=('car','truck','bus','vehicle','person','rider','motorcycle','bicycle','trailer')
    for it in info:
        label=str(labels.get(int(it['label_id']),'unknown')); ident=int(it['id'])
        if not any(k in label.lower() for k in keywords):continue
        mask=seg==ident; n=int(mask.sum()); vals=z[seg[uv[:,1],uv[:,0]]==ident]
        if n<50:continue
        record={'id':ident,'label':label,'score':float(it.get('score',0)),'mask_pixels':n,'lidar_points':len(vals),
                'depth_camera_z_median_m':float(np.median(vals)) if len(vals)>=10 else None,
                'depth_iqr_m':(np.percentile(vals,[25,75]).tolist() if len(vals)>=10 else None)};output.append(record)
        overlay=vis.copy();overlay[mask]=(.5*overlay[mask]+.5*np.array([0,180,255])).astype(np.uint8);vis=overlay
        ys,xs=np.nonzero(mask);text='%s n=%d z=%s'%(label,len(vals),round(float(np.median(vals)),1) if len(vals)>=10 else '?')
        cv2.putText(vis,text,(int(xs.min()),max(20,int(ys.min()))),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
    output_path=os.path.join(a.out,'instance_lidar_overlay.jpg');cv2.imwrite(output_path,vis)
    report={'instances':output,'model':model,'semantic_checkpoint_instance_output':'unverified',
            'camera_depth':'camera_optical_z','training_eligible':False,'limitations':['sparse points','occlusion boundary uncertain','timestamp timing not validated','not a dense real depth map']}
    with open(os.path.join(a.out,'instance_depth.json'),'w') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
