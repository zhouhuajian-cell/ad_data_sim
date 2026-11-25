#!/usr/bin/env python3
"""Audit exact frame pairing, XML calibration and candidate point-cloud projection."""
import argparse, glob, json, os, struct, sys, xml.etree.ElementTree as ET
import cv2
import numpy as np
sys.path.insert(0, '/opt/ue5_synth/tools')
from done_data_labels import parse_xml

def read_bin(path):
    with open(path,'rb') as f:
        header=f.read(4); n=struct.unpack('<I',header)[0]; data=f.read()
    if len(data)!=n*16: raise ValueError('点云头与文件长度不符')
    pts=np.frombuffer(data,dtype='<f4').reshape(n,4)
    if not np.isfinite(pts).all(): raise ValueError('点云包含非有限数')
    return pts

def main():
    p=argparse.ArgumentParser();p.add_argument('--clip-dir',required=True);p.add_argument('--stem',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    os.makedirs(a.out,exist_ok=True); dirs={k:os.path.join(a.clip_dir,v) for k,v in [('img','cam_st_x2f_front'),('pc','lidar_hs_pandar128_topmiddle'),('xml','label_lidar_hs_pandar128_topmiddle')]}
    names={k:{os.path.basename(x).split('.')[0]+'.'+os.path.basename(x).split('.')[1] for x in glob.glob(os.path.join(d,'*'+ext))} for (k,d),ext in zip(dirs.items(),['.jpg','.bin','.xml'])}
    imgfile=os.path.join(dirs['img'],a.stem+'.jpg');pcfile=os.path.join(dirs['pc'],a.stem+'.bin');xmlfile=os.path.join(dirs['xml'],a.stem+'.xml')
    points=read_bin(pcfile); img=cv2.imread(imgfile); assert img is not None
    d=parse_xml(xmlfile);cam=d['cameras']['cam_st_x2f_front'];R=np.array(cam['R']); t=np.array(cam['t']);K=np.array(cam['K']);h,w=img.shape[:2]
    xyz=points[:,:3].astype('float64'); campts=xyz@R.T+t
    forward=(campts[:,2]>.1)&(campts[:,2]<150); xyz=xyz[forward]; depth=campts[forward,2]
    for distort in (False,True):
        pxy,_=cv2.projectPoints(xyz,np.array(cam['rvec']),t,K,np.array(cam['distortion']) if distort else None)
        xy=pxy.reshape(-1,2); v=np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        xy=np.rint(xy[v]).astype(int); z=depth[v]; colors=cv2.applyColorMap(np.clip(z/80*255,0,255).astype('uint8').reshape(-1,1),cv2.COLORMAP_TURBO).reshape(-1,3)
        vis=img.copy()
        for (u,vv),c in zip(xy,colors): cv2.circle(vis,(int(u),int(vv)),1,tuple(int(i) for i in c),-1,cv2.LINE_AA)
        dest=os.path.join(a.out,'projection_distorted.jpg' if distort else 'projection_pinhole.jpg');cv2.imwrite(dest,vis,[cv2.IMWRITE_JPEG_QUALITY,93])
        print('distortion',distort,'points_in_image',len(xy),'output',dest,flush=True)
    allxml=sorted(glob.glob(os.path.join(dirs['xml'],'*.xml')));egos=[]
    for f in allxml:
        root=ET.parse(f).getroot(); node=root.find('.//EgoPoseParam')
        if node is None or node.get('validFlag')!='1': continue
        egos.append({'stem':os.path.basename(f)[:-4],'values':{k:float(node.findtext(k,'nan')) for k in ('wx','wy','wz','roll','pitch','yaw','qx','qy','qz','qw')}})
    out={'frame':a.stem,'file_format':'uint32 count then count * 4 float32','candidate_layout':'x,y,z,intensity','points':len(points),
         'percentiles_1_50_99':np.percentile(points,[1,50,99],axis=0).tolist(),'image_frames':len(names['img']),'lidar_frames':len(names['pc']),
         'xml_frames':len(names['xml']),'exact_filename_matches':len(names['img'] & names['pc'] & names['xml']),
         'timestamp_note':'filename equality only; exposure timing and lidar scan timing not verified',
         'extrinsic_source':'XML Lidar2CamParam', 'distortion_note':'image distortion state unverified',
         'ego_validflag_count':len(egos),'ego_first':egos[0] if egos else None,'ego_last':egos[-1] if egos else None,
         'ego_coordinate_convention':'unverified','training_eligible':False}
    with open(os.path.join(a.out,'audit.json'),'w',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
