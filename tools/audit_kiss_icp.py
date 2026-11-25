#!/usr/bin/env python3
"""Evaluate KISS-ICP against XML positions after rigid alignment; not ground truth."""
import argparse,glob,json,os,struct,time,xml.etree.ElementTree as ET
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

def cloud(path):
    with open(path,'rb') as f:
        n=struct.unpack('<I',f.read(4))[0]; b=f.read()
    if len(b)!=n*16: raise ValueError('unknown binary layout')
    p=np.frombuffer(b,dtype='<f4').reshape(n,4)[:,:3].astype(np.float64)
    return p[np.isfinite(p).all(axis=1)]

def main():
    a=argparse.ArgumentParser(); a.add_argument('--clip-dir',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    files=sorted(glob.glob(os.path.join(args.clip_dir,'lidar_hs_pandar128_topmiddle','*.bin')))
    if len(files)<2:raise ValueError('not enough scans')
    cfg=KISSConfig();cfg.data.deskew=False;cfg.mapping.voxel_size=1.0;cfg.registration.max_num_threads=4
    icp=KissICP(cfg); out=[]; xmlposes=[]; start=time.monotonic()
    for i,f in enumerate(files):
        stem=os.path.basename(f)[:-4]; path=os.path.join(args.clip_dir,'label_lidar_hs_pandar128_topmiddle',stem+'.xml')
        root=ET.parse(path).getroot(); ego=root.find('.//EgoPoseParam')
        if ego is None or ego.get('validFlag')!='1':raise ValueError('missing valid ego '+stem)
        pos=np.array([float(ego.findtext(k)) for k in ('wx','wy','wz')]);xmlposes.append(pos)
        pts=cloud(f);icp.register_frame(pts,np.empty(0,dtype=np.float64))
        T=np.array(icp.last_pose);out.append({'stem':stem,'pose':T.tolist(),'points':len(pts)})
        print('scan %d/%d' % (i+1,len(files)),flush=True)
    xyz=np.array([x['pose'][:3] for x in []]) if False else np.array([np.array(x['pose'])[:3,3] for x in out])
    ref=np.array(xmlposes);X=xyz[:,:2];Y=ref[:,:2];xc=X.mean(0);yc=Y.mean(0)
    u,_,vt=np.linalg.svd((X-xc).T@(Y-yc));rot=vt.T@u.T
    if np.linalg.det(rot)<0:vt[-1]*=-1;rot=vt.T@u.T
    aligned=(X-xc)@rot.T+yc;err=np.linalg.norm(aligned-Y,axis=1)
    report={'frames':len(files),'elapsed_s':time.monotonic()-start,'deskew':False,'deskew_reason':'no validated per-point times',
            'raw_coordinate_layout':'x,y,z,intensity (candidate)','xml_position_is_ground_truth':False,
            'alignment':'2D rigid SE2 fitted to all positions; no scale fitted',
            'trajectory_rmse_m':float(np.sqrt(np.mean(err**2))),'trajectory_max_error_m':float(err.max()),
            'icp_path_length_m':float(np.linalg.norm(np.diff(X,axis=0),axis=1).sum()),
            'xml_path_length_m':float(np.linalg.norm(np.diff(Y,axis=0),axis=1).sum()),
            'first_last_icp_m':float(np.linalg.norm(X[-1]-X[0])),'first_last_xml_m':float(np.linalg.norm(Y[-1]-Y[0])),
            'training_eligible':False,'poses':out}
    with open(args.output,'w') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='poses'}),flush=True)
if __name__=='__main__': main()
