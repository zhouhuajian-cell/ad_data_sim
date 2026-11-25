#!/usr/bin/env python3
"""Compose preview with bounded LiDAR-instance depth; uncertainty is reported."""
import json,os,sys
import cv2,numpy as np

def main():
    out=sys.argv[1];cfg=json.load(open(os.path.join(out,'scene.json')));folder=os.path.join(out,'frames');os.makedirs(folder,exist_ok=True);stats=[]
    for i,fr in enumerate(cfg['frames']):
        stem=fr['stem'];bg=cv2.imread(os.path.join(out,'bg',stem+'.jpg'));fg=cv2.imread(os.path.join(out,'rgba',stem+'.png'),cv2.IMREAD_UNCHANGED)
        if bg is None or fg is None or fg.shape!=(540,960,4):raise ValueError('missing RGB '+stem)
        synth=np.load(os.path.join(out,'z',stem,'depth.npy')); pack=np.load(os.path.join(out,'depth',stem+'.npz')); real=pack['real_z']; foreground=pack['foreground']>0;K=np.asarray(fr['K'])
        # Cycles Z pass is ray distance; compare it in camera optical-axis Z.
        yy,xx=np.indices(synth.shape);ray=np.sqrt(1+((xx-K[0,2])/K[0,0])**2+((yy-K[1,2])/K[1,1])**2)
        z=synth/ray; alpha=fg[:,:,3].astype(np.float32)/255.0;mask=alpha>0.03
        valid=mask&np.isfinite(real)&np.isfinite(z)&(z>0)&(z<150)
        occluded=valid&(real+0.3<z); unknown_foreground=mask&foreground&~valid;alpha[occluded|unknown_foreground]=0
        a=alpha[:,:,None];result=np.clip(fg[:,:,:3].astype(np.float32)*a+bg.astype(np.float32)*(1-a),0,255).astype('uint8')
        cv2.imwrite(os.path.join(folder,'%04d.jpg'%i),result,[cv2.IMWRITE_JPEG_QUALITY,94])
        stats.append({'stem':stem,'synthetic_pixels':int(mask.sum()),'depth_supported_pixels':int(valid.sum()),
                      'occluded_pixels':int(occluded.sum()),'unknown_foreground_suppressed':int(unknown_foreground.sum()),'depth_unknown_pixels':int((mask&~valid).sum())})
    with open(os.path.join(out,'composite_stats.json'),'w') as f:json.dump({'frames':stats,'training_eligible':False,
       'notes':['preview only','unknown foreground suppressed conservatively; unknown background left visible','instance-interior nearest interpolation <=6px',
                'cycle ray depth converted to camera z; requires independent validation','fixed 0.1 s timestep for dynamic car']},f,indent=2)
    print(json.dumps({'frames':len(stats),'frames_with_occlusion':sum(s['occluded_pixels']>0 for s in stats),
       'total_occluded_pixels':sum(s['occluded_pixels'] for s in stats),'total_unknown_pixels':sum(s['depth_unknown_pixels'] for s in stats)}))
if __name__=='__main__':main()
