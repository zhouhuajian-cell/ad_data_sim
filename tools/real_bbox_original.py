"""Read-only projection of real 3D labels onto the original distorted camera image."""
import os,sys,copy,functools
import numpy as np
import cv2
sys.path.insert(0,"/opt/ue5_synth/tools")
from label_gate import load_frame_labels
from done_data_labels import object_corners

@functools.lru_cache(maxsize=256)
def _projection(batch,clip,stem):
    for v in (batch,clip,stem):
        if not v or os.path.basename(v)!=v or v in ('.','..'): raise ValueError('invalid frame')
    cp=next((os.path.join(root,batch,clip) for root in ('/data/test_data/real_clips','/data/test_data/real_clips_extra') if os.path.isdir(os.path.join(root,batch,clip))),None)
    if not cp: raise ValueError('clip not found')
    lab=load_frame_labels(cp,xml_path=stem+'.xml')
    if not lab or not lab.get('camera'): raise ValueError('missing camera calibration')
    cam=lab['camera'];K=np.asarray(cam['K'],dtype=np.float64);R=np.asarray(cam['R'],dtype=np.float64);t=np.asarray(cam['t'],dtype=np.float64).reshape(3,1)
    dist=np.asarray(cam.get('distortion') or [],dtype=np.float64)
    if dist.size not in (0,4,5,8,12,14): raise ValueError('unsupported distortion model')
    if dist.size and not np.isfinite(dist).all():raise ValueError('invalid distortion')
    rv,_=cv2.Rodrigues(R);out=[]
    for ob in lab.get('objects') or []:
        if ob.get('delete') or not ob.get('in_front'):continue
        xyz=object_corners(ob)
        if not np.all((xyz@R.T+t.reshape(3))[:,2]>1e-5):continue
        uv,_=cv2.projectPoints(xyz,rv,t,K,dist if dist.size else None);uv=uv.reshape(-1,2)
        if not np.isfinite(uv).all():continue
        lo=uv.min(axis=0);hi=uv.max(axis=0)
        out.append({'x':float(ob['Wx']),'y':float(ob['Wy']),'z':float(ob['Wz']), 'kind':str(ob.get('kind') or ''),
          'bbox2d_orig':[round(float(lo[0]),2),round(float(lo[1]),2),round(float(hi[0]),2),round(float(hi[1]),2)],
          'corners_orig':[[round(float(x),2),round(float(y),2)] for x,y in uv]})
    return out

def enrich(ann,batch,clip,stem):
    result=copy.deepcopy(ann);projected=_projection(batch,clip,stem);used=set()
    for ob in result.get('objects') or []:
        if ob.get('source')!='real':continue
        box=ob.get('box3d') or {}
        if not all(k in box for k in ('x','y','z')):continue
        candidates=[(sum((float(box[k])-v[k])**2 for k in ('x','y','z')),i,v) for i,v in enumerate(projected) if i not in used]
        if not candidates:continue
        error,i,src=min(candidates,key=lambda item:item[0])
        if error>0.01:continue
        used.add(i);ob['bbox2d_orig']=src['bbox2d_orig'];ob['corners_orig']=src['corners_orig']
    result['original_geometry']='distorted_camera_projection'
    return result
