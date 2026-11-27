"""Preflight image-space overlaps for persistent OBJ trajectories."""
import math
import numpy as np
from real_calib import body_to_pixel_real


def projected_bbox(local, dimensions, camera, image_size, ground_z):
    if local is None or camera is None:
        return None
    x,y=float(local['x']),float(local['y'])
    length,width,height=map(float,dimensions)
    yaw=math.radians((math.degrees(float(local['yaw_rad']))+180.0)%360.0)
    c,s=math.cos(yaw),math.sin(yaw)
    points=[]
    for dz in (0.0,height):
        for dx,dy in ((-length/2,-width/2),(length/2,-width/2),
                      (length/2,width/2),(-length/2,width/2)):
            points.append([x+dx*c-dy*s,y+dx*s+dy*c,float(ground_z)+dz])
    uv,_,ok=body_to_pixel_real(camera,points)
    valid=ok & np.isfinite(uv).all(axis=1)
    if not valid.any():return None
    h,w=image_size;uv=uv[valid]
    box=[max(0.0,float(uv[:,0].min())),max(0.0,float(uv[:,1].min())),
         min(float(w-1),float(uv[:,0].max())),min(float(h-1),float(uv[:,1].max()))]
    return box if box[2]>box[0] and box[3]>box[1] else None


def too_much_overlap(a,b,threshold=.15):
    if a is None or b is None:return False
    overlap=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    smaller=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
    return bool(smaller>0 and overlap/smaller>threshold)
