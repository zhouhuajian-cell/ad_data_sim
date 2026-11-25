#!/usr/bin/env python3
"""Conservative oriented 3D placement collision checks; no image-space rejection."""
import math
import numpy as np


def _box(cx, cy, cz, length, width, height, yaw):
    values = np.asarray([cx,cy,cz,length,width,height,yaw],dtype=float)
    if not np.isfinite(values).all() or np.any(values[3:6] <= 0):
        raise ValueError('invalid 3D box')
    c,s=math.cos(yaw),math.sin(yaw)
    return {'center':values[:3], 'half':values[3:6]*.5,
            'axes':np.array([[c,s],[-s,c]],dtype=float)}


def overlaps(a, b, margin=0.25):
    """3D upright OBB separating-axis test; touching expanded boxes counts as collision."""
    if not math.isfinite(margin) or margin < 0: raise ValueError('invalid margin')
    if abs(a['center'][2]-b['center'][2]) > a['half'][2]+b['half'][2]+margin:
        return False
    d=b['center'][:2]-a['center'][:2]
    for axis in np.vstack((a['axes'],b['axes'])):
        ra=sum(a['half'][i]*abs(np.dot(a['axes'][i],axis)) for i in range(2))
        rb=sum(b['half'][i]*abs(np.dot(b['axes'][i],axis)) for i in range(2))
        if abs(np.dot(d,axis)) > ra+rb+margin: return False
    return True


def candidate_box(x,y,ground_z,lwh,yaw_deg):
    length,width,height=map(float,lwh)
    return _box(x,y,ground_z+height/2,length,width,height,math.radians(yaw_deg))


def annotation_box(item):
    return _box(item['Wx'],item['Wy'],item['Wz'],item['L'],item['W'],item['H'],item['yaw'])


def point_occupancy(box, points, ground_z, ground_margin=.25):
    """Count points in candidate volume, ignoring a thin ground band.

    Input must be synchronized LiDAR XYZ in the same coordinate system.
    Sparse/no points is not proof that the space is free.
    """
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all(): raise ValueError('invalid points')
    d=p[:,:2]-box['center'][:2]; local=d@box['axes'].T
    keep=(np.abs(local[:,0])<=box['half'][0])&(np.abs(local[:,1])<=box['half'][1])
    keep &= (p[:,2]>ground_z+ground_margin)&(p[:,2]<box['center'][2]+box['half'][2])
    return int(keep.sum())


def check_collision(candidate, labels, lidar_xyz=None, margin=.25, point_limit=5, ground_z=None):
    if labels is None: return {'ok':False,'reason':'missing_labels'}
    for item in labels:
        if item.get('delete') or not item.get('kind'): continue
        try: other=annotation_box(item)
        except (ValueError,TypeError,KeyError): return {'ok':False,'reason':'invalid_real_box'}
        if overlaps(candidate,other,margin):
            return {'ok':False,'reason':'real_3d_overlap','real_id':item.get('tracking_id')}
    if lidar_xyz is None or ground_z is None:
        return {'ok':False,'reason':'missing_lidar_or_ground'}
    count=point_occupancy(candidate,lidar_xyz,ground_z)
    return {'ok':count<=point_limit,'reason':'ok' if count<=point_limit else 'lidar_occupied',
            'lidar_points':count,'note':'zero points does not establish visibility or free space'}
