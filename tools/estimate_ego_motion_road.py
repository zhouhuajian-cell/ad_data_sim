#!/usr/bin/env python3
"""Road-motion audit for real OBJ clips; outputs estimates, never training labels."""
import argparse
import json
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, '/opt/ue5_synth/tools')
from label_gate import list_clip_frames, load_frame_labels
from obj_real3d_preview import _clip
from ground_masks import road_mask


def read_frame(path, xml, clip, width=960):
    img = cv2.imread(path)
    if img is None: raise ValueError('图像不可读: ' + path)
    h, w = img.shape[:2]
    d = load_frame_labels(clip, xml_path=xml) if xml else None
    if not d or not d.get('camera'): raise ValueError('缺少帧标定: ' + path)
    road = road_mask(img)
    if road is None or road.shape != (h, w): raise ValueError('缺少路面掩码')
    mask = (road > .5).astype(np.uint8)
    for obj in d.get('objects', []):
        b = obj.get('bbox2d')
        if b is None or len(b) != 4 or not np.isfinite(b).all(): continue
        x0, y0 = max(0, int(b[0])-8), max(0, int(b[1])-8)
        x1, y1 = min(w, int(np.ceil(b[2]))+8), min(h, int(np.ceil(b[3]))+8)
        if x1 > x0 and y1 > y0: mask[y0:y1, x0:x1] = 0
    mask[:int(h*.57)] = 0
    mask[int(h*.82):] = 0
    size = (width, round(h*width/w))
    gray = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) * 255
    cam = d['camera']; K = np.asarray(cam['K'], dtype=np.float64).reshape(3,3).copy()
    K[0, :] *= size[0] / w; K[1, :] *= size[1] / h
    R = np.asarray(cam['R'], dtype=np.float64).reshape(3,3)
    t = np.asarray(cam['t'], dtype=np.float64).reshape(3)
    G = K @ np.column_stack((R[:,0], R[:,1], t))
    if not np.isfinite(G).all() or abs(np.linalg.det(G)) < 1e-9: raise ValueError('地面投影退化')
    return gray, mask, G


def estimate(a, b):
    prev, mask0, G0 = a; curr, mask1, G1 = b
    points = cv2.goodFeaturesToTrack(prev, 1600, .005, 5, mask=mask0, blockSize=5)
    if points is None or len(points) < 50: return None, {'reason':'too_few_features'}
    dst, ok1, _ = cv2.calcOpticalFlowPyrLK(prev, curr, points, None, winSize=(21,21), maxLevel=3)
    if dst is None: return None, {'reason':'flow_failed'}
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, dst, None, winSize=(21,21), maxLevel=3)
    if back is None: return None, {'reason':'backward_flow_failed'}
    xy = np.rint(dst.reshape(-1,2)).astype(int); h,w=curr.shape
    inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    valid = (ok1[:,0]>0)&(ok2[:,0]>0)&inside
    valid &= np.linalg.norm((back-points).reshape(-1,2),axis=1)<1
    road=np.zeros(len(points),bool); road[inside]=mask1[xy[inside,1],xy[inside,0]]>0; valid &= road
    x,y=points.reshape(-1,2)[valid],dst.reshape(-1,2)[valid]
    if len(x)<50: return None, {'reason':'too_few_tracks','tracks':len(x)}
    xg=cv2.perspectiveTransform(x.reshape(-1,1,2),np.linalg.inv(G0)).reshape(-1,2)
    yg=cv2.perspectiveTransform(y.reshape(-1,1,2),np.linalg.inv(G1)).reshape(-1,2)
    valid=np.isfinite(xg).all(1)&np.isfinite(yg).all(1)
    for pts in (xg,yg): valid &= (pts[:,0]>3)&(pts[:,0]<60)&(np.abs(pts[:,1])<15)
    xg,yg=xg[valid],yg[valid]
    if len(xg)<50: return None, {'reason':'too_few_ground_tracks','tracks':len(xg)}
    fit,inliers=cv2.estimateAffinePartial2D(xg,yg,method=cv2.RANSAC,ransacReprojThreshold=.20,maxIters=3000,confidence=.995)
    if fit is None or inliers is None: return None, {'reason':'fit_failed'}
    keep=inliers[:,0]>0
    if keep.sum()<40 or keep.mean()<.35: return None, {'reason':'weak_ground_fit','tracks':len(xg),'inliers':int(keep.sum())}
    xg,yg=xg[keep],yg[keep]; xc,yc=xg.mean(0),yg.mean(0)
    u,_,vt=np.linalg.svd((xg-xc).T@(yg-yc)); rot=vt.T@u.T
    if np.linalg.det(rot)<0: vt[-1,:]*=-1; rot=vt.T@u.T
    trans=yc-rot@xc; residual=np.linalg.norm(xg@rot.T+trans-yg,axis=1)
    yaw=float(np.degrees(np.arctan2(rot[1,0],rot[0,0])))
    info={'reason':'candidate_only','tracks':len(valid),'inliers':int(keep.sum()),'inlier_ratio':float(keep.mean()),
          'median_residual_m':float(np.median(residual)), 'translation_m':trans.tolist(),'yaw_deg':yaw}
    if np.linalg.norm(trans)>5 or abs(yaw)>10 or info['median_residual_m']>.20:
        info['reason']='implausible_or_high_residual'; return None, info
    A=np.eye(3); A[:2,:2]=rot; A[:2,2]=trans
    return A,info


def anchor_score(first, current, cumulative):
    a,ma,G0=first; b,mb,G1=current
    orb=cv2.ORB_create(nfeatures=2500,fastThreshold=8)
    ka,da=orb.detectAndCompute(a,ma); kb,db=orb.detectAndCompute(b,mb)
    if da is None or db is None: return {'reason':'no_anchor_features'}
    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance < .70*n.distance]
    if len(good)<20: return {'reason':'too_few_anchor_matches','matches':len(good)}
    src=np.asarray([ka[m.queryIdx].pt for m in good],np.float32).reshape(-1,1,2)
    dst=np.asarray([kb[m.trainIdx].pt for m in good],np.float32)
    H=G1@cumulative@np.linalg.inv(G0)
    pred=cv2.perspectiveTransform(src,H).reshape(-1,2)
    e=np.linalg.norm(pred-dst,axis=1); finite=np.isfinite(e); e=e[finite]
    if len(e)<20: return {'reason':'invalid_anchor_projection','matches':len(e)}
    return {'reason':'diagnostic_only','matches':len(e),'within_4px':float(np.mean(e<=4)),
            'within_8px':float(np.mean(e<=8)),'median_px_at_960':float(np.median(e))}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--batch',required=True); p.add_argument('--clip',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); clip=_clip(args.batch,args.clip); frames=list_clip_frames(clip)
    if len(frames)<2: raise ValueError('至少需要两帧')
    previous=read_frame(*frames[0],clip); first=previous; cumulative=np.eye(3); chain=True; pairs=[]
    for index,(img,xml) in enumerate(frames[1:],1):
        current=read_frame(img,xml,clip)
        A,info=estimate(previous,current); previous=current
        if A is None: chain=False
        elif chain:
            cumulative=A@cumulative
            info['cumulative_ground_transform']=cumulative.tolist()
            if index%10==0 or index==len(frames)-1:
                info['anchor_audit']=anchor_score(first,current,cumulative)
        else: info['chain_status']='broken_no_silent_recovery'
        info['frame']=os.path.splitext(os.path.basename(img))[0]; pairs.append(info)
    result={'batch':args.batch,'clip':args.clip,'frames':len(frames),'mask_source':'heuristic_hsv_and_projected_real_boxes',
            'reference':'ground_plane_ego_frame','training_eligible':False,'independently_validated':False,
            'limitations':['no verified segmentation','no measured ego pose','planar ground approximation','lens distortion ignored',
                           'pairwise fit does not establish cumulative accuracy','not a real-scene depth map'], 'pairs':pairs}
    with open(args.output,'w',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'frames':len(frames),'accepted_pairs':sum(x['reason']=='candidate_only' for x in pairs),
                      'unbroken_prefix':sum('cumulative_ground_transform' in x for x in pairs),'output':args.output},ensure_ascii=False))

if __name__=='__main__': main()
