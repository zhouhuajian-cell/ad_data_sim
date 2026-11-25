#!/usr/bin/env python3
"""Nearest-neighbor audit of ICP alignment on approximate roadside static points."""
import json,os,sys,struct
import numpy as np
from scipy.spatial import cKDTree

def cloud(path):
    with open(path,'rb') as f:
        n=struct.unpack('<I',f.read(4))[0]; b=f.read()
    if len(b)!=n*16:raise ValueError('bad cloud length')
    return np.frombuffer(b,dtype='<f4').reshape(n,4)[:,:3].astype('float64')
def sample(p):
    q=p[np.isfinite(p).all(1)&(p[:,0]>4)&(p[:,0]<65)&(abs(p[:,1])>5)&(p[:,2]>.2)&(p[:,2]<8)]
    return q[::4]
def main():
    clip,trajectory,output=sys.argv[1:]; data=json.load(open(trajectory)); rec=data['poses'];root=os.path.join(clip,'lidar_hs_pandar128_topmiddle')
    initial=sample(cloud(os.path.join(root,rec[0]['stem']+'.bin'))); tree=cKDTree(initial); results=[]
    for i in [1,10,20,30,40,len(rec)-1]:
        q=sample(cloud(os.path.join(root,rec[i]['stem']+'.bin'))); T=np.array(rec[i]['pose']); aligned=q@T[:3,:3].T+T[:3,3]
        dist,_=tree.query(aligned,k=1,workers=4); results.append({'frame':i,'points':len(q),'median_nn_m':float(np.median(dist)),
            'within_0_25m':float(np.mean(dist<=.25)),'within_0_5m':float(np.mean(dist<=.5))})
    report={'method':'first-scan nearest neighbor on heuristic roadside subset','limits':['not independent ground truth','dynamic objects and visibility changes can affect scores','first scan has finite coverage'],
            'training_eligible':False,'results':results}
    json.dump(report,open(output,'w'),indent=2);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
