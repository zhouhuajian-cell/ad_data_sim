"""Deterministic texture-space wear and conservative preview compositing."""
import hashlib
import cv2
import numpy as np


def prepare_texture(tex, wear=0., dirt=0., seed=''):
    if tex is None or tex.ndim != 3 or tex.shape[2] not in (3,4):
        raise ValueError('invalid marking texture')
    wear=float(wear);dirt=float(dirt)
    if not (0 <= wear <= 1 and 0 <= dirt <= 1):raise ValueError('wear/dirt out of range')
    t=tex.copy()
    if t.shape[2]==3:t=np.dstack((t,np.full(t.shape[:2],255,np.uint8)))
    if wear==0 and dirt==0:return t
    h,w=t.shape[:2];seed_int=int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed_int)
    noise=rng.random((max(4,h//16),max(4,w//16))).astype(np.float32)
    noise=cv2.resize(noise,(w,h),interpolation=cv2.INTER_LINEAR)
    noise=cv2.GaussianBlur(noise,(0,0),1.0)
    if wear:
        keep=np.clip((noise-wear*.52)/(.18+wear*.20),0,1)
        t[:,:,3]=np.round(t[:,:,3].astype(np.float32)*keep).astype(np.uint8)
    if dirt:
        factor=1.-dirt*np.clip((.58-noise)*1.8,0.,.8)
        t[:,:,:3]=np.clip(t[:,:,:3].astype(np.float32)*factor[:,:,None],0,255).astype(np.uint8)
    return t


def composite(bg,rgb,alpha,road,opacity=.82,reflectivity=1.95,optical_blur=1.2,occlusion=True):
    opacity=float(opacity);reflectivity=float(reflectivity);optical_blur=float(optical_blur)
    if not (0<=opacity<=1 and 0.1<=reflectivity<=4 and 0<=optical_blur<=3):raise ValueError('appearance parameter out of range')
    if bg.shape!=rgb.shape or alpha.shape!=bg.shape[:2] or road.shape!=alpha.shape:raise ValueError('image/mask dimensions mismatch')
    a=alpha.astype(np.float32)/255.
    light=np.clip(cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY).astype(np.float32)/128.,.55,1.1)
    fg=np.clip(rgb.astype(np.float32)*light[:,:,None]*(reflectivity/1.95),0,255)
    prem=fg*a[:,:,None]
    if optical_blur>0:
        a=cv2.GaussianBlur(a,(0,0),optical_blur)
        prem=cv2.GaussianBlur(prem,(0,0),optical_blur)
    gate=(road>0).astype(np.float32) if occlusion else np.ones_like(a)
    a=np.clip(a*gate*opacity,0,1);prem=prem*gate[:,:,None]*opacity
    out=np.clip(prem+bg.astype(np.float32)*(1-a[:,:,None]),0,255).astype(np.uint8)
    return out,a
