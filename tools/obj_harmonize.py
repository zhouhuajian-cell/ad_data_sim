"""Conservative foreground-only appearance matching for OBJ previews.

RGB is modified only on visible synthetic body pixels; alpha, occlusion and
annotations are never altered.  State is per clip, not shared between jobs.
"""
import cv2
import numpy as np

class ForegroundHarmonizer:
    def __init__(self, strength=0.12, saturation=0.96, temporal=0.15):
        self.strength = float(np.clip(strength, 0, .3))
        self.saturation = float(np.clip(saturation, .85, 1))
        self.temporal = float(np.clip(temporal, .01, 1))
        self.shift = None

    def apply(self, bg, fg, body, real_mask=None):
        h,w=bg.shape[:2]
        body=np.asarray(body,dtype=bool)
        if body.shape!=(h,w) or fg.shape!=(h,w,4): raise ValueError('尺寸不匹配')
        if not body.any(): return fg, {'applied':False,'reason':'no_visible_body'}
        ys,xs=np.where(body)
        pad=24
        x0,x1=max(0,int(xs.min())-pad),min(w,int(xs.max())+pad+1)
        y0,y1=max(0,int(ys.min())-pad),min(h,int(ys.max())+pad+1)
        b=bg[y0:y1,x0:x1]; f=fg[y0:y1,x0:x1]; m=body[y0:y1,x0:x1]
        # Exclude all synthetic alpha (including shadows) and real foreground.
        ring=cv2.dilate(m.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))>0
        ring &= f[:,:,3] < 4
        if real_mask is not None: ring &= ~np.asarray(real_mask,dtype=bool)[y0:y1,x0:x1]
        if ring.sum()<100 or m.sum()<64: return fg, {'applied':False,'reason':'insufficient_pixels'}
        bl=cv2.cvtColor(b,cv2.COLOR_BGR2LAB).astype(np.float32)
        fl=cv2.cvtColor(f[:,:,:3],cv2.COLOR_BGR2LAB).astype(np.float32)
        # Bounded shifts only: road color is not a valid target for car paint.
        target=np.median(bl[ring],axis=0)-np.median(fl[m],axis=0)
        target=np.clip(target,[-12,-5,-5],[12,5,5])*self.strength
        self.shift=target if self.shift is None else ((1-self.temporal)*self.shift+self.temporal*target)
        fl[m]=np.clip(fl[m]+self.shift,0,255)
        rgb=cv2.cvtColor(fl.astype(np.uint8),cv2.COLOR_LAB2BGR)
        if self.saturation<1:
            hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
            hsv[:,:,1]=(hsv[:,:,1].astype(np.float32)*self.saturation).astype(np.uint8)
            rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        result=fg.copy(); region=result[y0:y1,x0:x1];region[:,:,:3][m]=rgb[m]
        return result, {'applied':True,'pixels':int(m.sum()),'lab_shift':[round(float(v),3) for v in self.shift]}
