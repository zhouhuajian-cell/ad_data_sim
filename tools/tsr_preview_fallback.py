"""Conservative single-frame sign preview; not a training renderer."""
import cv2
import numpy as np

def render(bg, bbox, asset, params):
    if asset.ndim!=3 or asset.shape[2]!=4: raise ValueError('标牌素材需要透明通道')
    h,w=bg.shape[:2]
    x0,y0,x1,y1=map(float,bbox)
    if x1<=x0 or y1<=y0: raise ValueError('标牌框无效')
    factor=float(np.clip(float(params.get('overcover',1.05)),.9,1.15))
    cx,cy=(x0+x1)/2,(y0+y1)/2
    bw,bh=(x1-x0)*factor,(y1-y0)*factor
    left,top=int(round(cx-bw/2)),int(round(cy-bh/2))
    right,bottom=int(round(cx+bw/2)),int(round(cy+bh/2))
    if left<0 or top<0 or right>w or bottom>h or right-left<3 or bottom-top<3:
        raise ValueError('标牌框超出画面或尺寸过小')
    lo=float(params.get('black_p5',28));hi=float(params.get('white_p95',218))
    if not 0<=lo<hi<=255: raise ValueError('黑电平必须小于白电平')
    sigma=float(np.clip(float(params.get('blur_sigma',.8)),0,1.6))
    blend=float(np.clip(float(params.get('color_blend',.2)),0,.45))
    # Area filtering avoids jagged edges when rendering small signs.
    patch=cv2.resize(asset,(right-left,bottom-top),interpolation=cv2.INTER_AREA)
    alpha=patch[:,:,3].astype(np.float32)/255.
    if not np.any(alpha>0):raise ValueError('素材完全透明')
    color=lo+patch[:,:,:3].astype(np.float32)*((hi-lo)/255.)
    roi=bg[top:bottom,left:right]
    scene=np.median(roi.reshape(-1,3),axis=0)
    cast=np.clip(scene-scene.mean(),-10,10)*blend
    color=np.clip(color+cast,0,255).astype(np.float32)
    if sigma>0:
        # Premultiplied blur prevents transparent pixels from making dark halos.
        weight=cv2.GaussianBlur(alpha,(0,0),sigma)
        pre=cv2.GaussianBlur(color*alpha[:,:,None],(0,0),sigma)
        color=pre/np.maximum(weight[:,:,None],1e-4)
    # Preserve original coverage: no bleeding into surrounding real pixels.
    out=bg.copy(); a=alpha[:,:,None]
    out[top:bottom,left:right]=np.clip(color*a+roi.astype(np.float32)*(1-a),0,255).astype(np.uint8)
    yy,xx=np.where(alpha>0.03)
    return out,[left+int(xx.min()),top+int(yy.min()),left+int(xx.max())+1,top+int(yy.max())+1]
