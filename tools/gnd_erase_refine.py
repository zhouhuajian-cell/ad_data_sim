"""Conservative GND erase-mask refinement for preview, not training truth."""
import cv2
import numpy as np


def refine_local(image, road, mask):
    """Expand strong white/yellow evidence only near model-detected markings."""
    road=(road>0).astype(np.uint8); orig=((mask>0)&(road>0)).astype(np.uint8)
    if not orig.any():return orig*255, {'added_pixels':0}
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    bg=cv2.GaussianBlur(gray,(15,15),3)
    diff=gray.astype(np.int16)-bg.astype(np.int16)
    white=(gray>=120)&(diff>=8)&(hsv[:,:,1]<100)
    yellow=(hsv[:,:,0]>=14)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>=65)&(hsv[:,:,2]>=100)&(diff>=4)
    near=cv2.dilate(orig,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0
    supplement=(white|yellow)&near&(road>0)
    result=(orig>0)|supplement
    return result.astype(np.uint8)*255, {'added_pixels':int(np.count_nonzero(result)-np.count_nonzero(orig))}


def ground_homography(K,Tcb,T_prev,T_cur,ground_z):
    """Project the same local planar patch into two camera frames."""
    K=np.asarray(K,dtype=np.float64);Tcb=np.asarray(Tcb,dtype=np.float64)
    prev=np.asarray(T_prev,dtype=np.float64);cur=np.asarray(T_cur,dtype=np.float64)
    xy=np.array([[8,-5],[30,-5],[30,5],[8,5]],dtype=np.float64)
    body=np.c_[xy,np.full(4,float(ground_z)),np.ones(4)]
    world=(cur@body.T).T
    def project(pose):
        camera=(np.linalg.inv(pose@Tcb)@world.T).T[:,:3]
        if np.any(camera[:,2]<=1):return None
        uv=(K@camera.T).T;uv=uv[:,:2]/uv[:,2,None]
        return uv if np.isfinite(uv).all() and np.max(np.abs(uv))<100000 else None
    src,dst=project(prev),project(cur)
    if src is None or dst is None:return None
    if abs(cv2.contourArea(src.astype(np.float32)))<10 or abs(cv2.contourArea(dst.astype(np.float32)))<10:return None
    H=cv2.getPerspectiveTransform(src.astype(np.float32),dst.astype(np.float32))
    return H if np.isfinite(H).all() and abs(np.linalg.det(H))>1e-12 else None


def refine_temporal(current,previous,H,road):
    """Borrow aligned past pixels only near current detection; never OR raw frames."""
    if H is None or previous is None:return current,0
    h,w=current.shape
    warped=cv2.warpPerspective(previous,H,(w,h),flags=cv2.INTER_NEAREST,borderValue=0)>0
    near=cv2.dilate((current>0).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
    added=warped&near&(road>0)&(current==0)
    if np.count_nonzero(added) > max(200, int(np.count_nonzero(current)*.10)):
        return current,0
    out=current.copy();out[added]=255
    return out,int(np.count_nonzero(added))
