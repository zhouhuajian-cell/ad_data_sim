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


def validate_ground_alignment(previous_image, current_image, previous_road, H,
                              min_points=12, max_median_px=3.0, max_p90_px=8.0):
    """Fail closed when the projected road disagrees with tracked image features.

    This is a consistency check, not proof of an accurate world pose. Restrict
    features to the previous road mask and reject unreliable optical flow tracks.
    """
    result = {'valid': False, 'reason': 'missing_input', 'points': 0}
    if previous_image is None or current_image is None or previous_road is None or H is None:
        return result
    if previous_image.shape != current_image.shape or previous_road.shape != previous_image.shape[:2]:
        result['reason'] = 'shape_mismatch'
        return result
    old = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    road = cv2.erode((previous_road > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
    points = cv2.goodFeaturesToTrack(old, 400, 0.01, 8, mask=road)
    if points is None:
        result['reason'] = 'insufficient_features'
        return result
    projected = cv2.perspectiveTransform(points, np.asarray(H, dtype=np.float64))
    if not np.isfinite(projected).all():
        result['reason'] = 'invalid_projection'
        return result
    # Road motion can exceed LK's default search range; use the candidate
    # projection as an initial estimate, then independently check round trips.
    target, status, _ = cv2.calcOpticalFlowPyrLK(
        old, new, points, projected.astype(np.float32).copy(),
        winSize=(31, 31), maxLevel=4, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    if target is None or status is None:
        result['reason'] = 'flow_failed'
        return result
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(
        new, old, target, points.copy(), winSize=(31, 31), maxLevel=4,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    if back is None or back_status is None:
        result['reason'] = 'flow_failed'
        return result
    projected = projected[:, 0]
    forward = target[:, 0]
    ok = ((status[:, 0] > 0) & (back_status[:, 0] > 0)
          & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.0)
          & np.isfinite(projected).all(axis=1) & np.isfinite(forward).all(axis=1))
    errors = np.linalg.norm(projected[ok] - forward[ok], axis=1)
    result['points'] = int(len(errors))
    if len(errors) < min_points:
        result['reason'] = 'insufficient_features'
        return result
    result['median_px'] = float(np.median(errors))
    result['p90_px'] = float(np.percentile(errors, 90))
    result['within_3px'] = int(np.count_nonzero(errors <= 3.0))
    result['valid'] = result['median_px'] <= max_median_px and result['p90_px'] <= max_p90_px
    result['reason'] = 'ok' if result['valid'] else 'projection_mismatch'
    return result


def stabilize_repaired_road(current, current_mask, previous, previous_mask, H,
                            weight=.35, max_channel_difference=12):
    """Experiment: blend only mutually erased, photometrically similar pixels.

    The caller must check the projection and road/foreground masks first. This
    doesn't stabilize the erase mask itself or establish training-quality labels.
    """
    if previous is None or previous_mask is None or H is None:
        return current, 0
    if current.shape != previous.shape or current_mask.shape != current.shape[:2] or previous_mask.shape != current_mask.shape:
        raise ValueError('temporal repair shape mismatch')
    h, w = current_mask.shape
    warped = cv2.warpPerspective(previous, H, (w, h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(previous_mask, H, (w, h), flags=cv2.INTER_NEAREST) > 0
    similar = np.max(np.abs(warped.astype(np.int16) - current.astype(np.int16)), axis=2) < max_channel_difference
    eligible = warped_mask & (current_mask > 0) & similar
    eligible = cv2.erode(eligible.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = current.copy()
    wgt = float(np.clip(weight, 0, 1))
    out[eligible] = np.rint((1-wgt)*current[eligible] + wgt*warped[eligible]).astype(np.uint8)
    return out, int(np.count_nonzero(eligible))
