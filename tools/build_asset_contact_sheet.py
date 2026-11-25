"""Build review contact sheets from existing per-model reference images."""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('output')
    a=p.parse_args()
    dirs=sorted(d for d in glob.glob(os.path.join(a.root,'*')) if os.path.isdir(d))
    tile_w,tile_h=384,260
    sheet=np.full((int(np.ceil(len(dirs)/5))*tile_h,tile_w*5,3),240,np.uint8)
    for i,d in enumerate(dirs):
        x,y=(i%5)*tile_w,(i//5)*tile_h
        fp=os.path.join(d,'ref_view.png')
        im=cv2.imread(fp,cv2.IMREAD_UNCHANGED)
        if im is not None:
            if im.ndim==3 and im.shape[2]==4:
                alpha=im[:,:,3:4].astype(np.float32)/255
                im=(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
            h,w=im.shape[:2];scale=min((tile_w-12)/w,(tile_h-50)/h)
            im=cv2.resize(im,(max(1,int(w*scale)),max(1,int(h*scale))))
            h,w=im.shape[:2];sheet[y+30:y+30+h,x+(tile_w-w)//2:x+(tile_w-w)//2+w]=im[:,:,:3]
        cv2.putText(sheet,os.path.basename(d)[:31],(x+6,y+21),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1,cv2.LINE_AA)
    if not dirs or not cv2.imwrite(a.output,sheet):
        raise RuntimeError('no models or failed to save sheet')
    print('review candidates:',len(dirs))


if __name__=='__main__':main()
