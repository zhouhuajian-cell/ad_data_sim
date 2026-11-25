#!/usr/bin/env python3
"""Render a review video from OBJ frames and their matching annotations."""
import argparse
import glob
import json
import os
import subprocess
import tempfile

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory')
    ap.add_argument('output')
    ap.add_argument('--fps', type=float, default=10)
    args = ap.parse_args()
    frames = sorted(glob.glob(os.path.join(args.directory, '*_obj.jpg')))
    if not frames:
        raise ValueError('no frames')
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    with tempfile.TemporaryDirectory(prefix='bbox_video_') as tmp:
        for i, path in enumerate(frames):
            stem = os.path.basename(path)[:-8]
            apath = os.path.join(args.directory, 'annotations', stem + '.json')
            with open(apath, encoding='utf-8') as f:
                ann = json.load(f)
            img = cv2.imread(path)
            if img is None:
                raise ValueError('unreadable frame: ' + path)
            h,w = img.shape[:2]
            for obj in ann.get('objects', []):
                box = obj.get('bbox2d_sim') or obj.get('bbox2d')
                if not box or len(box) != 4:
                    continue
                x1,y1,x2,y2 = [int(round(v)) for v in box]
                if x2 <= x1 or y2 <= y1:
                    continue
                synthetic = obj.get('source') == 'synthetic'
                color = (0, 190, 255) if synthetic else (70, 220, 70)
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                corners = obj.get('corners')
                if synthetic and corners and len(corners) == 8:
                    for a,b in edges:
                        p1=tuple(int(round(v)) for v in corners[a])
                        p2=tuple(int(round(v)) for v in corners[b])
                        cv2.line(img,p1,p2,color,2,cv2.LINE_AA)
                label = '%s %s %s' % ('SIM' if synthetic else 'REAL',
                                      obj.get('class',''), obj.get('track_id',''))
                if synthetic and obj.get('visible_ratio') is not None:
                    label += ' vis %.0f%%' % (100*obj['visible_ratio'])
                y = max(18,min(h-5,y1-5))
                cv2.putText(img,label,(max(0,min(w-1,x1)),y),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(img,label,(max(0,min(w-1,x1)),y),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
            cv2.putText(img,'%d/%d  %s' % (i+1,len(frames),stem), (20,35),
                        cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
            if not cv2.imwrite(os.path.join(tmp,'%06d.png' % i),img):
                raise RuntimeError('frame write failed')
        subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error',
                        '-framerate',str(args.fps),'-i',os.path.join(tmp,'%06d.png'),
                        '-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
                        args.output],check=True)
    print('rendered %d frames -> %s' % (len(frames),args.output))


if __name__ == '__main__':
    main()
