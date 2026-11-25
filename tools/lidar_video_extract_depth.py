import bpy,glob,os,sys,numpy as np
out=sys.argv[sys.argv.index('--')+1]; files=sorted(glob.glob(os.path.join(out,'z','*','*.exr')))
for i,p in enumerate(files):
    img=bpy.data.images.load(p); w,h=img.size[:]; a=np.empty(w*h*4,dtype=np.float32);img.pixels.foreach_get(a);a=a.reshape(h,w,4)[::-1,:,0]
    np.save(os.path.join(os.path.dirname(p),'depth.npy'),a);bpy.data.images.remove(img)
    print('depth %d/%d'%(i+1,len(files)),flush=True)
