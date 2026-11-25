import bpy, numpy as np, sys
argv = sys.argv
argv = argv[argv.index("--")+1:] if "--" in argv else []
src, dst = argv[0], argv[1]
img = bpy.data.images.load(src)
w, h = img.size
px = np.array(list(img.pixels), dtype=np.float32).reshape(h, w, 4)
z = px[:, :, 0]
# Blender pixels 自底向上存储 -> 翻转成自上而下
z = np.flipud(z).copy()
np.save(dst, z)
v = z[np.isfinite(z) & (z > 0.01) & (z < 1e5)]
print("[EXR2NPY] saved shape=%s  range=%.2f..%.2f  (flipped)" % (
    z.shape, v.min(), v.max()))
