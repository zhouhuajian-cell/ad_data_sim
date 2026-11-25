import sys
from pathlib import Path
job = Path(sys.argv[1])
sys.path.insert(0, '/data/avsynth/third_party/Hunyuan3D-2')
import torch
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
with Image.open(job / 'input.png') as im:
    image=im.convert('RGBA')
pipe=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    '/data/models/Hunyuan3D-2mini', subfolder='hunyuan3d-dit-v2-mini',
    variant='fp16', use_safetensors=True)
with torch.inference_mode():
    mesh=pipe(image=image,num_inference_steps=20,octree_resolution=256,num_chunks=10000)[0]
if mesh is None or len(mesh.vertices)==0:
    raise RuntimeError('生成的几何模型为空')
mesh.export(str(job/'shape.glb'))
