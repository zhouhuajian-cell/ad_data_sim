import sys
from pathlib import Path
import bpy
job=Path(sys.argv[sys.argv.index('--')+1])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=str(job/'paint.obj'))
if not any(o.type=='MESH' for o in bpy.context.scene.objects):
    raise RuntimeError('贴图结果没有网格')
if not bpy.data.materials:
    raise RuntimeError('贴图结果没有材质')
for material in bpy.data.materials:
    material.use_nodes=True
    nodes=material.node_tree.nodes
    principled=next((n for n in nodes if n.type=='BSDF_PRINCIPLED'),None)
    if principled is None: raise RuntimeError('没有 Principled BSDF 材质')
    for name,socket in [('paint.jpg','Base Color'),('paint_metallic.jpg','Metallic'),('paint_roughness.jpg','Roughness')]:
        path=job/name
        if not path.is_file(): raise RuntimeError('缺少贴图：'+name)
        image=bpy.data.images.load(str(path),check_existing=True)
        if socket!='Base Color': image.colorspace_settings.name='Non-Color'
        tex=nodes.new('ShaderNodeTexImage');tex.image=image
        material.node_tree.links.new(tex.outputs['Color'],principled.inputs[socket])
bpy.ops.export_scene.gltf(filepath=str(job/'model.glb'),export_format='GLB')
