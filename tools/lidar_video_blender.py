"""Render static cones and one moving 3D car in first LiDAR frame (preview)."""
import bpy, json, math, os, sys
from mathutils import Matrix,Vector
sys.path.insert(0,'/opt/ue5_synth/tools')
from cv_to_blender_cam import set_blender_camera
import numpy as np
arg=sys.argv[sys.argv.index('--')+1:]; out=arg[0]
data=json.load(open(os.path.join(out,'scene.json')));os.makedirs(os.path.join(out,'rgba'),exist_ok=True);os.makedirs(os.path.join(out,'z'),exist_ok=True)
sc=bpy.context.scene;bpy.ops.wm.read_factory_settings(use_empty=True);sc=bpy.context.scene
sc.render.engine='CYCLES';sc.render.resolution_x=960;sc.render.resolution_y=540;sc.render.resolution_percentage=100
sc.render.film_transparent=True;sc.cycles.samples=12;sc.cycles.use_denoising=True;sc.cycles.max_bounces=4
try:
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    for dev in prefs.devices:dev.use=True
    sc.cycles.device='GPU'
except Exception:sc.cycles.device='CPU'
world=bpy.data.worlds.new('World');sc.world=world;world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.64,.68,.74,1);world.node_tree.nodes['Background'].inputs[1].default_value=1.8
sun=bpy.data.lights.new('Sun','SUN');sun.energy=1.0;sun.angle=math.radians(25);so=bpy.data.objects.new('Sun',sun);sc.collection.objects.link(so);so.rotation_euler=(math.radians(50),0,math.radians(140))
cd=bpy.data.cameras.new('Camera');cam=bpy.data.objects.new('Camera',cd);sc.collection.objects.link(cam);sc.camera=cam

def import_model(path,name,normalization,target):
    before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=path)
    meshes=[o for o in bpy.data.objects if o not in before and o.type=='MESH'];
    if not meshes:raise ValueError('no mesh '+path)
    root=bpy.data.objects.new(name,None);sc.collection.objects.link(root)
    for o in meshes:
        mat=o.matrix_world.copy();o.parent=root;o.matrix_world=mat
    bpy.context.view_layer.update()
    coords=[o.matrix_world@Vector(c) for o in meshes for c in o.bound_box]
    lo=Vector(tuple(min(p[j] for p in coords) for j in range(3)));hi=Vector(tuple(max(p[j] for p in coords) for j in range(3)))
    extent=hi-lo; factor=target/max(extent[normalization],1e-6);root.scale=(factor,)*3;bpy.context.view_layer.update()
    coords=[o.matrix_world@Vector(c) for o in meshes for c in o.bound_box]
    lo=Vector(tuple(min(p[j] for p in coords) for j in range(3)));hi=Vector(tuple(max(p[j] for p in coords) for j in range(3)))
    root.location=(-(lo.x+hi.x)/2,-(lo.y+hi.y)/2,-lo.z);bpy.context.view_layer.update()
    carrier=bpy.data.objects.new(name+'_placement',None);sc.collection.objects.link(carrier);root.parent=carrier
    print('MODEL',name,'extents',tuple(round(x,3) for x in extent),'scale',factor,flush=True)
    return carrier

cone='/data/avsynth/assets_global/OBJ_3D/GLOBAL/engine_candidates/traffic-cone-4867e2a7.glb'
car='/data/assets_global/OBJ_3D/canonical/00_car_1e478fb5.glb';z=data['ground_z']
cones=[]
for i in range(3):
    o=import_model(cone,'cone%d'%i,2,.75);o.location=(22+2.0*i,-4.0,z);cones.append(o)
auto=import_model(car,'car',0,4.6);auto.rotation_euler.z=math.pi
# Ground plane is a shadow catcher; only foreground pixels participate in depth comparison.
bpy.ops.mesh.primitive_plane_add(size=150,location=(30,0,z));ground=bpy.context.object;ground.is_shadow_catcher=True
sc.view_layers[0].use_pass_z=True;sc.use_nodes=True;nt=sc.node_tree;nt.nodes.clear();rl=nt.nodes.new('CompositorNodeRLayers');cp=nt.nodes.new('CompositorNodeComposite');nt.links.new(rl.outputs['Image'],cp.inputs['Image'])
fo=nt.nodes.new('CompositorNodeOutputFile');fo.format.file_format='OPEN_EXR';fo.format.color_depth='32';fo.format.color_mode='RGBA';fo.file_slots[0].path='depth_';nt.links.new(rl.outputs['Depth'],fo.inputs['Image'])
sc.render.image_settings.file_format='PNG';sc.render.image_settings.color_mode='RGBA'
for i,f in enumerate(data['frames']):
    K=np.asarray(f['K']);T=np.asarray(f['T_lidar2cam'])@np.linalg.inv(np.asarray(f['T_lidar2first']))
    set_blender_camera(sc,cam,K,T,960,540)
    # A single explicit kinematic hypothesis; not measured real-world velocity.
    auto.location=(37.0+2.0*i*.1,-3.5,z)
    fo.base_path=os.path.join(out,'z',f['stem']);os.makedirs(fo.base_path,exist_ok=True)
    sc.render.filepath=os.path.join(out,'rgba',f['stem']+'.png');bpy.ops.render.render(write_still=True)
    print('RENDERED',i+1,len(data['frames']),f['stem'],flush=True)
