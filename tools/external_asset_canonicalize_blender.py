"""Normalize an external GLB into a reviewable staging output with Blender.

blender -b --python external_asset_canonicalize_blender.py -- input.glb output.glb metadata.json yaw_deg target_length_m
The operator must measure/choose the real length and model orientation.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector


def bounds(meshes):
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    if not pts:
        raise ValueError('model contains no meshes')
    return (Vector([min(p[k] for p in pts) for k in range(3)]),
            Vector([max(p[k] for p in pts) for k in range(3)]))


def main():
    args = sys.argv[sys.argv.index('--') + 1:]
    if len(args) not in (5, 6, 9):
        raise ValueError('expected input output metadata yaw_deg target_m [axis [length width height]]')
    source, output, meta, yaw, target = args[:5]
    axis = args[5] if len(args) >= 6 else 'x'
    if axis not in ('x', 'y', 'z'):
        raise ValueError('invalid scale axis')
    yaw, target = float(yaw), float(target)
    if not math.isfinite(yaw) or not math.isfinite(target) or target <= 0:
        raise ValueError('invalid yaw or target length')
    if os.path.abspath(source) == os.path.abspath(output):
        raise ValueError('output must differ from source')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source)
    meshes = [x for x in bpy.context.scene.objects if x.type == 'MESH']
    if not meshes:
        raise ValueError('no mesh objects')
    # Apply one common world transform; retain separate meshes and materials.
    for ob in meshes:
        m = ob.matrix_world.copy()
        ob.parent = None
        ob.matrix_world = m
    rot = Matrix.Rotation(math.radians(yaw), 4, 'Z')
    for ob in meshes:
        ob.matrix_world = rot @ ob.matrix_world
    bpy.context.view_layer.update()
    lo, hi = bounds(meshes)
    source_lwh = [float(x) for x in hi-lo]
    length = (hi-lo)['xyz'.index(axis)]
    if not math.isfinite(length) or length <= 1e-9:
        raise ValueError('model has invalid X extent after rotation')
    scale = target / length
    scales = [scale] * 3
    if len(args) == 9:
        requested = [float(v) for v in args[6:9]]
        extents = hi-lo
        if any(not math.isfinite(v) or v < .05 or v > 100 for v in requested):
            raise ValueError('invalid L/W/H')
        if any(v <= 1e-9 for v in extents):
            raise ValueError('model has zero extent')
        scales = [requested[i] / extents[i] for i in range(3)]
    transform = Matrix.Diagonal(Vector((*scales, 1.0)))
    for ob in meshes:
        ob.matrix_world = transform @ ob.matrix_world
    bpy.context.view_layer.update()
    lo, hi = bounds(meshes)
    center = Matrix.Translation((-(lo.x + hi.x)/2, -(lo.y + hi.y)/2, -lo.z))
    for ob in meshes:
        ob.matrix_world = center @ ob.matrix_world
    bpy.context.view_layer.update()
    lo, hi = bounds(meshes)
    if abs(lo.z) > 0.001:
        raise ValueError('ground alignment failed')
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    for ob in meshes:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.export_scene.gltf(filepath=output, export_format='GLB',
                              use_selection=True, export_apply=True)
    with open(meta, 'w', encoding='utf-8') as f:
        json.dump({'source': source, 'path': output, 'yaw_fix_deg': yaw,
                   'target_length_m': target, 'scale_axis': axis, 'scale_factor': scale,
                   'scale_factors_xyz': scales, 'dimension_mode': 'independent' if len(args)==9 else 'proportional',
                   'source_lwh': source_lwh,
                   'lwh': [float(x) for x in hi-lo], 'min_z': float(lo.z),
                   'preview_only': True, 'training_eligible': False}, f, indent=2)


if __name__ == '__main__':
    main()
