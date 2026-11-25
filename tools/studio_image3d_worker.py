"""Serialize geometry and PBR stages, publishing only a verified PBR GLB."""
import fcntl
import json
import os
import struct
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path('/data/avsynth/studio_image3d')
job = ROOT / sys.argv[1]
TOOLS = Path('/opt/ue5_synth/tools')
PBR_ROOT = '/data/avsynth/third_party/HunyuanPBR-low-vram'
PBR_CLI = PBR_ROOT + '/.venv/bin/hunyuan3d'
BLENDER = '/data/blender-4.2.23-linux-x64/blender'
current_stage = 'setup'

def update(state, **kw):
    tmp = job / 'status.tmp'
    tmp.write_text(json.dumps(dict(state=state, **kw), ensure_ascii=False), encoding='utf8')
    os.replace(tmp, job / 'status.json')

def run(cmd, stage, timeout, cwd=None, env=None):
    global current_stage
    current_stage = stage
    update('running', stage=stage)
    subprocess.run(cmd, cwd=cwd, env=env, check=True, timeout=timeout)

def verify_glb(path):
    with path.open('rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        size, kind = struct.unpack('<I4s', f.read(8))
        if magic != b'glTF' or version != 2 or kind != b'JSON' or length != path.stat().st_size:
            raise RuntimeError('生成的 GLB 格式不正确')
        doc = json.loads(f.read(size).decode('utf8').rstrip(' \x00'))
    for mat in doc.get('materials', []):
        pbr = mat.get('pbrMetallicRoughness', {})
        if 'baseColorTexture' in pbr and 'metallicRoughnessTexture' in pbr:
            return
    raise RuntimeError('生成模型缺少颜色或金属度/粗糙度贴图')

try:
    with open(ROOT / '.gpu.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from PIL import Image
        with Image.open(job / 'input.png') as im:
            opaque = im.convert('RGBA').getchannel('A').getextrema()[0] == 255
        run([sys.executable, str(TOOLS / 'studio_image3d_shape.py'), str(job)], 'geometry', 600)
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        run([PBR_CLI, '--cache-dir', '/data/avsynth/pbr_test_cache', 'texture',
             '--mesh', str(job / 'shape.glb'), '--image', str(job / 'input.png'),
             '--output', str(job / 'paint')], 'texture', 1800, cwd=PBR_ROOT, env=env)
        run([BLENDER, '-b', '--python-exit-code', '7', '--python',
             str(TOOLS / 'studio_image3d_pbr_export.py'), '--', str(job)], 'packaging', 300)
        verify_glb(job / 'model.glb')
        update('completed', opaque_input=opaque, has_base_color=True,
               has_metallic_roughness=True, has_normal_texture=False,
               note='生成式 PBR 模型，未经过质量验收，不自动取得训练资格')
except Exception as e:
    traceback.print_exc()
    msg = '任务超时' if isinstance(e, subprocess.TimeoutExpired) else str(e)
    update('failed', stage=current_stage, error=msg[:500])
