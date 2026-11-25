"""Isolated, serialized Hunyuan geometry-preview jobs."""
import fcntl
import io
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path('/data/avsynth/studio_image3d')
PYTHON = '/data/avsynth/venvs/hunyuan3d/bin/python'
WORKER = '/opt/ue5_synth/tools/studio_image3d_worker.py'

def job_dir(task):
    if not re.fullmatch(r'[0-9a-f]{32}', task or ''):
        raise ValueError('任务编号无效')
    path = ROOT / task
    if not path.is_dir():
        raise ValueError('任务不存在')
    return path

def submit(raw):
    if not raw or len(raw) > 20 * 1024 * 1024:
        raise ValueError('图片为空或超过 20 MB')
    with Image.open(io.BytesIO(raw)) as img:
        if img.format not in ('PNG', 'JPEG', 'WEBP'):
            raise ValueError('只支持 PNG、JPG 或 WebP')
        if img.width * img.height > 30000000:
            raise ValueError('图片尺寸过大')
        img = ImageOps.exif_transpose(img).convert('RGBA')
        if min(img.size) < 256:
            raise ValueError('图片短边至少需要 256 像素')
        img.thumbnail((2048, 2048))
        task = uuid.uuid4().hex
        path = ROOT / task
        path.mkdir(parents=True, mode=0o700)
        img.save(path / 'input.png')
    (path / 'status.json').write_text(json.dumps({'state':'queued'}), encoding='utf8')
    with open(path / 'worker.log', 'ab') as log:
        subprocess.Popen([PYTHON, WORKER, task], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=log, start_new_session=True)
    return {'ok': True, 'task': task}

def status(task):
    path = job_dir(task)
    return dict(json.loads((path / 'status.json').read_text(encoding='utf8')), ok=True)

def model(task):
    path = job_dir(task)
    if status(task).get('state') != 'completed' or not (path / 'model.glb').is_file():
        raise ValueError('模型尚未生成')
    return str(path / 'model.glb')
