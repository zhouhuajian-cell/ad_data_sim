"""Inspect a lossless marking/sign source and create a review preview."""
import io
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

from PIL import Image
Image.MAX_IMAGE_PIXELS = 40_000_000


def main():
    source, output, meta = sys.argv[1:4]
    raw = open(source, 'rb').read()
    if raw.lstrip().startswith((b'<svg', b'<?xml')):
        from defusedxml import ElementTree as SafeET
        import cairosvg
        root = SafeET.fromstring(raw)
        if root.tag.split('}')[-1] != 'svg':
            raise ValueError('not an SVG')
        viewbox = root.attrib.get('viewBox', '').replace(',', ' ').split()
        if len(viewbox) != 4:
            raise ValueError('SVG must provide a viewBox')
        vw, vh = float(viewbox[2]), float(viewbox[3])
        if vw <= 0 or vh <= 0 or not 0.01 <= vh/vw <= 50:
            raise ValueError('unsupported SVG aspect ratio')
        for node in root.iter():
            if node.tag.split('}')[-1].lower() in ('script', 'foreignobject', 'image', 'style'):
                raise ValueError('SVG contains unsupported active/external content')
            for key, value in node.attrib.items():
                low = value.lower().replace(' ', '')
                if key.split('}')[-1].lower().startswith('on') or 'url(' in low or 'javascript:' in low or (key.split('}')[-1] in ('href','src') and not value.startswith('#')):
                    raise ValueError('SVG external references or scripts are unsupported')
        # Render to a bounded high-resolution PNG; preserve original SVG separately.
        ratio = vh / vw
        output_width = max(16, int(min(2048, 8192 / ratio,
                                      math.sqrt(35_000_000 / ratio))))
        png = cairosvg.svg2png(bytestring=raw, output_width=output_width)
        im = Image.open(io.BytesIO(png))
        kind = 'svg'
    else:
        im = Image.open(io.BytesIO(raw))
        if im.format not in ('PNG', 'WEBP'):
            raise ValueError('use PNG, WebP or SVG')
        kind = im.format.lower()
    im.load()
    if min(im.size) < 16 or max(im.size) > 8192:
        raise ValueError('invalid image dimensions')
    rgba = im.convert('RGBA')
    alpha = rgba.getchannel('A')
    info = {'format':kind,'width':rgba.width,'height':rgba.height,
            'alpha_min':alpha.getextrema()[0], 'alpha_max':alpha.getextrema()[1],
            'source_is_vector':kind=='svg','training_eligible':False}
    rgba.save(output,'PNG',optimize=True)
    with open(meta,'w',encoding='utf-8') as f:json.dump(info,f,indent=2)


if __name__=='__main__':main()
