"""Shared current TSR renderer for batch jobs and single-frame preview."""
def render_sign(image, bbox, asset_rgba, dst_shape, jpeg_quality=60,
                pad_ratio=1.04, rotate=False, feather_ratio=0.0, coverage_bbox=None):
    from pipeline6 import run_pipeline6
    if rotate:
        try:
            from sign_rotate import estimate_angle, rotate_asset
            angle, _ = estimate_angle(image, bbox)
            if abs(angle) >= 1.0:
                asset_rgba = rotate_asset(asset_rgba, angle)
        except Exception:
            pass  # Same rotation fallback as batch_walk.
    if dst_shape == 'unknown':
        dst_shape = 'circle'
    result, info = run_pipeline6(image, bbox, asset_rgba, dst_shape,
                                 jpeg_quality=jpeg_quality, pad_ratio=pad_ratio,
                                 feather_ratio=feather_ratio)
    # A partial red contour can shrink the rendered sign enough to expose the
    # original face.  Reject such results rather than produce a mislabeled sign.
    if info.get('ok') and dst_shape == 'circle':
        rendered = info.get('bbox')
        if rendered and len(rendered) == 4:
            original = bbox if coverage_bbox is None else coverage_bbox
            sw, sh = float(original[2])-float(original[0]), float(original[3])-float(original[1])
            rw, rh = float(rendered[2])-float(rendered[0]), float(rendered[3])-float(rendered[1])
            if sw <= 0 or sh <= 0 or rw < sw * .85 or rh < sh * .85:
                return image.copy(), {'ok': False, 'reason': 'refined_bbox_undercoverage',
                                      'source_bbox': list(original), 'refined_bbox': list(rendered)}
    return result, info
