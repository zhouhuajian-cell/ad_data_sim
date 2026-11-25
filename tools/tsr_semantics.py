"""Conservative, country-local TSR asset matching for preview and batches."""
import os
import re


def semantic_name(value, country=''):
    name = os.path.splitext(os.path.basename(str(value or '').strip().lower()))[0]
    prefix = str(country or '').lower() + '_'
    if country and name.startswith(prefix):
        name = name[len(prefix):]
    return re.sub(r'_v\d+$', '', name)


def compatible(source, target, country=''):
    src, dst = semantic_name(source, country), semantic_name(target, country)
    if not src or not dst:
        return False
    if src == dst:
        return True
    # Speed-limit values are deliberately interchangeable only when the
    # detector reports the broad family or a concrete speed limit.
    return bool(re.fullmatch(r'speed_limit(?:_\d+)?', src) and
                re.fullmatch(r'speed_limit_\d+', dst))


def compatible_explicit(source, target, country=''):
    """Explicitly selected replacement may use a detector's broad family.

    Broad-family matches are proposals, not proof of the original subtype.
    Specific left/right classes still cannot be swapped.
    """
    src, dst = semantic_name(source,country), semantic_name(target,country)
    return compatible(src,dst,country) or bool(src in ('mandatory','prohibition','warning') and dst.startswith(src+'_'))


def candidate_shape(observed, inferred, confidence):
    if observed != 'unknown':return observed, 'measured'
    if inferred in ('circle','triangle','rect') and float(confidence) >= .35:
        return inferred, 'class_inferred_review'
    return 'unknown', 'unknown'


def valid_box(box, shape, min_size):
    try:
        h, w = shape[:2]
        x1, y1, x2, y2 = map(float, box)
        if not all(map(lambda n: n == n and abs(n) != float('inf'), (x1,y1,x2,y2))):
            return False
        return 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and min(x2-x1,y2-y1) >= min_size and (x2-x1)*(y2-y1) <= .08*w*h and .45 <= (x2-x1)/(y2-y1) <= 2.2
    except (TypeError, ValueError):
        return False
