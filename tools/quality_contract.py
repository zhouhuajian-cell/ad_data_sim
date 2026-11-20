"""统一的输出质量检查。

本模块检查：
1. 标注框有效；
2. 现有像素质量门控；
3. 模型检出的标牌与目标位置重合。

它不证明细类别或限速数字正确，也不替代原始目标身份复核。
"""
import cv2
import numpy as np

from auto_qc import qc_check, _get_model


def iou(a, b):
    """两个 [x1,y1,x2,y2] 框的 IoU。退化框返回 0。"""
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    if area_a <= 0 or area_b <= 0:
        return 0.0

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(area_a + area_b - inter, 1e-9)


def aggregate(records):
    """把逐目标记录聚合成帧状态。

    - 任一 failed      -> "failed"
    - 无 failed 但有 unchecked -> "unchecked"（可转人工复核，不当成失败）
    - 全部 passed      -> "passed"
    """
    if not records:
        return "failed"
    statuses = {r.get("status") for r in records}
    if "failed" in statuses:
        return "failed"
    if "unchecked" in statuses:
        return "unchecked"
    return "passed"


def check_frame(
    original,
    composed,
    objects,
    min_size=24,
    conf_min=0.35,
    min_iou=0.50,
    edge_max=38.0,
    bright_max=38.0,
):
    """返回 (帧状态, 检查记录)。

    帧状态为 "passed" / "failed" / "unchecked" 三态之一。
    调用方应只把 "passed" 计入自动合格集，"unchecked" 转人工复核。
    阈值是初始配置，需要用固定真实样本集标定。
    """
    records = []
    if not objects:
        return "failed", [{"status": "failed", "reason": "no_objects"}]
    if getattr(original, "shape", None) != getattr(composed, "shape", None):
        return "failed", [{"status": "failed", "reason": "image_shape_mismatch"}]

    height, width = composed.shape[:2]
    model = None

    for index, obj in enumerate(objects):
        record = {
            "object_index": index,
            "status": "failed",
            "semantic_status": "not_checked",
        }

        try:
            box = np.asarray(obj["bbox"], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("invalid_bbox")

            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                # 越界属于明确失败，不走 except 兜底
                record.update(status="failed", reason="bbox_out_of_bounds")
                records.append(record)
                continue

            if min(x2 - x1, y2 - y1) < min_size:
                # 小目标是 Recall 的重要来源，不能一律丢弃。
                # 标记为 unchecked 并降低细属性监督，由任务协议决定取舍。
                record.update(
                    status="unchecked",
                    reason="target_too_small",
                    size_px=float(min(x2 - x1, y2 - y1)),
                    suggested_action="keep_for_recall_drop_fine_attributes",
                )
                records.append(record)
                continue

            passed, pixel_info = qc_check(
                original,
                composed,
                box,
                obj.get("scene_shape", "circle"),
                edge_jump_max=edge_max,
                bright_diff_max=bright_max,
            )
            record["pixel_check"] = pixel_info
            if not passed:
                record["reason"] = "pixel_check_failed"
                records.append(record)
                continue

            if model is None:
                model = _get_model()
            if model is None:
                record.update(status="unchecked", reason="model_unavailable")
                records.append(record)
                continue

            pad = int(max(x2 - x1, y2 - y1) * 0.35)
            left = max(0, int(np.floor(x1)) - pad)
            top = max(0, int(np.floor(y1)) - pad)
            right = min(width, int(np.ceil(x2)) + pad)
            bottom = min(height, int(np.ceil(y2)) + pad)

            crop = composed[top:bottom, left:right]
            ch, cw = crop.shape[:2]
            if ch < 2 or cw < 2:
                record.update(status="unchecked", reason="crop_too_small")
                records.append(record)
                continue

            scale = max(1.0, 320.0 / max(ch, cw))
            rw = max(1, int(round(cw * scale)))
            rh = max(1, int(round(ch * scale)))

            crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_CUBIC)
            sx, sy = rw / cw, rh / ch

            try:
                prediction = model(crop, conf=conf_min, verbose=False)[0]
            except Exception as exc:
                record.update(
                    status="unchecked",
                    reason="inference_failed",
                    error=str(exc)[:200],
                )
                records.append(record)
                continue

            best = None
            if prediction.boxes is not None:
                for detected in prediction.boxes:
                    bx1, by1, bx2, by2 = (
                        detected.xyxy[0].detach().cpu().tolist()
                    )
                    mapped = [
                        bx1 / sx + left,
                        by1 / sy + top,
                        bx2 / sx + left,
                        by2 / sy + top,
                    ]
                    overlap = iou(box, mapped)
                    if best is None or overlap > best["iou"]:
                        best = {
                            "iou": float(overlap),
                            "confidence": float(detected.conf[0].item()),
                            "class_id": int(detected.cls[0].item()),
                        }

            record["model_check"] = best
            if best is None or best["iou"] < min_iou:
                # 模型检不出 ≠ 样本无价值。困难样本是 Recall 的来源，
                # 转人工复核而不是直接丢弃。
                record.update(status="unchecked",
                              reason="no_matching_detection",
                              suggested_action="review_hard_example")
            else:
                record.update(status="passed", reason="OK")

        except Exception as exc:
            record.update(reason="check_error", error=str(exc)[:200])

        records.append(record)

    return aggregate(records), records
