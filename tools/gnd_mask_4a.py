import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


class MapillaryMarkingExtractor:
    def __init__(self, model_name="/data/models/mask2former_mapillary", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(self.device)
        self.model.eval()

        # 动态匹配所有与地面标线/斑马线相关的类别 ID，避免硬编码出错
        self.marking_ids = [
            idx for idx, label in self.model.config.id2label.items()
            if "marking" in label.lower() or "crosswalk" in label.lower()
        ]
        # VEHICLE-MASK-API: 车辆/行人类别（遮挡合成用）
        _veh_kw = ("rider", "bicycle", "bus", "car", "caravan",
                   "motorcycle", "other vehicle", "trailer", "truck",
                   "person")
        self.vehicle_ids = [
            idx for idx, label in self.model.config.id2label.items()
            if label.lower() in _veh_kw
        ]

    @torch.no_grad()
    def get_vehicle_mask(self, img_bgr, prob_thresh=0.5):
        """VEHICLE-MASK-API: 返回车辆/行人像素 mask (uint8 0/255)。"""
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img,
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        class_probs = F.softmax(outputs.class_queries_logits,
                                dim=-1)[..., :-1]
        mask_probs = outputs.masks_queries_logits.sigmoid()
        sem_logits = torch.einsum("bqc,bqhw->bchw",
                                  class_probs, mask_probs)
        sem_logits = F.interpolate(sem_logits, size=(h, w),
                                   mode="bilinear", align_corners=False)
        if not self.vehicle_ids:
            return np.zeros((h, w), np.uint8)
        vprob = sem_logits[0, self.vehicle_ids, :, :].sum(dim=0).cpu().numpy()
        return (vprob > prob_thresh).astype(np.uint8) * 255

    @torch.no_grad()
    def get_marking_mask(self, img_bgr, prob_thresh=0.12):
        """
        SOFT-PROB: 用软概率阈值替代 argmax，专门捞回中远景细小标线
        prob_thresh: 0.10 ~ 0.18，越低对远景细线越敏感
        """
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        inputs = self.processor(images=pil_img,
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # 1. 手动合成每个类别的像素级 logits
        class_probs = F.softmax(outputs.class_queries_logits,
                                dim=-1)[..., :-1]  # 去掉 no-object
        mask_probs = outputs.masks_queries_logits.sigmoid()
        sem_logits = torch.einsum("bqc,bqhw->bchw",
                                  class_probs, mask_probs)

        # 2. 上采样回原图尺寸
        sem_logits = F.interpolate(sem_logits, size=(h, w),
                                   mode="bilinear", align_corners=False)

        # 3. 累加所有 marking / crosswalk 类别的置信度
        marking_prob = sem_logits[0, self.marking_ids, :,
                                  :].sum(dim=0).cpu().numpy()

        # 4. 低阈值截断
        mask = (marking_prob > prob_thresh).astype(np.uint8) * 255
        return mask


def extract_raw_paint(img_bgr, bright_thr=14, shadow_thr=8,
                      far_bright_thr=4, far_shadow_thr=3,
                      far_zone=None):
    """无形状过滤的轻量 Top-Hat 抠油漆笔画。

    far_zone: (y0, y1) 远景区域，该区间内用更低的阈值（细线对比度低）
    """
    import cv2
    import numpy as np

    b, g, r = cv2.split(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    h, w = L.shape[:2]

    # 大面积背景估计
    k_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    bg = cv2.morphologyEx(L, cv2.MORPH_OPEN, k_bg)
    tophat = cv2.subtract(L, bg)

    med = float(np.median(L))
    is_shadow = L < (med * 0.72)

    # 默认阈值
    paint = np.where(is_shadow, tophat > shadow_thr, tophat > bright_thr)

    # ZONE-B-2: 远景区用更低阈值 + 小核 Top-Hat（抓 1~2px 细线）
    if far_zone is not None:
        fy0, fy1 = int(far_zone[0]), int(far_zone[1])
        if fy1 > fy0:
            k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            bg_far = cv2.morphologyEx(L, cv2.MORPH_OPEN, k_small)
            tophat_far = cv2.subtract(L, bg_far)
            far_paint = np.where(
                is_shadow,
                tophat_far > far_shadow_thr,
                tophat_far > far_bright_thr)
            paint[fy0:fy1, :] = far_paint[fy0:fy1, :]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = ((hh >= 15) & (hh <= 38) & (ss >= 70) & (vv >= 90))
    white = (tophat > 26) & (ss < 75) & (L > 135)

    raw = (paint | yellow | white).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    _n, _lab, _st, _ = cv2.connectedComponentsWithStats(raw, 8)
    _clean = np.zeros_like(raw)
    thr_area = 12
    if far_zone is not None:
        # 远景细线成块小，面积下限降到 4
        fy0, fy1 = int(far_zone[0]), int(far_zone[1])
        for _i in range(1, _n):
            a = _st[_i, cv2.CC_STAT_AREA]
            cy = _st[_i, cv2.CC_STAT_TOP] + _st[_i, cv2.CC_STAT_HEIGHT] / 2.0
            lim = 4 if (fy0 <= cy <= fy1) else thr_area
            if a >= lim:
                _clean[_lab == _i] = 255
    else:
        for _i in range(1, _n):
            if _st[_i, cv2.CC_STAT_AREA] >= thr_area:
                _clean[_lab == _i] = 255
    return _clean


def extract_far_thin_line(img_bgr, road_mask, vehicle_mask, far_zone,
                          thr=15):
    """FAR-THIN-LINE: 正交双 1D 顶帽，专抓远景细虚线与细实线。

    ORTHO-TOPHAT 核心原理（形态学铁律）：
      顶帽的核尺寸必须【大于目标宽度】，开运算才能抹掉目标、顶帽才有响应。

      · 横向短条（w 30~50px, h 6~10px）：用【竖直核 (1,17)】
        17 > 10，竖直开运算瞬间抹平横条 → 顶帽响应 40~70
      · 竖向细线（h 30~80px, w 1~12px）：用【水平核 (15,1)】
        15 > 12，水平开运算抹平整条竖线 → 顶帽响应强

      两个分支取并集。

    三重门控（防误擦树影/光斑）：
      ① 仅 far_zone 内 + road_mask 水平内缩（避路缘石/护栏）
      ② 扣除膨胀后的 vehicle_mask（护车轮底盘）
      ③ 连通域走向门控（横条 w<=80&h<=14&w>=3h / 竖条 h<=80&w<=14&h>=3w）
    """
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    y0, y1 = int(far_zone[0]), int(far_zone[1])
    out = np.zeros((h, w), np.uint8)
    if y1 <= y0:
        return out, {"reason": "empty_zone"}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ---------- 分支 A：横向短条（竖直核 1x17）----------
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17))
    th_horiz = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_v)

    # ---------- 分支 B：竖向细线（水平核 15x1）----------
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    th_vert = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_h)

    # ---------- 作用范围 ----------
    # ROAD-IN-ENVELOPE: 先填车洞再内缩，避免沿车辆轮廓啃掉路面
    #   road_mask 在有车处是镂空的，直接 erode 会把车旁 12px 路面一起削掉
    #   先 road_envelope = road_mask | vehicle_mask，再轻微内缩 6px
    road_env = cv2.bitwise_or(road_mask, vehicle_mask)
    # 向上扩 15px（避免切掉上沿附近的远景标线）
    _pad = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    road_env = cv2.dilate(road_env, _pad, iterations=1)
    # 水平内缩 6px（只避外侧路缘石，不啃车辆轮廓）
    k_er = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
    road_in = cv2.erode(road_env, k_er, iterations=1)
    # 最终合法路面 = 内缩后的外轮廓 & ~vehicle_mask
    road_in = cv2.bitwise_and(road_in, cv2.bitwise_not(vehicle_mask))
    zone = np.zeros((h, w), np.uint8)
    zone[y0:y1, :] = 255
    roi = cv2.bitwise_and(road_in, zone)

    # VEH-FAR-SLIM: 远景扣车只外扩 1px（原来是 2px，远景车小易误吃旁侧虚线）
    veh_x = cv2.dilate(vehicle_mask,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    roi = cv2.bitwise_and(roi, cv2.bitwise_not(veh_x))
    if np.count_nonzero(roi) < 100:
        return out, {"reason": "tiny_roi"}

    # ---------- 候选 ----------
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = ((hh >= 15) & (hh <= 38) & (ss >= 60) & (vv >= 80))

    in_roi = (roi > 127)
    cand_h = ((th_horiz > thr) & in_roi) | (yellow & in_roi)
    cand_v = ((th_vert > thr) & in_roi) | (yellow & in_roi)

    stats = {"horiz_px": int(cand_h.sum()), "vert_px": int(cand_v.sum())}

    # ---------- 分支 A 过滤：横条 ----------
    mA = (cand_h.astype(np.uint8)) * 255
    nA, labA, stA, _ = cv2.connectedComponentsWithStats(mA, 8)
    cntA = 0
    for i in range(1, nA):
        bw = stA[i, cv2.CC_STAT_WIDTH]
        bh = stA[i, cv2.CC_STAT_HEIGHT]
        area = stA[i, cv2.CC_STAT_AREA]
        if bw <= 90 and bh <= 16 and bw >= 2 * max(bh, 1) and 12 <= area <= 800:
            out[labA == i] = 255
            cntA += 1

    # ---------- 分支 B 过滤：竖条 ----------
    mB = (cand_v.astype(np.uint8)) * 255
    nB, labB, stB, _ = cv2.connectedComponentsWithStats(mB, 8)
    cntB = 0
    for i in range(1, nB):
        bw = stB[i, cv2.CC_STAT_WIDTH]
        bh = stB[i, cv2.CC_STAT_HEIGHT]
        area = stB[i, cv2.CC_STAT_AREA]
        if bh <= 90 and bw <= 16 and bh >= 2 * max(bw, 1) and 12 <= area <= 800:
            out[labB == i] = 255
            cntB += 2

    stats["horiz_comp"] = cntA
    stats["vert_comp"] = cntB
    stats["thr"] = thr
    return out, stats


def build_final_erase_mask_4a(img_bgr, v3_mask, drivable_mask, vehicle_mask,
                               extractor, prob_thresh=0.15, dilate_px=3):
    """ZONE-B: 动态路面分区 + 远景放宽 + 精细膨胀。

    分区依据 road_mask 的纵坐标范围 [y_min, y_max]（不写死 0.5H）：
      远景区 = 顶部 35%  → prob_thresh=0.05, Top-Hat 6/5, dilate 2px
      近中景 = 底部 65%  → prob_thresh=0.15, Top-Hat 14/8, dilate 3px
    """
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    tot = float(h * w)

    # ---- ① 动态分区（基于路面实际纵坐标）----
    ys, xs = np.where(drivable_mask > 127)
    if len(ys) > 100:
        y_min, y_max = int(ys.min()), int(ys.max())
        # ROAD-EXPAND-UP: 远景上沿向上扩 15px
        y_min = max(0, y_min - 15)
    else:
        y_min, y_max = int(h * 0.45), h
    # ZONE-B2: 从 35% 扩到 55%，覆盖实测残留区间 y=610~786
    y_split = int(y_min + (y_max - y_min) * 0.55)
    far_zone = (y_min, y_split)

    # ---- ② 分区语义圈地 ----
    sem_near = extractor.get_marking_mask(img_bgr, prob_thresh=prob_thresh)
    sem_far = extractor.get_marking_mask(img_bgr, prob_thresh=0.05)
    sem_mask = sem_near.copy()
    if y_split > y_min:
        sem_mask[y_min:y_split, :] = sem_far[y_min:y_split, :]

    expanded_road = cv2.dilate(
        drivable_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    sem_roi = cv2.bitwise_and(sem_mask, expanded_road)

    # ---- ③ 分区 raw_paint（远景联动放宽）----
    raw_paint = extract_raw_paint(img_bgr, far_zone=far_zone)

    # ---- ④ 交集 ----
    precise = cv2.bitwise_and(sem_roi, raw_paint)

    # ---- ④.5 FAR-THIN-MERGE: 远景细线兜底 ----
    thin_line, thin_info = extract_far_thin_line(
        img_bgr, drivable_mask, vehicle_mask, far_zone)
    if y_split > y_min:
        precise_far = precise.copy()
        precise_far[y_min:y_split, :] = cv2.bitwise_or(
            precise[y_min:y_split, :], thin_line[y_min:y_split, :])
    else:
        precise_far = precise

    # ---- ⑤ 分区膨胀（远景 2px，近中景 3px）----
    near_d = cv2.dilate(
        precise_far, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (dilate_px, dilate_px)))
    # DIL2-LOCK: 远景膨胀 2px（4px 会把擦除率从 4.20% 拉到 6.70%，负优化）
    far_d = cv2.dilate(
        precise_far, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    final_mask = near_d.copy()
    if y_split > y_min:
        final_mask[y_min:y_split, :] = far_d[y_min:y_split, :]

    # ---- ⑥ 车辆保护：vehicle_mask 外扩 2px 再扣 ----
    veh_expand = cv2.dilate(vehicle_mask,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(veh_expand))

    try:
        extractor._last_stats = {
            "far_zone": [int(y_min), int(y_split)],
            "sem_near_pct": round(np.count_nonzero(sem_near) / tot * 100, 2),
            "sem_far_pct": round(np.count_nonzero(sem_far) / tot * 100, 2),
            "raw_paint_pct": round(np.count_nonzero(raw_paint) / tot * 100, 2),
            "precise_pct": round(np.count_nonzero(precise) / tot * 100, 2),
            "thin_line_pct": round(np.count_nonzero(thin_line) / tot * 100, 2),
            "thin_info": thin_info,
            "final_pct": round(np.count_nonzero(final_mask) / tot * 100, 2),
        }
    except Exception:
        pass
    return final_mask
