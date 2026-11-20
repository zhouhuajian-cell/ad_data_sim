以下是一套完整的 **4 周 OBJ + GND 混合仿真落地执行方案**，全部脚本统一挂载在 `/data/avsynth/HybridPipeline/` 下，与现有 `/data/avsynth` 工程并行运行、零破坏原有小样链路。

---

## 全局目录结构（一次性初始化）

```text
/data/avsynth/
├── Config/DefaultEngine.ini                     # 关 Lumen/光追，开 CustomDepth+Stencil
├── state/jobs.db                                # SQLite 持久化任务与帧级状态库
└── HybridPipeline/
    ├── configs/
    │   ├── assets_registry.json                 # Week 2: 真实 3D 资产库与细分类映射（替代空 ASSET_HINTS）[cite: 16]
    │   └── camera_profiles.json                 # 实车前视相机内外参标定库
    ├── manifests/                               # JSONL 任务清单（按批次落盘）
    ├── scripts/
    │   ├── state_db.py                          # Week 1: SQLite 状态管理
    │   ├── gen_week1_manifest.py                # Week 1: 100 帧压测清单生成
    │   ├── ue_batch_worker.py                   # Week 1-2: UE 常驻前景批渲染 + 像素级遮挡 Pass
    │   ├── validate_and_postproc.py             # Week 1: 日志扫雷 + Shadow Matte + 强校验
    │   ├── obj_3d_to_real_compositor.py         # Week 2: 阴影压底 + V4.4 降质融合 + Gate 4
    │   ├── gnd_ipm_engine.py                    # Week 3: 实拍 2.5D IPM 标线难例 + 占道遮挡联动
    │   ├── export_dataset.py                    # Week 4: 版本化 COCO + Polyline + Mask 导出
    │   └── batch_orchestrator.py                # Week 4: 三段解耦大批量调度器
    └── run_week1_batch.sh                       # Week 1 压测入口

```

---

## Week 1：基建止血 & UE 常驻前景工厂

**本周目标**：清理失效脏队列、消除 RTX 4070 Ti 触发 Vulkan `VK_ERROR_DEVICE_LOST` 的隐患，实现单次启动 UE 连续渲染 100 帧透明前景、阴影图和独立 Stencil 掩码零崩溃。

### 核心执行步骤

1. **清场与建库**：将含失效关卡 `/Game/Maps/road_straight` 的 360 条旧 `/data/queue/tasks.jsonl` 重命名归档；初始化 `/data/avsynth/state/jobs.db` 替代原服务进程内存状态。


2. **引擎减负配置**：在 `/data/avsynth/Config/DefaultEngine.ini` 关闭重度全局光照与光追（`r.DynamicGlobalIlluminationMethod=0`、`r.ReflectionMethod=0`、`r.RayTracing=False`、`r.Shadow.Virtual.Enable=0`），并强制开启 `r.CustomDepth=3` 和 `r.PostProcessing.PropagateAlpha=1`。
3. **部署 4 个底座脚本**：按上一轮提供的完整代码落盘 `state_db.py`、`gen_week1_manifest.py`、`ue_batch_worker.py`、`validate_and_postproc.py` 与 `run_week1_batch.sh`（使用 `flock` 锁定 `/tmp/avsynth_gpu.lock`，每 50 帧强制执行一次 UE GC）。
4. **对接现有 Stencil 材质**：查看 `/data/avsynth/Content/Python/preview_obj.py` 与 `/data/avsynth/ObjSmokeTools/run_obj_smoke.sh`，将现有 Stencil PostProcess 材质路径填入 `AVSYNTH_STENCIL_MAT`，确保前景实例从 `101` 开始编号，绝不与地面共用 `1`。



### Week 1 验收门禁

* `bash /data/avsynth/HybridPipeline/run_week1_batch.sh` 单次跑完 100 帧，`validate_and_postproc.py` 返回 `status=SUCCEEDED passed=100/100`。
* `ue_worker.log` 无 `VK_ERROR_DEVICE_LOST` 或 Python `Traceback`；`nvidia-smi` 观察 RTX 4070 Ti 显存稳定在 `3~5 GB` 无持续爬坡。



---

## Week 2：OBJ 3D-to-Real 合成闭环（真实资产 + 像素遮挡 + V4.4 降质）

**本周目标**：替换掉 Cube/Sphere/Cylinder/Cone 几何代理，接入真实 Corner Case 3D 资产，跑通“相机投影布景 → 像素级真实遮挡率 → 真实路采底图融合（含 TSR V4.4 降质）→ Paired Delta Gate 4”全链路。

### 1. 真实 3D 资产注册表（`configs/assets_registry.json`）

导入 UE（`/Game/Assets/OBJ/...`）前在 Blender 统一：**轴心（Pivot）锁在底部中心 `(0,0,0)`、车头朝向 `+X`、尺寸按真实公制校验**。彻底解决旧小样将 `car` 与 `truck` 全部压扁为 `vehicle` 的标签丢失问题：

```json
{
  "assets": {
    "tma_truck_01": {
      "ue_mesh_path": "/Game/Assets/OBJ/SM_Truck_Workzone.SM_Truck_Workzone",
      "fine_class": "truck_construction",
      "coarse_class": "truck",
      "coco_id": 2,
      "size_m": [7.2, 2.5, 3.2],
      "is_real_mesh": true
    },
    "flatbed_truck_01": {
      "ue_mesh_path": "/Game/Assets/OBJ/SM_Truck_Flatbed.SM_Truck_Flatbed",
      "fine_class": "truck_flatbed",
      "coarse_class": "truck",
      "coco_id": 2,
      "size_m": [8.5, 2.55, 2.9],
      "is_real_mesh": true
    },
    "cone_std_01": {
      "ue_mesh_path": "/Game/Assets/OBJ/SM_Cone_Orange.SM_Cone_Orange",
      "fine_class": "cone_standing",
      "coarse_class": "cone",
      "coco_id": 5,
      "size_m": [0.38, 0.38, 0.70],
      "is_real_mesh": true
    },
    "water_barrier_01": {
      "ue_mesh_path": "/Game/Assets/OBJ/SM_WaterBarrier.SM_WaterBarrier",
      "fine_class": "barrier_water",
      "coarse_class": "barrier",
      "coco_id": 6,
      "size_m": [1.5, 0.45, 0.85],
      "is_real_mesh": true
    },
    "ped_worker_01": {
      "ue_mesh_path": "/Game/Assets/OBJ/SM_Ped_Worker.SM_Ped_Worker",
      "fine_class": "pedestrian_worker",
      "coarse_class": "pedestrian",
      "coco_id": 4,
      "size_m": [0.55, 0.45, 1.75],
      "is_real_mesh": true
    }
  }
}

```

### 2. 在 `ue_batch_worker.py` 补上两个 Week 2 改动

1. **真实资产缩放修正**：真实 3D 模型导出时已带真实厘米尺度且轴心在底部中心，在 `spawn_frame_objects()` 中区分真实资产与代理几何：


```python
if spec.get("is_real_mesh", False):
    loc = unreal.Vector(x_cm, y_cm, z_cm)              # 底部中心直接贴 Z=0
    scale = unreal.Vector(1.0, 1.0, 1.0)               # 保持真实物理比例
else:
    loc = unreal.Vector(x_cm, y_cm, z_cm + sz * 50.0)  # 代理几何抬高半高
    scale = unreal.Vector(sx, sy, sz)

```


2. **新增 Pass 4（真实像素级遮挡率）**：在 `Pass 3` 导完合图 Stencil 后，当一帧包含 `>= 2` 个目标时，循环单独显示每个目标并用 `stencil_cam` 快速截取单体完整掩码 `{fid}_full_stencil_{iid}.png`（无光照开销，耗时 `< 5ms`），彻底替代 2D 框重叠近似算法。



### 3. OBJ 3D-to-Real 合成与 V4.4 降质脚本（`scripts/obj_3d_to_real_compositor.py`）

```python
import cv2
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable


def ground_pixel_to_ego_xy(
    u: float, v: float,
    fx: float, fy: float, cx: float, cy: float,
    cam_h: float, pitch_deg: float
) -> Optional[Tuple[float, float]]:
    """将实拍路面像素点 (u, v) 按相机内外参反投为自车 3D 地面坐标 (X_m, Y_m)"""
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    # 相机系(右X, 下Y, 前Z) -> 自车系(前X, 右Y, 上Z)
    p = np.deg2rad(pitch_deg)  # 俯仰角：向下为负
    cos_p, sin_p = np.cos(p), np.sin(p)
    x_dir = ray_cam[2] * cos_p + ray_cam[1] * sin_p
    y_dir = ray_cam[0]
    z_dir = -ray_cam[1] * cos_p + ray_cam[2] * sin_p

    if z_dir >= -1e-4:  # 射线朝向地平线以上，无路面交点
        return None
    t = -cam_h / z_dir
    x_ego, y_ego = float(t * x_dir), float(t * y_dir)
    if not (5.0 <= x_ego <= 80.0):
        return None
    return (round(x_ego, 2), round(y_ego, 2))


def align_3d_fg_photometry_v44(
    bg_img: np.ndarray,
    fg_bgr: np.ndarray,
    fg_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    dist_m: float
) -> np.ndarray:
    """
    复用 TSR V4.4 材质解耦与物理雾霾思路：
    1. 从目标落脚点邻近实拍背景提取局部亮度/色温/空气灰雾底噪 K_bg(P12) 与高光上限 W_bg(P70~P92)
    2. 对低饱和车身/中性面做物理雾霾钳位，对高饱和警示橙/红反光条保纯度不压灰
    """
    h_img, w_img = bg_img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = max(20, int((x2 - x1) * 0.4)), max(20, int((y2 - y1) * 0.4))
    sx1, sy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    sx2, sy2 = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)

    # 仅采目标周围的真实环境像素
    ctx_bg = bg_img[sy1:sy2, sx1:sx2]
    ctx_mask = fg_mask[sy1:sy2, sx1:sx2] == 0
    if ctx_mask.sum() < 50:
        return fg_bgr

    bg_pix = ctx_bg[ctx_mask]
    bg_lab = cv2.cvtColor(bg_pix.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    bg_l = bg_lab[:, 0]

    # V4.4 锚点：K_bg 取 P12，W_bg 取 P70~P92 中位
    p12 = np.percentile(bg_l, 12)
    p70, p92 = np.percentile(bg_l, [70, 92])
    K_bg = np.median(bg_pix[bg_l <= p12], axis=0).astype(np.float32)
    w_sel = (bg_l >= p70) & (bg_l <= p92)
    W_bg = np.median(bg_pix[w_sel], axis=0).astype(np.float32) if w_sel.sum() > 0 else np.percentile(bg_pix, 85, axis=0).astype(np.float32)
    W_bg = np.minimum(W_bg, np.percentile(bg_pix, 95, axis=0).astype(np.float32))
    K_bg = np.clip(np.minimum(K_bg, W_bg - 25.0), 0.0, 220.0)

    roi_fg = fg_bgr[y1:y2, x1:x2].copy()
    roi_m = fg_mask[y1:y2, x1:x2] > 0
    if roi_m.sum() < 16:
        return fg_bgr

    fg_hsv = cv2.cvtColor(roi_fg, cv2.COLOR_BGR2HSV).astype(np.float32)
    # 软材质解耦权重：S<=45 为中性车身/金属/轮胎，S>=100 为警示橙锥桶/反光条
    w_chroma = np.clip((fg_hsv[:, :, 1] - 45.0) / 55.0, 0.0, 1.0)[:, :, None]

    # 1. 中性面物理动态范围与距离雾霾映射
    fg_norm = roi_fg.astype(np.float32) / 255.0
    neutral_mapped = np.clip(K_bg + fg_norm * (W_bg - K_bg), 0, 255)
    # 与原 3D 渲染明暗按 0.55:0.45 混合，保留立体光影层次同时压进实拍动态范围
    neutral_out = 0.55 * neutral_mapped + 0.45 * roi_fg.astype(np.float32)

    # 2. 高饱和警示面：仅按环境白点压亮度上限 + 轻度距离空气透视雾霾，不砸饱和度
    haze_alpha = float(np.clip((dist_m - 12.0) / 140.0, 0.0, 0.28))
    max_v = float(np.max(W_bg))
    chroma_hsv = fg_hsv.copy()
    chroma_hsv[:, :, 2] = np.clip(chroma_hsv[:, :, 2] * min(1.0, max_v / (np.percentile(fg_hsv[:, :, 2][roi_m], 90) + 1e-5)), 0, max_v)
    chroma_bgr = cv2.cvtColor(chroma_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    chroma_out = chroma_bgr * (1.0 - haze_alpha) + W_bg * haze_alpha

    blended = np.clip(chroma_out * w_chroma + neutral_out * (1.0 - w_chroma), 0, 255).astype(np.uint8)
    out_full = fg_bgr.copy()
    out_full[y1:y2, x1:x2] = blended
    return out_full


def composite_obj_frame(
    bg_img: np.ndarray,
    fg_rgb: np.ndarray,
    shadow_matte: np.ndarray,
    inst_stencil: np.ndarray,
    obj_meta_list: List[Dict],
    full_stencil_dir: Optional[Path] = None,
    frame_id: str = "",
    shadow_strength: float = 0.58,
    jpeg_q_range: Tuple[int, int] = (58, 74)
) -> Tuple[np.ndarray, List[Dict]]:
    """
    核心 3D-to-Real 合成函数：
    阴影冷色调压底 -> 逐实例精确掩码与真实像素遮挡率 -> V4.4 局部光度与退化融合 -> JPEG 伪影
    """
    H, W = bg_img.shape[:2]
    fg_resized = cv2.resize(fg_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    sh_resized = cv2.resize(shadow_matte, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    st_resized = cv2.resize(inst_stencil, (W, H), interpolation=cv2.INTER_NEAREST)
    st_gray = st_resized[:, :, 0] if st_resized.ndim == 3 else st_resized

    # Step 1: 真实阴影压底（室外漫反射阴影偏冷蓝：BGR 衰减权重 R > G > B）
    shadow_atten = np.stack([
        1.0 - sh_resized * shadow_strength * 0.85,  # B 通道保留更多蓝天漫反射
        1.0 - sh_resized * shadow_strength * 0.95,  # G
        1.0 - sh_resized * shadow_strength * 1.05,  # R 衰减最明显
    ], axis=-1)
    comp = np.clip(bg_img.astype(np.float32) * shadow_atten, 0, 255).astype(np.uint8)

    # Step 2: 逐实例计算紧致 2D 框、真实像素遮挡率并做局部 V4.4 光度对齐
    fg_aligned = fg_resized[:, :, :3].copy()
    valid_annotations = []
    union_fg_mask = np.zeros((H, W), dtype=np.uint8)

    for obj in obj_meta_list:
        iid = int(obj["instance_id"])
        inst_mask = (st_gray == iid)
        vis_px = int(inst_mask.sum())
        if vis_px < 64:
            continue

        ys, xs = np.where(inst_mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        union_fg_mask[inst_mask] = 255

        # 计算真实像素级遮挡率（非 2D 框重叠近似）[cite: 16]
        occ_ratio = 0.0
        occ_method = "unoccluded_single"
        if full_stencil_dir is not None:
            full_p = full_stencil_dir / f"{frame_id}_full_stencil_{iid}.png"
            if full_p.exists():
                f_st = cv2.imread(str(full_p), cv2.IMREAD_GRAYSCALE)
                if f_st is not None:
                    f_st = cv2.resize(f_st, (W, H), interpolation=cv2.INTER_NEAREST)
                    full_px = int(np.sum(f_st == iid))
                    if full_px > 0:
                        occ_ratio = round(float(np.clip(1.0 - vis_px / float(full_px), 0.0, 1.0)), 4)
                        occ_method = "pixel_stencil_ratio"

        dist_m = float(obj["pos_ego_m"][0])
        fg_aligned = align_3d_fg_photometry_v44(bg_img, fg_aligned, inst_mask.astype(np.uint8) * 255, bbox, dist_m)

        valid_annotations.append({
            "instance_id": iid,
            "fine_class": obj["fine_class"],
            "coarse_class": obj.get("coarse_class", obj["fine_class"]),
            "coco_id": obj.get("coco_id", 1),
            "bbox_xyxy": list(bbox),
            "pose_ego_m": {
                "x": obj["pos_ego_m"][0], "y": obj["pos_ego_m"][1], "z": obj["pos_ego_m"][2],
                "yaw_deg": obj["yaw_deg"], "roll_deg": obj.get("roll_deg", 0.0)
            },
            "visible_pixels": vis_px,
            "occlusion_ratio": occ_ratio,
            "occlusion_method": occ_method,
            "asset_id": obj.get("asset_id", "")
        })

    if not valid_annotations:
        return bg_img, []

    # Step 3: 轻度光学模糊 + 原图 CMOS 高频噪点匹配 + 极窄抗锯齿羽化融合
    ys_u, xs_u = np.where(union_fg_mask > 0)
    ux1, uy1 = max(0, int(xs_u.min()) - 6), max(0, int(ys_u.min()) - 6)
    ux2, uy2 = min(W, int(xs_u.max()) + 6), min(H, int(ys_u.max()) + 6)

    roi_bg = comp[uy1:uy2, ux1:ux2]
    roi_fg = fg_aligned[uy1:uy2, ux1:ux2]
    roi_m = union_fg_mask[uy1:uy2, ux1:ux2]

    # 匹配原图锐度与噪点
    bg_lap = float(np.var(cv2.Laplacian(cv2.cvtColor(roi_bg, cv2.COLOR_BGR2GRAY), cv2.CV_64F)))
    if float(np.var(cv2.Laplacian(cv2.cvtColor(roi_fg, cv2.COLOR_BGR2GRAY), cv2.CV_64F))) > bg_lap * 1.35:
        roi_fg = cv2.GaussianBlur(roi_fg, (3, 3), 0.7)

    bg_only = roi_m == 0
    if bg_only.sum() > 50:
        residual = (roi_bg.astype(np.float32) - cv2.medianBlur(roi_bg, 3).astype(np.float32))[bg_only]
        n_std = float(np.std(residual)) * 0.75
        if n_std > 0.8:
            roi_fg = np.clip(roi_fg.astype(np.float32) + np.random.normal(0, n_std, roi_fg.shape), 0, 255).astype(np.uint8)

    soft_a = (cv2.GaussianBlur(roi_m, (3, 3), 0).astype(np.float32) / 255.0)[:, :, None]
    roi_blend = np.clip(roi_fg.astype(np.float32) * soft_a + roi_bg.astype(np.float32) * (1.0 - soft_a), 0, 255).astype(np.uint8)

    # Step 4: 局部车载 JPEG 压缩一致性（仅在前景及外扩 2px 内生效）
    q = random.randint(*jpeg_q_range)
    ok, buf = cv2.imencode(".jpg", roi_blend, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        comp_jpg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        jm = (cv2.GaussianBlur(cv2.dilate(roi_m, k), (3, 3), 0).astype(np.float32) / 255.0)[:, :, None]
        roi_blend = np.clip(comp_jpg.astype(np.float32) * jm + roi_blend.astype(np.float32) * (1.0 - jm), 0, 255).astype(np.uint8)

    comp[uy1:uy2, ux1:ux2] = roi_blend
    return comp, valid_annotations

```

### Week 2 验收门禁

* 至少导入 **1 辆真实施工/货车 + 2 种障碍物（锥桶/水马）+ 1 个行人**，在 50 帧真实路采底图上完成 3D-to-Real 合成。
* 人工抽检 50 帧叠框图：车轮/锥桶底部紧贴地面且带方向阴影、无浮空或明显“3D 贴片感”；遮挡样本的 `occlusion_ratio` 与肉眼观感吻合；线上 OBJ 检测器 Paired Delta Gate 4 通过率 `≥ 80%`。

---

## Week 3：GND 2.5D IPM 难例引擎 & 施工占道联合联动

**本周目标**：彻底脱离 UE 路面重建与分段 2D 框局限，纯 CPU 跑通实拍道路 2.5D IPM 标线难例引擎（磨损、积水反光、双线涂改、硬阴影），实现与 Week 2 OBJ 掩码的施工占道联动扣减，导出标准折线 JSON 与独立语义掩码（独立 ID `11~16`，根治与路面共用 Stencil `1` 的问题）。

### 完整可执行脚本（`scripts/gnd_ipm_engine.py`）

```python
import cv2
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional

# 独立 GND 语义 ID 表（彻底避开原版与 road 共用 stencil=1 的冲突）[cite: 16]
GND_SEMANTIC_IDS = {
    "lane_solid_white": 11,
    "lane_dashed_white": 12,
    "lane_solid_yellow": 13,
    "lane_double_yellow": 14,
    "stop_line": 15,
    "crosswalk": 16,
    "road_edge_white": 17
}


def build_ground_homography(
    fx: float, fy: float, cx: float, cy: float,
    cam_h: float, pitch_deg: float,
    x_range: Tuple[float, float] = (5.0, 55.0),   # 前向 5m~55m
    y_range: Tuple[float, float] = (-10.0, 10.0), # 左右 ±10m
    res_m_per_px: float = 0.02                    # 2cm/px 精度
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """根据实车相机内外参直接推导 H_img_to_bev 与 H_bev_to_img"""
    bev_w = int(round((y_range[1] - y_range[0]) / res_m_per_px))
    bev_h = int(round((x_range[1] - x_range[0]) / res_m_per_px))

    # 在 Z=0 地面采 4 个公制角点，分别求其 BEV 像素坐标与前视图像像素坐标
    world_pts = [
        (x_range[0], y_range[0]), (x_range[0], y_range[1]),
        (x_range[1], y_range[1]), (x_range[1], y_range[0])
    ]
    p = np.deg2rad(pitch_deg)
    cos_p, sin_p = np.cos(p), np.sin(p)

    img_pts, bev_pts = [], []
    for x_m, y_m in world_pts:
        # 自车系(前X, 右Y, 上Z=0) -> 相机系(右Xc, 下Yc, 前Zc)
        dz = -cam_h  # 地面在相机下方
        xc = y_m
        yc = -dz * cos_p + x_m * sin_p
        zc = x_m * cos_p + dz * sin_p
        u = fx * (xc / zc) + cx
        v = fy * (yc / zc) + cy
        img_pts.append([u, v])

        bu = (y_m - y_range[0]) / res_m_per_px
        bv = (x_range[1] - x_m) / res_m_per_px  # 远处在上，近处在下
        bev_pts.append([bu, bv])

    H_img2bev = cv2.getPerspectiveTransform(np.array(img_pts, np.float32), np.array(bev_pts, np.float32))
    H_bev2img = np.linalg.inv(H_img2bev)
    return H_img2bev, H_bev2img, (bev_w, bev_h)


def _multi_scale_noise(h: int, w: int, scales: Tuple[int, ...] = (8, 24, 64)) -> np.ndarray:
    """零依赖多尺度平滑噪声，替代外部 Perlin 库"""
    acc = np.zeros((h, w), dtype=np.float32)
    weights = [0.55, 0.30, 0.15]
    for s, wt in zip(scales, weights):
        gh, gw = max(2, h // s), max(2, w // s)
        grid = np.random.rand(gh, gw).astype(np.float32)
        acc += wt * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip((acc - acc.min()) / (acc.max() - acc.min() + 1e-5), 0.0, 1.0)


def _project_pts(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([pts[:, :2].astype(np.float32), np.ones((len(pts), 1), np.float32)], axis=1)
    out = (H @ pts_h.T).T
    return out[:, :2] / (out[:, 2:3] + 1e-6)


class GNDIPMEngine:
    def __init__(self, res_m_per_px: float = 0.02):
        self.res = res_m_per_px

    def _draw_lane_mask_bev(self, bev_shape: Tuple[int, int], bev_lines: List[Dict], width_m: float = 0.16) -> np.ndarray:
        h, w = bev_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        thick = max(3, int(round(width_m / self.res)))
        for ln in bev_lines:
            pts = np.round(ln["pts_bev"]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 255, thickness=thick, lineType=cv2.LINE_AA)
        return mask

    def op_wear_and_fading(self, bev_img: np.ndarray, lane_mask: np.ndarray, severity: float = 0.55) -> Tuple[np.ndarray, np.ndarray]:
        """算子 A：轮胎碾压磨损/剥落（用标线旁 22cm 真实沥青纹理平移回填，零涂抹伪影）"""
        h, w = bev_img.shape[:2]
        shift_px = max(8, int(round(0.22 / self.res)))
        asphalt_left = np.roll(bev_img, shift_px, axis=1)
        asphalt_right = np.roll(bev_img, -shift_px, axis=1)
        asphalt_fill = ((asphalt_left.astype(np.float32) + asphalt_right.astype(np.float32)) * 0.5).astype(np.uint8)

        # 生成沿行车方向拉长的条状磨损噪声
        noise = _multi_scale_noise(h // 2, w, scales=(6, 18, 48))
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        wear_alpha = np.clip((noise - (1.0 - severity)) / (severity + 1e-5), 0.0, 1.0)
        wear_alpha *= (lane_mask.astype(np.float32) / 255.0)

        a3 = wear_alpha[:, :, None]
        out = np.clip(asphalt_fill.astype(np.float32) * a3 + bev_img.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)
        edit_mask = (wear_alpha > 0.05).astype(np.uint8) * 255
        return out, edit_mask

    def op_wet_puddle_glare(self, bev_img: np.ndarray, sky_color_bgr: np.ndarray, intensity: float = 0.55) -> Tuple[np.ndarray, np.ndarray]:
        """算子 B：雨后积水与逆光镜面反光（拉高局部高光 + 天空倒影压低标线对比度）"""
        h, w = bev_img.shape[:2]
        puddle = _multi_scale_noise(h, w, scales=(32, 80, 160))
        puddle_mask = np.clip((puddle - 0.58) / 0.25, 0.0, 1.0)
        puddle_mask = cv2.GaussianBlur(puddle_mask, (15, 15), 0)

        # 水洼区垂直模糊模拟水面倒影，再混合天空高亮反光
        reflected = cv2.GaussianBlur(bev_img, (3, 19), 0).astype(np.float32)
        sky_layer = np.full_like(reflected, sky_color_bgr.reshape(1, 1, 3), dtype=np.float32)
        glare = np.clip(reflected * (1.0 - intensity) + sky_layer * intensity + 18.0, 0, 255)

        m3 = puddle_mask[:, :, None]
        out = np.clip(glare * m3 + bev_img.astype(np.float32) * (1.0 - m3), 0, 255).astype(np.uint8)
        edit_mask = (puddle_mask > 0.05).astype(np.uint8) * 255
        return out, edit_mask

    def op_ghost_line_or_tar(self, bev_img: np.ndarray, bev_lines: List[Dict], mode: str = "ghost") -> Tuple[np.ndarray, np.ndarray]:
        """算子 C：新旧双线幽灵线（横向偏移 20~40cm 灰白旧线）或黑柏油反光涂改带"""
        h, w = bev_img.shape[:2]
        out = bev_img.copy()
        edit_mask = np.zeros((h, w), dtype=np.uint8)
        if not bev_lines:
            return out, edit_mask

        target = random.choice(bev_lines)
        offset_px = int(round(random.uniform(0.22, 0.42) / self.res)) * random.choice([-1, 1])
        pts = np.round(target["pts_bev"]).astype(np.int32).copy()
        pts[:, 0] = np.clip(pts[:, 0] + offset_px, 0, w - 1)
        poly = pts.reshape(-1, 1, 2)

        thick = max(4, int(round(0.15 / self.res)))
        overlay = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(overlay, [poly], False, 255, thickness=thick, lineType=cv2.LINE_AA)
        noise = _multi_scale_noise(h, w, scales=(8, 20, 48))

        if mode == "ghost":
            # 低对比度斑驳灰白旧线
            alpha = (overlay.astype(np.float32) / 255.0) * np.clip(noise * 0.55, 0.1, 0.48)
            color = np.array([175.0, 178.0, 180.0], dtype=np.float32)
        else:
            # 深黑色沥青涂改带（略带反光）
            alpha = (overlay.astype(np.float32) / 255.0) * np.clip(0.65 + noise * 0.25, 0.5, 0.88)
            color = np.array([38.0, 38.0, 40.0], dtype=np.float32)

        a3 = alpha[:, :, None]
        out = np.clip(color * a3 + out.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)
        edit_mask[overlay > 0] = 255
        return out, edit_mask

    def op_cast_shadow(self, bev_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """算子 D：护栏/树木斜切硬阴影（室外偏冷色调压暗）"""
        h, w = bev_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        # 随机画 2~4 条斜向阴影带
        for _ in range(random.randint(2, 4)):
            y0 = random.randint(0, h - 1)
            thick = random.randint(int(0.6 / self.res), int(2.5 / self.res))
            slope = random.uniform(-0.45, 0.45)
            p1 = (0, y0)
            p2 = (w - 1, int(y0 + slope * w))
            cv2.line(mask, p1, p2, 1.0, thickness=thick)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        # 冷色调阴影衰减系数 (B, G, R)
        atten = np.array([0.52, 0.46, 0.42], dtype=np.float32).reshape(1, 1, 3)
        m3 = mask[:, :, None]
        darkened = bev_img.astype(np.float32) * atten
        out = np.clip(darkened * m3 + bev_img.astype(np.float32) * (1.0 - m3), 0, 255).astype(np.uint8)
        return out, (mask > 0.05).astype(np.uint8) * 255

    def generate(
        self,
        bg_img: np.ndarray,
        gt_lanes: List[Dict],
        camera_cfg: Dict,
        active_ops: Tuple[str, ...] = ("wear", "shadow"),
        obj_instance_mask: Optional[np.ndarray] = None  # 传入 Week 2 的 OBJ 掩码实现施工占道联动
    ) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        返回: (编辑后前视图, 标准折线真值 JSON 列表, 独立语义掩码 PNG)
        gt_lanes 每项格式: {"line_id": 1, "line_type": "lane_dashed_white", "polyline_img": [[x,y], ...]}
        """
        H_img, W_img = bg_img.shape[:2]
        H_i2b, H_b2i, (bw, bh) = build_ground_homography(
            camera_cfg["fx"], camera_cfg["fy"], camera_cfg["cx"], camera_cfg["cy"],
            camera_cfg["height_m"], camera_cfg["pitch_deg"], res_m_per_px=self.res
        )

        # 1. 投影实拍图与真值折线到 BEV 平面
        bev_img = cv2.warpPerspective(bg_img, H_i2b, (bw, bh), flags=cv2.INTER_LINEAR)
        bev_lines = []
        for ln in gt_lanes:
            pts = np.array(ln["polyline_img"], dtype=np.float32)
            if len(pts) < 2:
                continue
            bev_lines.append({**ln, "pts_bev": _project_pts(pts, H_i2b)})

        lane_mask_bev = self._draw_lane_mask_bev((bh, bw), bev_lines)
        total_edit_bev = np.zeros((bh, bw), dtype=np.uint8)
        applied_tags = []

        # 2. 按需组合执行难例算子
        sky_bgr = np.median(bg_img[:max(20, H_img // 5), :, :].reshape(-1, 3), axis=0).astype(np.float32)

        if "wear" in active_ops:
            sev = round(random.uniform(0.40, 0.72), 2)
            bev_img, em = self.op_wear_and_fading(bev_img, lane_mask_bev, severity=sev)
            total_edit_bev = cv2.bitwise_or(total_edit_bev, em)
            applied_tags.append(f"wear_{int(sev*100)}pct")
        if "puddle" in active_ops:
            bev_img, em = self.op_wet_puddle_glare(bev_img, sky_bgr, intensity=random.uniform(0.4, 0.65))
            total_edit_bev = cv2.bitwise_or(total_edit_bev, em)
            applied_tags.append("wet_puddle_glare")
        if "ghost" in active_ops or "tar" in active_ops:
            m = "ghost" if "ghost" in active_ops else "tar"
            bev_img, em = self.op_ghost_line_or_tar(bev_img, bev_lines, mode=m)
            total_edit_bev = cv2.bitwise_or(total_edit_bev, em)
            applied_tags.append(f"{m}_line")
        if "shadow" in active_ops:
            bev_img, em = self.op_cast_shadow(bev_img)
            total_edit_bev = cv2.bitwise_or(total_edit_bev, em)
            applied_tags.append("cast_shadow")

        # 3. 反投回原前视图，仅在编辑区域羽化写回，100% 保留原图天空/车辆/建筑
        warped_img = cv2.warpPerspective(bev_img, H_b2i, (W_img, H_img), flags=cv2.INTER_LINEAR)
        warped_edit = cv2.warpPerspective(total_edit_bev, H_b2i, (W_img, H_img), flags=cv2.INTER_NEAREST)
        soft_m = (cv2.GaussianBlur(warped_edit, (5, 5), 0).astype(np.float32) / 255.0)[:, :, None]
        out_img = np.clip(warped_img.astype(np.float32) * soft_m + bg_img.astype(np.float32) * (1.0 - soft_m), 0, 255).astype(np.uint8)

        # 4. 生成标准折线真值 + 独立语义掩码，并与 Week 2 OBJ 实例掩码做占道遮挡扣减
        sem_mask = np.zeros((H_img, W_img), dtype=np.uint8)
        out_lanes = []
        for ln in bev_lines:
            sem_id = GND_SEMANTIC_IDS.get(ln["line_type"], 11)
            pts_img = np.array(ln["polyline_img"], dtype=np.float32)
            # 画语义掩码（从 BEV 恒定线宽反投回图像，透视宽度天然精确）
            one_bev = self._draw_lane_mask_bev((bh, bw), [ln], width_m=0.15)
            one_img = cv2.warpPerspective(one_bev, H_b2i, (W_img, H_img), flags=cv2.INTER_NEAREST)
            sem_mask[one_img > 127] = sem_id

            # 逐点标记可见性 (1=可见, 0=被施工卡车/锥桶遮挡)
            poly_vis = []
            for x, y in pts_img[:, :2]:
                xi, yi = int(round(x)), int(round(y))
                vis = 1
                if obj_instance_mask is not None and 0 <= xi < W_img and 0 <= yi < H_img:
                    if obj_instance_mask[yi, xi] > 0:
                        vis = 0
                poly_vis.append([round(float(x), 1), round(float(y), 1), vis])

            out_lanes.append({
                "line_id": ln["line_id"],
                "line_type": ln["line_type"],
                "semantic_id": sem_id,
                "degradation_applied": applied_tags,
                "polyline_img": poly_vis,
                "is_ghost_line": False
            })

        # 被 OBJ 前景盖住的路面语义掩码置 0（不可见）
        if obj_instance_mask is not None:
            sem_mask[obj_instance_mask > 0] = 0

        return out_img, out_lanes, sem_mask

```

### Week 3 验收门禁

* 产出 50 帧涵盖 `wear` / `puddle` / `ghost` / `shadow` 及 `workzone 占道联合` 的标线样本。
* 折线反投回前视图与真实标线边缘对齐误差 `≤ 2px`；语义掩码像素值严格为 `11~17`（与背景 `0` 完全分离，彻底解决 Stencil `1` 冲突）；被卡车/锥桶遮挡的折线点 `vis` 正确翻为 `0`。



---

## Week 4：三段解耦批产、混训验真 & 接入 `8010` 工作台

**本周目标**：按 TSR 已验证的“GPU/CPU 三段解耦”架构批量产出 3,000~5,000 帧 Corner Case，导出版本化数据集完成一次真实混训 mAP 验收；确认有效后再把参数表单与 SQLite 状态接到 `[http://10.2.248.34:8010/](http://10.2.248.34:8010/)`。

### 1. 三段解耦大批量调度器（`scripts/batch_orchestrator.py`）

严格遵循 **单卡 RTX 4070 Ti 同一时刻只跑一个 GPU 进程** 的铁律：

* **Stage 1（纯 GPU，UE 分块渲染）**：把 3,000 帧大清单按 `chunk_size = 300` 切块，每块单次启动 UE 渲染完成后彻底退出释放显存，失败块自动重试最多 2 次。
* **Stage 2（纯 CPU 多进程）**：开启 `ProcessPoolExecutor(max_workers=8)` 并发执行 `obj_3d_to_real_compositor.py` 与 `gnd_ipm_engine.py`，几分钟即可合成数千帧。
* **Stage 3（纯 GPU Batch）**：独占 `/tmp/avsynth_gpu.lock` 加载线上检测/车道线模型，批量跑 Paired Delta Gate 4 筛选并写回 `jobs.db`。

```python
import os
import json
import math
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from state_db import JobStateDB

UE_BIN = "/data/UnrealEngine/Engine/Binaries/Linux/UnrealEditor-Cmd"
UPROJECT = "/data/avsynth/avsynth.uproject"
MAP_PATH = "/Game/Maps/codex_validation"
LOCK_FILE = "/tmp/avsynth_gpu.lock"


def run_ue_chunk(chunk_manifest: Path, out_dir: Path, log_path: Path, max_retries: int = 2) -> bool:
    """Stage 1: 分块运行 UE 前景工厂，每块结束自动退出彻底释放 4070 Ti 显存[cite: 16]"""
    env = os.environ.copy()
    env["AVSYNTH_MANIFEST"] = str(chunk_manifest)
    env["AVSYNTH_OUT_DIR"] = str(out_dir)

    cmd = [
        "flock", "-w", "1200", LOCK_FILE,
        "xvfb-run", "-a", "-s", "-screen 0 1280x720x24",
        UE_BIN, UPROJECT, MAP_PATH,
        "-ExecutePythonScript=/data/avsynth/HybridPipeline/scripts/ue_batch_worker.py",
        "-NullRHI=0", "-RenderOffscreen", "-unattended", "-nop4", "-nosplash", "-nosound",
        f"-abslog={log_path}"
    ]
    for attempt in range(1, max_retries + 1):
        subprocess.run(cmd, env=env, check=False)
        log_txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
        if "VK_ERROR_DEVICE_LOST" not in log_txt and "批渲染完成" in log_txt:
            return True
        print(f"[WARN] UE chunk {chunk_manifest.name} 第 {attempt} 次尝试异常，准备重试")
    return False


def orchestrate_batch(job_id: str, full_manifest: Path, work_dir: Path, chunk_size: int = 300):
    db = JobStateDB()
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = work_dir / "ue_raw"
    raw_dir.mkdir(exist_ok=True)

    lines = [l for l in full_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    db.create_job(job_id, "hybrid_prod", str(full_manifest), str(work_dir), str(work_dir / "ue.log"), len(lines), {"chunk_size": chunk_size})

    # Stage 1: 切块执行 UE 前景渲染
    num_chunks = math.ceil(len(lines) / chunk_size)
    for c in range(num_chunks):
        c_manifest = work_dir / f"chunk_{c:03d}.jsonl"
        c_manifest.write_text("\n".join(lines[c * chunk_size:(c + 1) * chunk_size]) + "\n", encoding="utf-8")
        ok = run_ue_chunk(c_manifest, raw_dir, work_dir / f"ue_chunk_{c:03d}.log")
        if not ok:
            print(f"[ERROR] chunk_{c:03d} 多次重试失败，记录断点继续后续块")

    # Stage 2: 此处调用 ProcessPoolExecutor 并发跑 OBJ 3D-to-Real 合成 + GND IPM 编辑
    # Stage 3: 获取 GPU 锁执行 Batch Gate 4 筛选，更新 jobs.db 并触发 export_dataset.py
    print(f"[DONE] 批次 {job_id} Stage 1 完成，共切分 {num_chunks} 块，准备进入 CPU 合成与 Gate 4")

```

### 2. 版本化数据集导出器（`scripts/export_dataset.py`）

将通过 Gate 4 的样本按训练侧可直接消费的格式打包，彻底分离 OBJ COCO 与 GND 折线/掩码，并在 `dataset_manifest.json` 显式写明图像存储位深与解码类型，避免旧版 EXR/PNG 位深歧义：

```text
/data/datasets/avsynth_v20261215/
├── dataset_manifest.json       # 版本号、资产清单、相机标定、位深/色彩空间、遮挡算法口径
├── images/                     # 最终合成 RGB (8-bit sRGB PNG/JPG)
├── obj_coco.json               # 标准 COCO（保留 fine_class 与 coarse_class，含 occlusion_ratio）
├── gnd_polylines.json          # 标准车道线折线（每点带 [x, y, vis] 与退化标签）
└── masks/
    ├── obj_instance/           # OBJ 实例掩码 (uint8/uint16 PNG)
    └── gnd_semantic/           # GND 独立语义掩码 (uint8 PNG, ID 11~17)

```

### 3. 小批混训验真方案

* **配比建议**：首批产出 **3,000 帧 OBJ（施工占道、倒地锥桶、异形卡车、高遮挡行人）+ 2,000 帧 GND（磨损、积水反光、双线涂改、占道遮挡）**，按真实训练集的 **10%~15%** 比例混入（稀有长尾类按 1:1 补齐）。
* **验收指标**：在**纯真实路采 Corner Case 测试集**上对比基线模型与混训模型——要求目标长尾类的 Recall/mAP 净提升（OBJ 目标 `+3~5%`、GND 难例 F1 `+2~4%`），且常规清晰路采测试集指标不回退（浮动在 `±0.3%` 以内）。

### 4. 接入 `8010` 工作台（仅在混训验证有效后开发）

保留现有 `[http://10.2.248.34:8010/](http://10.2.248.34:8010/)` 顶部 OBJ/GND 标签，做 3 个最小侵入式改造：

1. **参数真正下发**：将前端种类、数量、距离、朝向、磨损/反光/阴影开关直接序列化为 JSONL 清单（替代现有 3 个固定预设脚本），传入 `batch_orchestrator.py`。


2. **状态从 SQLite 读取**：后端任务列表、单帧进度、错误日志与抽检图全部直接查 `/data/avsynth/state/jobs.db`，服务重启后历史批次和断点续跑状态不丢失。


3. **支持“单帧即时预览 → 5 帧核对 → 批量提交”三段按钮**：复用同一个 `ue_batch_worker.py` 与 `gnd_ipm_engine.py`，彻底避免预览脚本和批量脚本出现两套逻辑。



---

如果你马上开工 Week 1，建议先在 `10.2.248.34` 上看一眼现有 `/data/avsynth/ObjSmokeTools/run_obj_smoke.sh` 里的 UE 启动参数和 Stencil 材质路径——有需要可以直接贴出来，我帮你把 `ue_batch_worker.py` 的环境对接细节一次对死。