#!/usr/bin/env python3
"""GND 渲染会话：净面只读缓存 + 无状态重绘。

严格满足三条关键逻辑：
  ① 同一帧只在首次加载时跑一次擦除，clean_bg 存入缓存并 setflags(write=False)
  ② 每次换素材/改参数，必须从 clean_bg.copy() 全新拷贝画布，严禁在上一轮输出上叠加
  ③ 最后一步永远用原始 raw_img 的 vehicle_mask 回贴车辆前景
"""
import hashlib
import os
import pickle
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0, "/opt/ue5_synth/tools")


DISK_CACHE_DIR = "/data/render/_gnd_clean_cache"


def _disk_path(frame_id):
    import hashlib as _h
    k = _h.md5(frame_id.encode("utf-8")).hexdigest()[:16]
    return os.path.join(DISK_CACHE_DIR, k + ".npz")


def _disk_load(frame_id):
    p = _disk_path(frame_id)
    if not os.path.isfile(p):
        return None
    try:
        z = __import__("numpy").load(p)
        clean = z["clean"]
        clean.setflags(write=False)
        return {"raw": z["raw"], "clean": clean,
                "vehicle_mask": z["vm"], "drivable": z["drv"]}
    except Exception:
        return None


def _disk_save(frame_id, rec):
    try:
        os.makedirs(DISK_CACHE_DIR, exist_ok=True)
        import numpy as _np
        _np.savez_compressed(_disk_path(frame_id),
                             raw=rec["raw"], clean=rec["clean"],
                             vm=rec["vehicle_mask"], drv=rec["drivable"])
    except Exception:
        pass


class GNDRenderSession:
    def __init__(self, max_frames=200):
        # frame_id -> {"raw": img, "clean": clean_bg, "vehicle_mask": v_mask}
        self._cache = {}
        self._lock = threading.Lock()
        self._max = max_frames

    def prepare_frame(self, frame_id, raw_bgr, drivable_mask, vehicle_mask,
                      erase_fn):
        """
        阶段 1：切帧时只执行一次，生成绝对干净的只读净面底图。
        erase_fn(raw, drivable, vehicle) -> clean_bg
        """
        with self._lock:
            if frame_id in self._cache:
                return self._cache[frame_id]["clean"]

        # DISK-CACHE: 先查磁盘（跨进程共享）
        _dr = _disk_load(frame_id)
        if _dr is not None:
            with self._lock:
                self._cache[frame_id] = _dr
            return _dr["clean"]

        clean_bg = erase_fn(raw_bgr, drivable_mask, vehicle_mask)
        # 设为只读，防止后续任何函数意外原地修改 (in-place mutation)
        clean_bg.setflags(write=False)
        rec = {
            "raw": raw_bgr.copy(),
            "clean": clean_bg,
            "vehicle_mask": vehicle_mask.copy(),
            "drivable": drivable_mask.copy(),
        }
        with self._lock:
            if len(self._cache) >= self._max:
                try:
                    self._cache.pop(next(iter(self._cache)))
                except Exception:
                    pass
            self._cache[frame_id] = rec
        # 存磁盘供其他进程复用
        _disk_save(frame_id, rec)
        return clean_bg

    def render_asset(self, frame_id, asset_config, render_fn, blend_fn):
        """
        阶段 2：每次更换素材/调参数时调用，完全无状态。
        render_fn(shape, asset_config) -> (new_gnd_bgr, new_gnd_alpha)
        blend_fn(clean_copy, new_gnd_bgr, new_gnd_alpha) -> merged
        """
        rec = self._cache.get(frame_id)
        if rec is None:
            return None

        # 强制深拷贝一份全新的净面底图，彻底杜绝上一次素材残留
        canvas = rec["clean"].copy()

        # 仅渲染当前选中的新素材
        new_gnd_bgr, new_gnd_alpha = render_fn(canvas.shape, asset_config)
        if new_gnd_alpha is None or not np.any(new_gnd_alpha > 0.005):
            # 无素材 → 直接回原图的车辆前景叠加
            merged = canvas
        else:
            merged = blend_fn(canvas, new_gnd_bgr, new_gnd_alpha)

        # 回贴真实车辆前景，确保新素材不覆盖车身/轮胎
        v_mask = rec["vehicle_mask"] > 127
        final_img = np.where(v_mask[..., None], rec["raw"], merged)
        return final_img.astype(np.uint8)

    # ---------- 查询 ----------

    def has(self, frame_id):
        return frame_id in self._cache

    def get_clean(self, frame_id):
        """取净面底图（只读）。"""
        rec = self._cache.get(frame_id)
        return None if rec is None else rec["clean"]

    def get_raw(self, frame_id):
        rec = self._cache.get(frame_id)
        return None if rec is None else rec["raw"]

    def get_vehicle_mask(self, frame_id):
        rec = self._cache.get(frame_id)
        return None if rec is None else rec["vehicle_mask"]

    def clear(self):
        with self._lock:
            self._cache.clear()

    def stats(self):
        return {"cached": len(self._cache),
                "ids": list(self._cache.keys())[:20]}


# 全局单例
_SESSION = GNDRenderSession()


def get_session():
    return _SESSION
