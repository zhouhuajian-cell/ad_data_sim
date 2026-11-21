#!/usr/bin/env python3
"""提高 P/R 的检测训练（针对性优化）。

诊断结论：
  1) 跨场景泛化差（见过 90%+，没见过 0-26%）→ 已扩到 82 clip
  2) 大目标（近处）漏检严重（>128px 只有 35.9%）→ 提高 imgsz
  3) rider/bus 样本少 → 类别加权

优化：
  - imgsz 1024 → 1280（大目标友好）
  - 数据增强加强（mosaic/mixup/scale）
  - 训练更久（100 epoch）+ 更耐心早停（patience 30）
  - 从上一版权重继续（迁移学习）
"""
import argparse
import os
import shutil
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/train/lane_obj/data.yaml")
    ap.add_argument("--model", default="/opt/ad_mining/lane_obj_v1.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--project", default="/data/train/runs")
    ap.add_argument("--name", default="obj_v2_pr")
    a = ap.parse_args()

    from ultralytics import YOLO

    # 起点：有旧权重就从它继续
    start = a.model if os.path.isfile(a.model) else \
        "/opt/ad_mining/yolov8x.pt"
    print("=== PR 优化训练 ===")
    print("  数据:  ", a.data)
    print("  起点:  ", start)
    print("  epochs:", a.epochs, " imgsz:", a.imgsz, " batch:", a.batch)

    model = YOLO(start)
    t0 = time.time()

    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        project=a.project,
        name=a.name,
        exist_ok=True,
        patience=30,
        workers=4,
        device=0,
        amp=True,
        cache=False,
        # 学习率（重启）
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        # ★ 提升召回的数据增强
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.5,
        degrees=3.0,
        translate=0.15,
        scale=0.6,        # 大范围缩放，覆盖大小目标
        shear=2.0,
        perspective=0.0005,
        flipud=0.2,       # 允许垂直翻转（安全场景多变）
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,   # 实例增强，提升小目标
        close_mosaic=15,  # 最后 15 轮关掉 mosaic
        # 验证
        val=True,
        plots=True,
        save=True,
        save_period=25,
    )

    dt = (time.time() - t0) / 60
    print("\n=== 训练完成 %.1f 分钟 ===" % dt)

    best = os.path.join(a.project, a.name, "weights", "best.pt")
    if os.path.isfile(best):
        dst = "/opt/ad_mining/lane_obj_v2.pt"
        shutil.copy2(best, dst)
        print("最佳:", best, "→", dst)

        # 详细验证
        m = YOLO(dst)
        r = m.val(data=a.data, imgsz=a.imgsz, batch=a.batch, verbose=False,
                  plots=True)
        print("\n=== 最终指标 ===")
        print("  mAP50:    %.4f" % r.box.map50)
        print("  mAP50-95: %.4f" % r.box.map)
        try:
            mp, mr = r.box.mp, r.box.mr
            print("  Precision: %.4f" % mp)
            print("  Recall:    %.4f" % mr)
        except Exception:
            pass
        names = m.names
        for i, c in enumerate(names.values() if isinstance(names, dict)
                              else names):
            if i < len(r.box.maps):
                print("  %-12s mAP50-95=%.3f" % (c, r.box.maps[i]))
    else:
        print("!! 无 best.pt")


if __name__ == "__main__":
    main()
