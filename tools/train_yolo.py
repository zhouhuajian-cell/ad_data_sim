#!/usr/bin/env python3
"""用 done_data 真值训练 YOLOv8 检测模型。

起点：如果已存在 yolov8x.pt，从它微调；否则从头训。
数据：/data/train/lane_obj/data.yaml
输出：/data/train/runs/obj_det/weights/best.pt
"""
import argparse
import os
import shutil
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/train/lane_obj/data.yaml")
    ap.add_argument("--model", default="/opt/ad_mining/yolov8x.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="/data/train/runs")
    ap.add_argument("--name", default="obj_det")
    a = ap.parse_args()

    from ultralytics import YOLO

    print("=== 训练配置 ===")
    print("  数据:  ", a.data)
    print("  起点:  ", a.model)
    print("  epochs:", a.epochs)
    print("  imgsz: ", a.imgsz)
    print("  batch: ", a.batch)

    t0 = time.time()
    model = YOLO(a.model)

    results = model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        project=a.project,
        name=a.name,
        exist_ok=True,
        patience=20,
        workers=4,
        device=0,
        amp=True,
        # 数据增强（夜间/雨天场景需要）
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # 验证
        val=True,
        plots=True,
        save=True,
        save_period=10,
    )

    dt = time.time() - t0
    print("\n=== 训练完成 %.1f 分钟 ===" % (dt / 60))

    best = os.path.join(a.project, a.name, "weights", "best.pt")
    if os.path.isfile(best):
        dst = "/opt/ad_mining/lane_obj_v1.pt"
        shutil.copy2(best, dst)
        print("最佳模型:", best)
        print("已复制到:", dst)

        # 验证
        m = YOLO(dst)
        r = m.val(data=a.data, imgsz=a.imgsz, batch=a.batch, verbose=False)
        print("\n=== 验证指标 ===")
        try:
            print("  mAP50:    %.4f" % r.box.map50)
            print("  mAP50-95: %.4f" % r.box.map)
            for i, c in enumerate(m.names.values() if isinstance(m.names, dict)
                                  else m.names):
                if i < len(r.box.maps):
                    print("  %-12s mAP50-95=%.3f" % (c, r.box.maps[i]))
        except Exception as e:
            print("  指标读取失败:", e)
    else:
        print("!! 未找到 best.pt")


if __name__ == "__main__":
    main()
