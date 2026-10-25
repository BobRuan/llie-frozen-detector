# -*- coding: utf-8 -*-
"""
自检：LIS 标注里的类别名是否和图像内容对得上。
在正常光图上跑 COCO 检测器，统计每个 LIS 标注类别的框，最常和哪个 COCO 类别的检测框重合（IoU >= 0.5）。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_folder import imread_rgb  # noqa: E402
from run_pipeline_darkdriving import iou_one_to_many  # noqa: E402

DATA = PROJECT_ROOT / "data" / "LIS_sample"


def main():
    from ultralytics import YOLO
    det = YOLO(str(PROJECT_ROOT / "weights" / "yolo11s.pt"))
    entries = json.load(open(DATA / "labels.json", encoding="utf-8"))
    table = collections.defaultdict(collections.Counter)
    for e in entries:
        img = cv2.cvtColor(imread_rgb(next((DATA / "normal").glob(f"{e['id']}.*"))), cv2.COLOR_RGB2BGR)
        r = det.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
        boxes, cls = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        for b in e["boxes"]:
            gt = np.array([b["x1"], b["y1"], b["x2"], b["y2"]], dtype=np.float32)
            hit = "(没检到)"
            if len(boxes):
                ious = iou_one_to_many(gt, boxes)
                if ious.max() >= 0.5:
                    hit = det.names[int(cls[ious.argmax()])]
            table[b["category"]][hit] += 1
    for cat, c in table.items():
        print(f"LIS 标注为 {cat:<12} -> 检测器认为：{c.most_common(4)}")


if __name__ == "__main__":
    main()
