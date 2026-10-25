# -*- coding: utf-8 -*-
"""
把 data\\LIS\\{train,test}\\labels.json 转成 MMDetection 能直接读的 COCO 格式，给对比方法（YOLA 等）训练和评测用。

    data\\LIS\\{train,test}\\coco_dark.json     图像指向暗图 dark\\<编号>.JPG

类别名已经按 run_pipeline_lis.py 的 NAME_FIX 改正（官方 json 的类别名顺序是错的）。
类别顺序固定为下面的 LIS_CLASSES。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "LIS"
LIS_CLASSES = ("bicycle", "car", "motorbike", "bus", "bottle", "chair", "diningtable", "tvmonitor")
# 和 run_pipeline_lis.py 相同
NAME_FIX = {"bicycle": "bicycle", "chair": "car", "diningtable": "motorbike", "bottle": "bus",
            "motorbike": "bottle", "car": "chair", "tvmonitor": "diningtable", "bus": "tvmonitor"}


def main():
    for split in ("train", "test"):
        root = DATA_DIR / split
        entries = json.load(open(root / "labels.json", encoding="utf-8"))
        images, anns = [], []
        for e in entries:
            img = next((root / "dark").glob(f"{e['id']}.*"))
            images.append({"id": e["id"], "file_name": f"dark/{img.name}", "width": e["width"], "height": e["height"]})
            for b in e["boxes"]:
                w, h = b["x2"] - b["x1"], b["y2"] - b["y1"]
                anns.append({"id": len(anns) + 1, "image_id": e["id"],
                             "category_id": LIS_CLASSES.index(NAME_FIX[b["category"]]) + 1,
                             "bbox": [b["x1"], b["y1"], w, h], "area": w * h, "iscrowd": 0})
        coco = {"images": images, "annotations": anns,
                "categories": [{"id": k + 1, "name": c} for k, c in enumerate(LIS_CLASSES)]}
        with open(root / "coco_dark.json", "w", encoding="utf-8") as f:
            json.dump(coco, f)
        print(f"{split}: {len(images)} 张，{len(anns)} 个框 -> {root / 'coco_dark.json'}")


if __name__ == "__main__":
    main()
