# -*- coding: utf-8 -*-
"""
LIS 测试集的逐场景对比图：每个场景一张大图，7 组输入按原尺寸（1200×800）排成 2 行，
画上标注框（绿）和检测框（红，置信度 >= 0.25）。

输入：run_pipeline_lis.py 存在 outputs\\lis_test\\ 下的增强结果，和 data\\LIS\\test\\
输出：outputs\\lis_test\\gallery\\<场景编号>.jpg

    python scripts\\make_lis_gallery.py                 # 和 overview.jpg 相同的 8 个场景
    python scripts\\make_lis_gallery.py --ids 1802 1810  # 指定场景
    python scripts\\make_lis_gallery.py --num 30        # 等间隔取 30 个场景
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline_lis import LIS_TO_COCO, NAME_FIX, CONF_SHOW, draw, imread_rgb, save_jpg  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "LIS" / "test"
ENH_DIR = PROJECT_ROOT / "outputs" / "lis_test"
DETECTOR = PROJECT_ROOT / "weights" / "yolo11s.pt"
IMGSZ = 1280
GROUPS = ["normal", "dark", "auto_gamma", "cidnet_generalization",
          "cidnet_lolv2real", "ft_image_only", "ft_yolo_feat"]
PER_ROW = 4


def bar(text, width):
    b = np.zeros((48, width, 3), np.uint8)
    cv2.putText(b, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*", default=[])
    ap.add_argument("--num", type=int, default=8, help="不给 --ids 时，等间隔取多少个场景")
    args = ap.parse_args()

    entries = {e["id"]: e for e in json.load(open(DATA_DIR / "labels.json", encoding="utf-8"))}
    ids = sorted(entries)
    picks = args.ids or [ids[k] for k in np.linspace(0, len(ids) - 1, args.num).round().astype(int)]
    path_of = {"normal": lambda i: next((DATA_DIR / "normal").glob(f"{i}.*")),
               "dark": lambda i: next((DATA_DIR / "dark").glob(f"{i}.*"))}
    for g in GROUPS[2:]:
        path_of[g] = (lambda g: lambda i: ENH_DIR / g / f"{i}.png")(g)

    from ultralytics import YOLO
    det = YOLO(str(DETECTOR))
    out = ENH_DIR / "gallery"
    out.mkdir(exist_ok=True)
    for i in picks:
        gt = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in entries[i]["boxes"]]
        names = sorted({NAME_FIX[b["category"]] for b in entries[i]["boxes"]})
        cells = []
        for g in GROUPS:
            rgb = imread_rgb(path_of[g](i))
            r = det.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=IMGSZ, conf=CONF_SHOW,
                            classes=sorted(LIS_TO_COCO.values()), verbose=False)[0]
            boxes, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            cells.append(np.vstack([bar(f"{g}   det {len(boxes)} / gt {len(gt)}", rgb.shape[1]),
                                    draw(rgb, gt, (boxes, conf))]))
        while len(cells) % PER_ROW:
            legend = np.zeros_like(cells[0])
            for k, t in enumerate([f"scene {i}", "objects: " + ", ".join(names),
                                   "green = ground truth", f"red = YOLO11s detection (conf >= {CONF_SHOW})"]):
                cv2.putText(legend, t, (30, 120 + 70 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(legend)
        rows = [np.hstack(cells[k:k + PER_ROW]) for k in range(0, len(cells), PER_ROW)]
        save_jpg(out / f"{i}.jpg", np.vstack(rows), 90)
        print(f"{out / f'{i}.jpg'}", flush=True)


if __name__ == "__main__":
    main()
