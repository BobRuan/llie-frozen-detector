# -*- coding: utf-8 -*-
"""
从 LIS 数据集（Instance Segmentation in the Dark, IJCV 2023）的官方 Google Drive 里取一部分测试集配对图。

LIS：同一场景、三脚架固定拍摄的短曝光（暗）/长曝光（正常光）图像对，带 8 类物体的标注（COCO json）。
暗图的文件名编号 = 正常光图编号 + 1。
官方来源：https://github.com/Linwei-Chen/LIS -> Google Drive 文件夹 1KpC82G_H1CI35lmnB2LYr9aK3FQcahAC
用 HTTP Range 只取需要的图，不下载整个包（RGB-normal.zip 4.1 GB，RGB-dark.zip 0.3 GB）。

    python scripts\\fetch_lis_sample.py --num 150
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_darkdriving_sample import HttpRangeFile  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "LIS_sample"
LOG_FILE = PROJECT_ROOT / "logs" / "fetch_lis_sample.log"
GDRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
IDS = {
    "annotations": "1JSlFFUniFe9CG3WPSw8eI4Pk2_U4nTvz",
    "normal": "1DlY15BXjHUGLCS9sqKE0sdN8BQi7Huaw",
    "dark": "1tbfncHwnXE9Xs3ZWNSS3ZPZB82bsfWA-",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=150, help="取多少对；0 = 测试集全部")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')}  来源：LIS 官方 Google Drive")
    ann_zip = out / "annotations.zip"
    if not ann_zip.exists():
        urllib.request.urlretrieve(GDRIVE.format(IDS["annotations"]), str(ann_zip))
    az = zipfile.ZipFile(ann_zip)
    coco = json.loads(az.read("annotations/lis_coco_JPG_test.json").decode("utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = sorted(coco["images"], key=lambda im: im["id"])
    log(f"测试集 {len(images)} 对，类别 {list(cats.values())}")
    if args.num and args.num < len(images):
        images = sorted(random.Random(args.seed).sample(images, args.num), key=lambda im: im["id"])

    boxes = {im["id"]: [] for im in images}
    for a in coco["annotations"]:
        if a["image_id"] in boxes and not a.get("iscrowd"):
            x, y, w, h = a["bbox"]
            boxes[a["image_id"]].append({"category": cats[a["category_id"]],
                                         "x1": x, "y1": y, "x2": x + w, "y2": y + h})

    index = []
    for part in ("normal", "dark"):
        fh = HttpRangeFile(GDRIVE.format(IDS[part]), chunk=8 * 2 ** 20)
        zf = zipfile.ZipFile(fh)
        by_stem = {Path(i.filename).stem: i for i in zf.infolist()
                   if i.filename.lower().endswith((".jpg", ".png")) and "__MACOSX" not in i.filename}
        fh.chunk = 3 * 2 ** 20
        (out / part).mkdir(exist_ok=True)
        for im in sorted(images, key=lambda im: by_stem[str(im["id"] + (part == "dark"))].header_offset):
            info = by_stem[str(im["id"] + (part == "dark"))]
            (out / part / f"{im['id']}{Path(info.filename).suffix}").write_bytes(zf.read(info))
        log(f"{part}: 已保存 {len(images)} 张，实际下载 {fh.bytes_fetched / 2 ** 20:.1f} MB")

    for im in images:
        index.append({"id": im["id"], "width": im["width"], "height": im["height"], "boxes": boxes[im["id"]]})
    with open(out / "labels.json", "w", encoding="utf-8") as f:
        json.dump(index, f)
    log(f"完成：{len(images)} 对，保存到 {out}（图像按正常光图的编号命名）")


if __name__ == "__main__":
    main()
