# -*- coding: utf-8 -*-
"""
下载完整的 LIS 数据集（训练集 1561 对 + 测试集 669 对），整理成和 LIS_sample 相同的结构：

    data\\LIS\\{train,test}\\{normal,dark}\\<正常光图编号>.<后缀>
    data\\LIS\\{train,test}\\labels.json

官方来源：https://github.com/Linwei-Chen/LIS -> Google Drive 文件夹 1KpC82G_H1CI35lmnB2LYr9aK3FQcahAC
RGB-normal.zip 4.1 GB，RGB-dark.zip 0.3 GB，annotations.zip 29 MB。下载支持断点续传，重跑会接着下。
暗图的文件名编号 = 正常光图编号 + 1。labels.json 里的类别名保持官方 json 的原样（顺序是错的，
由 run_pipeline_lis.py 的 NAME_FIX 改正）。

    python scripts\\fetch_lis_full.py
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_lis_sample import GDRIVE, IDS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "LIS"
LOG_FILE = PROJECT_ROOT / "logs" / "fetch_lis_full.log"
ZIP_NAMES = {"annotations": "annotations.zip", "normal": "RGB-normal.zip", "dark": "RGB-dark.zip"}
SPLITS = ("train", "test")


def download(url, dst, log):
    """断点续传下载；已下完的直接跳过。"""
    part = dst.with_suffix(dst.suffix + ".part")
    if dst.exists():
        log(f"{dst.name}: 已存在（{dst.stat().st_size / 2 ** 20:.0f} MB），跳过")
        return
    for attempt in range(50):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if "text/html" in r.headers.get("Content-Type", ""):
                    raise IOError("Google Drive 返回的是网页而不是文件（可能是下载配额用完了）")
                if have and r.status != 206:
                    have = 0   # 服务器不支持续传，从头来
                total = have + int(r.headers.get("Content-Length", 0))
                t0, last = time.time(), time.time()
                with open(part, "ab" if have else "wb") as f:
                    got = have
                    while True:
                        buf = r.read(4 * 2 ** 20)
                        if not buf:
                            break
                        f.write(buf)
                        got += len(buf)
                        if time.time() - last > 30:
                            last = time.time()
                            log(f"{dst.name}: {got / 2 ** 20:.0f} / {total / 2 ** 20:.0f} MB  "
                                f"{(got - have) / 2 ** 20 / (time.time() - t0):.1f} MB/s")
            if got >= total:
                part.rename(dst)
                log(f"{dst.name}: 下载完成，{got / 2 ** 20:.0f} MB")
                return
        except IOError as e:
            log(f"{dst.name}: 第 {attempt + 1} 次中断（{e}），10 秒后续传")
            time.sleep(10)
    raise IOError(f"{dst.name} 下载失败")


def main():
    ap = argparse.ArgumentParser()
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
    for key in ("annotations", "dark", "normal"):
        download(GDRIVE.format(IDS[key]), out / ZIP_NAMES[key], log)

    az = zipfile.ZipFile(out / ZIP_NAMES["annotations"])
    zips = {p: zipfile.ZipFile(out / ZIP_NAMES[p]) for p in ("normal", "dark")}
    by_stem = {p: {Path(i.filename).stem: i for i in z.infolist()
                   if i.filename.lower().endswith((".jpg", ".png")) and "__MACOSX" not in i.filename}
               for p, z in zips.items()}
    for split in SPLITS:
        coco = json.loads(az.read(f"annotations/lis_coco_JPG_{split}.json").decode("utf-8"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        images = sorted(coco["images"], key=lambda im: im["id"])
        boxes = {im["id"]: [] for im in images}
        for a in coco["annotations"]:
            if a["image_id"] in boxes and not a.get("iscrowd"):
                x, y, w, h = a["bbox"]
                boxes[a["image_id"]].append({"category": cats[a["category_id"]],
                                             "x1": x, "y1": y, "x2": x + w, "y2": y + h})
        for part in ("normal", "dark"):
            (out / split / part).mkdir(parents=True, exist_ok=True)
            for im in images:
                info = by_stem[part][str(im["id"] + (part == "dark"))]
                dst = out / split / part / f"{im['id']}{Path(info.filename).suffix}"
                if not dst.exists() or dst.stat().st_size != info.file_size:
                    dst.write_bytes(zips[part].read(info))
        with open(out / split / "labels.json", "w", encoding="utf-8") as f:
            json.dump([{"id": im["id"], "width": im["width"], "height": im["height"], "boxes": boxes[im["id"]]}
                       for im in images], f)
        log(f"{split}: {len(images)} 对，{sum(len(b) for b in boxes.values())} 个框，保存到 {out / split}")
    log("完成（图像按正常光图的编号命名）")


if __name__ == "__main__":
    main()
