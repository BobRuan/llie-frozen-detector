# -*- coding: utf-8 -*-
"""
从 BDD100K 官方镜像 http://dl.yf.io/bdd100k/ 只取一小部分验证集图像，不下载整个 5.3 GB 的包。

做法和 fetch_darkdriving_sample.py 一样：用 HTTP Range 只读 zip 的目录和需要的文件。
按标注里的 timeofday 属性分成白天（daytime）和夜间（night），各随机取 N 张（固定随机种子），
只要求图里至少有一个 car 标注。BDD100K 的白天图和夜间图不是同一场景，没有配对关系。

    python scripts\\fetch_bdd100k_sample.py --num 200
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import random
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_darkdriving_sample import HttpRangeFile  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "BDD100K_sample"
LOG_FILE = PROJECT_ROOT / "logs" / "fetch_bdd100k_sample.log"
BASE = "http://dl.yf.io/bdd100k/"
SPLITS = {"day": "daytime", "night": "night"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=200, help="白天、夜间各取多少张")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    LOG_FILE.parent.mkdir(exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')}  来源：{BASE}")

    # ---------- 标注 ----------
    # 镜像上的 bdd100k_det_20_labels.zip 里其实不是检测标注；用 bdd100k_labels.zip（每张图一个 json）
    lf = HttpRangeFile(BASE + "bdd100k_labels.zip", chunk=8 * 2 ** 20)
    lz = zipfile.ZipFile(lf)
    val_infos = sorted((i for i in lz.infolist() if "/val/" in i.filename and i.filename.endswith(".json")),
                       key=lambda i: i.header_offset)
    frames = []
    for i in val_infos:
        d = json.loads(lz.read(i))
        objs = d["frames"][0]["objects"] if "frames" in d else d.get("labels") or []
        boxes = [{"category": o["category"], **{k: o["box2d"][k] for k in ("x1", "y1", "x2", "y2")}}
                 for o in objs if "box2d" in o]
        frames.append({"name": Path(i.filename).stem + ".jpg", "attributes": d.get("attributes", {}),
                       "boxes": boxes})
    log(f"验证集标注 {len(frames)} 帧，下载 {lf.bytes_fetched / 2 ** 20:.1f} MB")

    rng = random.Random(args.seed)
    picked = {}
    for split, tod in SPLITS.items():
        pool = [f for f in frames if f["attributes"].get("timeofday") == tod
                and any(b["category"] == "car" for b in f["boxes"])]
        pool.sort(key=lambda f: f["name"])
        picked[split] = rng.sample(pool, min(args.num, len(pool)))
        log(f"{split}（timeofday={tod}）：验证集里有 car 标注的共 {len(pool)} 张，取 {len(picked[split])} 张")

    # ---------- 图像 ----------
    imf = HttpRangeFile(BASE + "bdd100k_images_100k.zip", chunk=8 * 2 ** 20)
    iz = zipfile.ZipFile(imf)
    by_name = {Path(i.filename).name: i for i in iz.infolist() if "/val/" in i.filename}
    imf.chunk = 160 * 1024   # 目录读完后，单张图约 60 KB，预读调小免得多下
    for split, fs in picked.items():
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        for f in sorted(fs, key=lambda f: by_name[f["name"]].header_offset):
            (out / split / "images" / f["name"]).write_bytes(iz.read(by_name[f["name"]]))
        with open(out / split / "labels.json", "w", encoding="utf-8") as fp:
            json.dump(sorted(fs, key=lambda f: f["name"]), fp)
        log(f"{split}: 已保存 {len(fs)} 张")
    log(f"完成：图像包实际下载 {imf.bytes_fetched / 2 ** 20:.1f} MB，保存到 {out}")


if __name__ == "__main__":
    main()
