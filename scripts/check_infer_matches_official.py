# -*- coding: utf-8 -*-
"""
自检：infer_folder.py 的输出是否和官方 eval.py 一致，以及分块推理和整图推理差多少。
先跑 reproduce_lolv2_real.py，再跑本脚本。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
INFER = str(ROOT / "scripts" / "infer_folder.py")


def read(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR).astype(np.int16)


def infer(*args):
    subprocess.run([sys.executable, INFER, *map(str, args)], check=True, stdout=subprocess.DEVNULL)


def main():
    # 1) 和官方 eval.py --lol_v2_real --best_GT_mean（alpha=0.84）逐像素比较
    mine, official = OUT / "_check_alpha084", OUT / "lolv2_real" / "best_GT_mean"
    infer("--alpha", 0.84, "--output", mine)
    diffs = [np.abs(read(mine / f.name) - read(f)).max() for f in sorted(official.glob("*.png"))]
    print(f"infer_folder 对比官方 eval.py：{len(diffs)} 张图，最大像素差 = {max(diffs)}")

    # 2) 把一张测试图放大到驾驶图常见尺寸，比较整图和分块
    src = ROOT / "data" / "LOLv2" / "Real_captured" / "Test" / "Low" / "00690.png"
    img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    in_dir = OUT / "_size_test_in"
    in_dir.mkdir(parents=True, exist_ok=True)
    for name, (w, h) in {"a_1280x720": (1280, 720), "b_1920x1080": (1920, 1080),
                         "c_2448x2048": (2448, 2048)}.items():
        cv2.imencode(".png", cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC))[1].tofile(
            str(in_dir / f"{name}.png"))
    full, tiled = OUT / "_size_test_full", OUT / "_size_test_tile768"
    infer("--input", in_dir, "--output", full, "--tile", -1)
    infer("--input", in_dir, "--output", tiled, "--tile", 768)
    for f in sorted(full.glob("*.png")):
        e = np.abs(read(f) - read(tiled / f.name))
        print(f"{f.stem} 整图 vs 分块768：平均像素差 {e.mean():.3f}，最大 {e.max()}")


if __name__ == "__main__":
    main()
