# -*- coding: utf-8 -*-
"""
生成 低光 | 增强 | GT 的并排对比图。

默认用 reproduce_lolv2_real.py 产出的 best_GT_mean 结果。PyCharm 里右键 Run 即可。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "LOLv2" / "Real_captured" / "Test"

# ============================ 默认设置 ============================
LOW_DIR = TEST_DIR / "Low"
GT_DIR = TEST_DIR / "Normal"
ENH_DIR = PROJECT_ROOT / "outputs" / "lolv2_real" / "best_GT_mean"
OUT_DIR = PROJECT_ROOT / "outputs" / "comparison"
NUM_IMAGES = 6     # 在测试集里等间隔挑几张
# ==================================================================


def imread(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"读不了图片：{path}")
    return img


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255.0 ** 2 / (mse + 1e-8))


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def gt_name(low_name):
    """LOL-v2-Real 里 Low 叫 lowXXXXX.png，GT 叫 normalXXXXX.png。"""
    return low_name.replace("low", "normal", 1) if low_name.startswith("low") else low_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", default=str(LOW_DIR))
    ap.add_argument("--enh", default=str(ENH_DIR))
    ap.add_argument("--gt", default=str(GT_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--num", type=int, default=NUM_IMAGES)
    args = ap.parse_args()

    low_dir, enh_dir, gt_dir, out_dir = map(Path, (args.low, args.enh, args.gt, args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in low_dir.glob("*.png"))
    picks = [names[i] for i in np.linspace(0, len(names) - 1, args.num).round().astype(int)]

    rows = []
    for name in picks:
        low, enh = imread(low_dir / name), imread(enh_dir / name)
        gt_path = gt_dir / name
        if not gt_path.exists():
            gt_path = gt_dir / gt_name(name)
        gt = imread(gt_path)
        row = np.hstack([
            label(low, f"Low  {name}"),
            label(enh, f"HVI-CIDNet  PSNR {psnr(enh, gt):.2f} dB"),
            label(gt, "Ground truth"),
        ])
        ok, buf = cv2.imencode(".jpg", row, [cv2.IMWRITE_JPEG_QUALITY, 92])
        buf.tofile(str(out_dir / (Path(name).stem + "_compare.jpg")))
        rows.append(row)
        print(f"{name}: PSNR {psnr(enh, gt):.2f} dB")

    grid = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    buf.tofile(str(out_dir / "comparison_grid.jpg"))
    print(f"已保存到 {out_dir}")


if __name__ == "__main__":
    main()
