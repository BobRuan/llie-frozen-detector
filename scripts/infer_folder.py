# -*- coding: utf-8 -*-
"""
用 HVI-CIDNet 增强一个文件夹里的所有图片。

用法一（PyCharm）：改下面“默认设置”里的 INPUT_DIR / OUTPUT_DIR，右键 Run。
用法二（命令行）：
    E:\\Anaconda\\envs\\cidnet\\python.exe scripts\\infer_folder.py ^
        --input D:\\some\\night_images --output D:\\some\\enhanced

解释器请选 E:\\Anaconda\\envs\\cidnet\\python.exe。路径请用绝对路径。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code" / "HVI-CIDNet"

# ============================ 默认设置 ============================
INPUT_DIR = PROJECT_ROOT / "data" / "LOLv2" / "Real_captured" / "Test" / "Low"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "infer_demo"
# LOL 测试图用 LOLv2_real\w_perc.pth；真实夜间/驾驶图建议换成 generalization.pth
WEIGHT = PROJECT_ROOT / "weights" / "LOLv2_real" / "w_perc.pth"
ALPHA = 1.0        # 输出亮度缩放（官方 eval.py 的 --alpha），1.0 = 不缩放
GAMMA = 1.0        # 输入 gamma 曲线（官方 eval.py 的 --gamma），<1 更亮
TILE = 0           # 分块大小；0 = 自动（小图整图推理，大图自动分块）；-1 = 强制整图
TILE_OVERLAP = 64  # 分块重叠像素，用来消除拼接缝
# 自动模式下，超过这个像素数就分块。实测整图推理 1280x720 约占 1.3 GB 显存，按像素数线性增长
MAX_FULL_PIXELS = 1920 * 1088
AUTO_TILE = 768
# ==================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
FACTOR = 8  # 网络下采样 3 次，输入边长要是 8 的倍数


def imread_rgb(path):
    """np.fromfile + imdecode：路径里有中文也能读。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"读不了图片：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"编码失败：{path}")
    buf.tofile(str(path))


def load_model(weight_path, alpha, device):
    sys.path.insert(0, str(CODE_DIR))
    from net.CIDNet import CIDNet
    model = CIDNet().to(device)
    model.load_state_dict(torch.load(str(weight_path), map_location="cpu"))
    model.eval()
    # 和官方 eval.py 的 v2 / unpaired 分支一致
    model.trans.gated2 = True
    model.trans.alpha = alpha
    return model


def forward_padded(model, x, gamma):
    """把输入反射填充到 8 的倍数，推理后裁回原尺寸。"""
    _, _, h, w = x.shape
    ph, pw = (-h) % FACTOR, (-w) % FACTOR
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    y = model(x ** gamma)
    return torch.clamp(y, 0, 1)[:, :, :h, :w]


def tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def ramp(n, overlap, device):
    """分块的一维融合权重：边缘线性过渡，中间为 1。"""
    i = torch.arange(n, device=device, dtype=torch.float32)
    return torch.clamp(torch.minimum(i + 1, n - i) / max(overlap, 1), max=1.0)


def forward_tiled(model, x, gamma, tile, overlap):
    _, _, h, w = x.shape
    th, tw = min(tile, h), min(tile, w)
    out = torch.zeros_like(x)
    weight = torch.zeros(1, 1, h, w, device=x.device)
    mask = (ramp(th, overlap, x.device)[:, None] * ramp(tw, overlap, x.device)[None, :])[None, None]
    for y0 in tile_starts(h, th, tile - overlap):
        for x0 in tile_starts(w, tw, tile - overlap):
            patch = forward_padded(model, x[:, :, y0:y0 + th, x0:x0 + tw], gamma)
            out[:, :, y0:y0 + th, x0:x0 + tw] += patch * mask
            weight[:, :, y0:y0 + th, x0:x0 + tw] += mask
    return out / weight


@torch.no_grad()
def enhance(model, rgb, gamma, tile, overlap, device):
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    h, w = rgb.shape[:2]
    if tile == 0 and h * w > MAX_FULL_PIXELS:
        tile = AUTO_TILE
    if tile > 0 and (h > tile or w > tile):
        y, mode = forward_tiled(model, x, gamma, tile, overlap), f"tile{tile}"
    else:
        y, mode = forward_padded(model, x, gamma), "full"
    out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return out, mode


def main():
    ap = argparse.ArgumentParser(description="HVI-CIDNet 文件夹推理")
    ap.add_argument("--input", type=str, default=str(INPUT_DIR), help="输入图片文件夹")
    ap.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="输出文件夹")
    ap.add_argument("--weight", type=str, default=str(WEIGHT), help="权重 .pth")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--tile", type=int, default=TILE, help="分块大小，0=自动，-1=强制整图")
    ap.add_argument("--tile_overlap", type=int, default=TILE_OVERLAP)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"文件夹里没有图片：{in_dir}")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"infer_{time.strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}  weight={args.weight}  alpha={args.alpha}  gamma={args.gamma}")
    log(f"input={in_dir}  ({len(files)} 张)\noutput={out_dir}")
    model = load_model(args.weight, args.alpha, device)

    t_all = time.time()
    for i, p in enumerate(files, 1):
        t0 = time.time()
        rgb = imread_rgb(p)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        out, mode = enhance(model, rgb, args.gamma, args.tile, args.tile_overlap, device)
        imwrite_rgb(out_dir / (p.stem + ".png"), out)
        mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0
        log(f"[{i}/{len(files)}] {p.name}  {rgb.shape[1]}x{rgb.shape[0]}  {mode}  "
            f"{time.time() - t0:.2f}s  peak {mem:.0f} MiB")
    log(f"完成：{len(files)} 张，用时 {time.time() - t_all:.1f}s，日志 {log_path}")


if __name__ == "__main__":
    main()
