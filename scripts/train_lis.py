# -*- coding: utf-8 -*-
"""
在 LIS 训练集（暗图 / 正常光图配对）上微调 HVI-CIDNet。

损失 = 官方的图像损失（和 code\\HVI-CIDNet\\train.py 相同）+ yolo_weight × YOLO 特征损失。
YOLO 特征损失：把增强图和正常光图分别送进冻结的 YOLO11s，让送入检测头的三层特征（P3/P4/P5）接近。
YOLO 的参数始终不更新，只把梯度传回 CIDNet；评测时用的还是原来那个 yolo11s.pt。

两组实验：
    对照组（只用图像损失）   python scripts\\train_lis.py --name image_only --yolo_weight 0
    加 YOLO 特征损失         python scripts\\train_lis.py --name yolo_feat  --yolo_weight 1

输入：data\\LIS\\train\\{normal,dark}\\（由 fetch_lis_full.py 取得）
输出：weights\\lis_ft\\<name>\\epoch_XX.pth 和 last.ckpt（断点续训用），日志在 logs\\
中断后加 --resume 接着训。PyCharm 里解释器选 E:\\Anaconda\\envs\\cidnet\\python.exe。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code" / "HVI-CIDNet"

# ============================ 默认设置 ============================
DATA_DIR = PROJECT_ROOT / "data" / "LIS" / "train"
OUT_ROOT = PROJECT_ROOT / "weights" / "lis_ft"
INIT_WEIGHT = PROJECT_ROOT / "weights" / "LOLv2_real" / "w_perc.pth"   # 现成权重里在 LIS 上检测最好的
DETECTOR = PROJECT_ROOT / "weights" / "yolo11s.pt"
EPOCHS = 50
BATCH = 2            # 6 GB 显存、不开混合精度时实测能用的最大值
CROP = 256           # 官方训练的裁剪尺寸；不缩放原图，物体尺度和评测时（imgsz 1280）基本一致
LR = 1e-4            # 官方学习率
LR_MIN = 1e-6
YOLO_WEIGHT = 1.0
WORKERS = 4
SAVE_EVERY = 5
SEED = 0
# 官方 train.py 的损失权重
HVI_WEIGHT, L1_WEIGHT, D_WEIGHT, E_WEIGHT, P_WEIGHT = 1.0, 1.0, 0.5, 50.0, 1e-2
# ==================================================================


def imread_rgb(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"读不了图片：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class LisPairs(Dataset):
    """暗图和正常光图在同一位置随机裁剪，加随机翻转。"""

    def __init__(self, root, crop):
        root = Path(root)
        ids = [e["id"] for e in json.load(open(root / "labels.json", encoding="utf-8"))]
        self.pairs = [(next((root / "dark").glob(f"{i}.*")), next((root / "normal").glob(f"{i}.*"))) for i in ids]
        self.crop = crop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        dark, normal = (imread_rgb(p) for p in self.pairs[k])
        h, w = dark.shape[:2]
        y0, x0 = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
        dark = dark[y0:y0 + self.crop, x0:x0 + self.crop]
        normal = normal[y0:y0 + self.crop, x0:x0 + self.crop]
        if random.random() < 0.5:
            dark, normal = dark[:, ::-1], normal[:, ::-1]
        if random.random() < 0.5:
            dark, normal = dark[::-1], normal[::-1]
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255.0)
        return to_t(dark), to_t(normal)


class YoloFeatures:
    """冻结的 YOLO，取送入检测头的几层特征（不跑检测头）。输入是 0–1 的 RGB，边长要是 32 的倍数。"""

    def __init__(self, weight, device):
        from ultralytics import YOLO
        self.net = YOLO(str(weight)).model.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.taps = list(self.net.model[-1].f)   # 检测头的输入层编号，YOLO11 是 [16, 19, 22]

    def __call__(self, x):
        # 和 ultralytics 的 BaseModel._predict_once 相同的逐层前向，到最后一个取特征的层为止
        y, outs = [], []
        for m in self.net.model[:max(self.taps) + 1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.net.save else None)
            if m.i in self.taps:
                outs.append(x)
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="这次训练的名字，决定权重和日志的保存位置")
    ap.add_argument("--yolo_weight", type=float, default=YOLO_WEIGHT, help="YOLO 特征损失的权重；0 = 对照组")
    ap.add_argument("--p_weight", type=float, default=P_WEIGHT, help="感知损失权重；0 = 不用（不加载 VGG19）")
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--init", default=str(INIT_WEIGHT), help="起点权重")
    ap.add_argument("--detector", default=str(DETECTOR))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true", help="从 last.ckpt 接着训")
    ap.add_argument("--max_iters", type=int, default=0, help="每个 epoch 最多跑多少步；只用于试跑")
    args = ap.parse_args()

    out = OUT_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "logs"
    log_path = log_dir / f"train_lis_{args.name}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    os.chdir(CODE_DIR)   # 官方感知损失按相对路径找 VGG 权重
    sys.path.insert(0, str(CODE_DIR))
    from net.CIDNet import CIDNet
    from loss.losses import L1Loss, SSIM, EdgeLoss, PerceptualLoss

    model = CIDNet().to(device)
    model.load_state_dict(torch.load(args.init, map_location="cpu"))
    L1 = L1Loss(loss_weight=L1_WEIGHT, reduction="mean").to(device)
    D = SSIM(weight=D_WEIGHT).to(device)
    E = EdgeLoss(loss_weight=E_WEIGHT).to(device)
    P = None
    if args.p_weight > 0:
        P = PerceptualLoss({"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},
                           perceptual_weight=1.0, criterion="mse").to(device)
    yolo = YoloFeatures(args.detector, device) if args.yolo_weight > 0 else None

    def image_loss(a, b):
        loss = L1(a, b) + D(a, b) + E(a, b)
        return loss + args.p_weight * P(a, b)[0] if P is not None else loss

    loader = DataLoader(LisPairs(args.data, args.crop), batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=LR_MIN)
    start = 0
    if args.resume and (out / "last.ckpt").exists():
        ck = torch.load(out / "last.ckpt", map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start = ck["epoch"]

    log(f"==== {args.name}  {len(loader.dataset)} 对  epochs={args.epochs} batch={args.batch} crop={args.crop} "
        f"lr={args.lr} yolo_weight={args.yolo_weight} p_weight={args.p_weight} seed={args.seed}")
    log(f"起点权重 {args.init}" + (f"  从第 {start} 个 epoch 续训" if start else ""))
    if yolo:
        log(f"YOLO 特征层 {yolo.taps}（冻结，{args.detector}）")

    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        t0, sums, n = time.time(), {"image": 0.0, "yolo": 0.0}, 0
        for it, (dark, normal) in enumerate(loader):
            if args.max_iters and it >= args.max_iters:
                break
            dark, normal = dark.to(device, non_blocking=True), normal.to(device, non_blocking=True)
            out_rgb = model(dark)
            out_hvi, gt_hvi = model.HVIT(out_rgb), model.HVIT(normal)
            l_img = image_loss(out_rgb, normal) + HVI_WEIGHT * image_loss(out_hvi, gt_hvi)
            l_yolo = torch.zeros((), device=device)
            if yolo:
                with torch.no_grad():
                    ref = yolo(normal)
                cur = yolo(out_rgb.clamp(0, 1))
                l_yolo = sum(F.l1_loss(a, b) for a, b in zip(cur, ref)) / len(ref)
            loss = l_img + args.yolo_weight * l_yolo
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"损失变成 {loss.item()}（epoch {epoch} 第 {it} 步），停止训练")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums["image"] += l_img.item()
            sums["yolo"] += l_yolo.item()
            n += 1
        scheduler.step()
        log(f"epoch {epoch:3d}/{args.epochs}  图像损失 {sums['image'] / n:.4f}  YOLO 特征损失 {sums['yolo'] / n:.4f}  "
            f"lr {optimizer.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f} s")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch}, out / "last.ckpt")
        if epoch % SAVE_EVERY == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), out / f"epoch_{epoch:03d}.pth")
    log(f"完成，权重在 {out}")


if __name__ == "__main__":
    main()
