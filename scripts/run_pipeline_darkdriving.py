# -*- coding: utf-8 -*-
"""
完整管线（小样本试跑）：夜间图 -> HVI-CIDNet 增强 -> 冻结的 YOLO 检测 -> 和白天、夜间原图比较。

输入：data\\DarkDriving_sample\\{day,night}\\{images,labels}（由 fetch_darkdriving_sample.py 取得）
输出：outputs\\darkdriving_sample\\ 下的增强图、对比图，logs\\ 下的日志和结果 json

PyCharm 里右键 Run 即可；解释器选 E:\\Anaconda\\envs\\cidnet\\python.exe。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_folder import enhance, imread_rgb, imwrite_rgb, load_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================ 默认设置 ============================
DATA_DIR = PROJECT_ROOT / "data" / "DarkDriving_sample"
OUT_DIR = PROJECT_ROOT / "outputs" / "darkdriving_sample"
LLIE_WEIGHT = PROJECT_ROOT / "weights" / "generalization.pth"
DETECTOR = PROJECT_ROOT / "weights" / "yolo11s.pt"   # COCO 预训练，不训练不微调；没有会自动下载
IMGSZ = 1280          # 检测器输入长边
COCO_CAR = 2          # COCO 的 car 类；DarkDriving 只标了 Car（类别 0）
CONF_SHOW = 0.25      # 画图和统计 TP/FP/FN 用的置信度阈值
IOU_THR = 0.5
# 增强的三种做法（原图 2448x2048，6 GB 显存整图放不下）：
#   tile768   原分辨率分块推理
#   resize1280 先缩到长边 1280 整图增强，再放大回原尺寸
#   full      原分辨率整图推理。6 GB 显存放不下，实测会卡在共享内存上几分钟一张，默认不跑
ENH_VARIANTS = ["tile768", "resize1280"]
# ==================================================================


def read_labels(path, w, h):
    """YOLO txt -> [[x1,y1,x2,y2], ...]（像素坐标）。"""
    boxes = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        p = line.split()
        if len(p) >= 5:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def iou_one_to_many(box, boxes):
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def evaluate(dets, gts):
    """dets: {名字: (boxes Nx4, conf N)}，gts: {名字: boxes Mx4}。返回 AP50 和阈值 CONF_SHOW 下的 TP/FP/FN。"""
    n_gt = sum(len(g) for g in gts.values())
    records = sorted(((c, name, b) for name, (bs, cs) in dets.items() for b, c in zip(bs, cs)),
                     key=lambda r: -r[0])
    used = {name: np.zeros(len(g), dtype=bool) for name, g in gts.items()}
    tps = []
    for conf, name, box in records:
        hit = False
        if len(gts[name]):
            ious = iou_one_to_many(box, gts[name])
            ious[used[name]] = 0
            j = int(ious.argmax())
            if ious[j] >= IOU_THR:
                used[name][j], hit = True, True
        tps.append(hit)
    tps = np.array(tps, dtype=bool)
    confs = np.array([r[0] for r in records])
    tp_c, fp_c = np.cumsum(tps), np.cumsum(~tps)
    recall = tp_c / max(n_gt, 1)
    precision = tp_c / np.maximum(tp_c + fp_c, 1)
    # 全点插值 AP
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[1], precision, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    keep = confs >= CONF_SHOW
    tp, fp = int(tps[keep].sum()), int((~tps[keep]).sum())
    return {"AP50": ap, "TP": tp, "FP": fp, "FN": n_gt - tp, "n_gt": n_gt}


def enhance_variant(model, rgb, variant, device):
    h, w = rgb.shape[:2]
    if variant == "tile768":
        return enhance(model, rgb, 1.0, 768, 64, device)[0]
    if variant == "full":
        return enhance(model, rgb, 1.0, -1, 0, device)[0]
    if variant == "resize1280":
        s = 1280 / max(h, w)
        small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        out = enhance(model, small, 1.0, -1, 0, device)[0]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    raise ValueError(variant)


def draw(rgb, gt, det_boxes, det_conf, thick):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for b in gt:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), thick)
    for b, c in zip(det_boxes, det_conf):
        if c >= CONF_SHOW:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), thick)
    return img


def titled(img, text, width):
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    bar = np.zeros((28, width, 3), np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def crop_around(img, gt, size=320):
    """以面积最大的 GT 框为中心裁一块，方便看清很小的车。"""
    h, w = img.shape[:2]
    b = gt[((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).argmax()]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    side = int(min(max(size, (b[2] - b[0]) * 1.5, (b[3] - b[1]) * 1.5), h, w))
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return img[y0:y0 + side, x0:x0 + side]


def save_jpg(path, img, q=90):
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tofile(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--llie_weight", default=str(LLIE_WEIGHT))
    ap.add_argument("--detector", default=str(DETECTOR))
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--variants", nargs="*", default=ENH_VARIANTS)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "logs"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_darkdriving_{stamp}.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = sorted(p.stem for p in (data / "night" / "images").glob("*.jpg"))
    log(f"样本 {len(names)} 对  device={device}")
    log(f"增强权重 {args.llie_weight}\n检测器 {args.detector}  imgsz={args.imgsz}  只看 COCO car 类")

    # ---------- 1. 增强 ----------
    llie = load_model(args.llie_weight, 1.0, device)
    images = {"day": {}, "night": {}}
    for n in names:
        images["day"][n] = imread_rgb(data / "day" / "images" / f"{n}.jpg")
        images["night"][n] = imread_rgb(data / "night" / "images" / f"{n}.jpg")
    for v in args.variants:
        key = f"enh_{v}"
        images[key] = {}
        (out / key).mkdir(exist_ok=True)
        t0 = time.time()
        for n in names:
            images[key][n] = enhance_variant(llie, images["night"][n], v, device)
            imwrite_rgb(out / key / f"{n}.png", images[key][n])
        log(f"[增强] {key}: {(time.time() - t0) / len(names):.2f} s/张")
    del llie
    torch.cuda.empty_cache()

    # ---------- 2. 检测 ----------
    from ultralytics import YOLO
    det = YOLO(args.detector)
    gts = {}
    for n in names:
        h, w = images["night"][n].shape[:2]
        gts[n] = read_labels(data / "night" / "labels" / f"{n}.txt", w, h)
    dets, results = {}, {}
    for key, imgs in images.items():
        dets[key] = {}
        for n in names:
            r = det.predict(cv2.cvtColor(imgs[n], cv2.COLOR_RGB2BGR), imgsz=args.imgsz, conf=0.001,
                            classes=[COCO_CAR], device=0 if device.type == "cuda" else "cpu", verbose=False)[0]
            dets[key][n] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
        results[key] = evaluate(dets[key], gts)

    log("\n检测结果（Car，IoU 0.5；TP/FP/FN 按置信度 >= %.2f 统计）" % CONF_SHOW)
    log(f"{'输入':<18}{'AP50':<8}{'TP':<5}{'FP':<5}{'FN':<5}{'GT 框数':<6}")
    for key, r in results.items():
        log(f"{key:<20}{r['AP50']:<8.3f}{r['TP']:<5}{r['FP']:<5}{r['FN']:<5}{r['n_gt']:<6}")

    # ---------- 3. 可视化 ----------
    keys = list(images.keys())
    full_rows, crop_rows = [], []
    for n in names:
        thick = max(2, images["night"][n].shape[1] // 400)
        drawn = {k: draw(images[k][n], gts[n], *dets[k][n], thick) for k in keys}
        full_rows.append(np.hstack([titled(drawn[k], f"{k}  {n[:14]}", 480) for k in keys]))
        crops = {k: crop_around(draw(images[k][n], gts[n], *dets[k][n], 1), gts[n]) for k in keys}
        crop_rows.append(np.hstack([titled(cv2.resize(crops[k], (320, 320), interpolation=cv2.INTER_NEAREST),
                                           k, 320) for k in keys]))
    save_jpg(out / "overview_full.jpg", np.vstack(full_rows), 85)
    save_jpg(out / "overview_crops.jpg", np.vstack(crop_rows), 92)

    summary = {"time": stamp, "n_images": len(names), "detector": Path(args.detector).name,
               "imgsz": args.imgsz, "llie_weight": Path(args.llie_weight).name, "results": results}
    with open(log_dir / f"pipeline_darkdriving_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n图：{out / 'overview_full.jpg'}（整图）和 {out / 'overview_crops.jpg'}（车辆附近裁剪）")
    log("绿框 = 标注，红框 = 检测结果")


if __name__ == "__main__":
    main()
