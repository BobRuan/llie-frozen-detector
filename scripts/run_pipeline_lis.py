# -*- coding: utf-8 -*-
"""
LIS 配对数据上的完整管线：正常光图 / 暗图 / 增强后的暗图，分别做
  1) 图像层面：增强图和同一场景的正常光图有多接近（PSNR / SSIM）
  2) 检测层面：同一个冻结的 YOLO 的 mAP50（8 类）

输入：data\\LIS_sample\\{normal,dark}\\ 和 labels.json（由 fetch_lis_sample.py 取得）
输出：outputs\\lis_sample\\，日志和结果在 logs\\
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "HVI-CIDNet"))
from infer_folder import enhance, imread_rgb, imwrite_rgb, load_model  # noqa: E402
from measure import calculate_psnr, calculate_ssim  # noqa: E402  官方评测脚本里的实现
from run_pipeline_darkdriving import CONF_SHOW, evaluate, save_jpg, titled  # noqa: E402

# ============================ 默认设置 ============================
DATA_DIR = PROJECT_ROOT / "data" / "LIS_sample"
OUT_DIR = PROJECT_ROOT / "outputs" / "lis_sample"
DETECTOR = PROJECT_ROOT / "weights" / "yolo11s.pt"   # COCO 预训练，不训练不微调
IMGSZ = 1280
NUM_VIS = 8
# 增强方法：名字 -> CIDNet 权重
CIDNET_WEIGHTS = {
    "cidnet_generalization": PROJECT_ROOT / "weights" / "generalization.pth",
    "cidnet_lolv2real": PROJECT_ROOT / "weights" / "LOLv2_real" / "w_perc.pth",
}
AUTO_GAMMA_TARGET = 0.4   # 对照：逐张选 gamma，使平均亮度变成 0.4（不用参考图）
# LIS 类别名 -> COCO 类别编号
LIS_TO_COCO = {"bicycle": 1, "car": 2, "motorbike": 3, "bus": 5,
               "bottle": 39, "chair": 56, "diningtable": 60, "tvmonitor": 62}
# 官方 json 的 categories 表里类别名的顺序是错的（check_lis_classes.py 验证过：标成 car 的其实是椅子，等等）。
# 实际顺序是论文里写的 bicycle, car, motorbike, bus, bottle, chair, diningtable, tvmonitor。
# 这里把 json 里的名字改回真实类别。
NAME_FIX = {"bicycle": "bicycle", "chair": "car", "diningtable": "motorbike", "bottle": "bus",
            "motorbike": "bottle", "car": "chair", "tvmonitor": "diningtable", "bus": "tvmonitor"}
# ==================================================================


def auto_gamma(rgb, target):
    x = rgb.astype(np.float32) / 255.0
    g = np.log(target) / np.log(np.clip(x.mean(), 1e-4, 0.999))
    return (np.power(x, min(g, 1.0)) * 255.0).round().astype(np.uint8)


def draw(rgb, gt, det):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for b in gt:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    for b, c in zip(det[0], det[1]):
        if c >= CONF_SHOW:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--detector", default=str(DETECTOR))
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--weights", nargs="*", default=[], metavar="名字=路径",
                    help="额外要评测的 CIDNet 权重，例如 ft_yolo=weights\\lis_ft\\yolo_feat\\epoch_050.pth")
    ap.add_argument("--only_extra", action="store_true", help="只评测 --weights 给的权重（外加正常光图和暗图），省时间")
    ap.add_argument("--reuse", action="store_true", help="输出文件夹里已有完整的增强结果就直接用，不重新增强")
    args = ap.parse_args()
    cidnet_weights = dict({} if args.only_extra else CIDNET_WEIGHTS,
                          **{k: Path(v) for k, v in (w.split("=", 1) for w in args.weights)})

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "logs"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_lis_{stamp}.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = json.load(open(data / "labels.json", encoding="utf-8"))
    for e in entries:
        for b in e["boxes"]:
            b["category"] = NAME_FIX[b["category"]]
    ids = [e["id"] for e in entries]
    find = lambda part, i: next((data / part).glob(f"{i}.*"))
    log(f"{len(ids)} 对  device={device}  检测器 {args.detector}  imgsz={args.imgsz}")

    # ---------- 1. 准备各组图像（增强结果存盘，之后按路径读） ----------
    groups = {"normal": lambda i: find("normal", i), "dark": lambda i: find("dark", i)}
    if not args.only_extra:
        (out / "auto_gamma").mkdir(exist_ok=True)
        for i in ids:
            imwrite_rgb(out / "auto_gamma" / f"{i}.png", auto_gamma(imread_rgb(find("dark", i)), AUTO_GAMMA_TARGET))
        groups["auto_gamma"] = lambda i: out / "auto_gamma" / f"{i}.png"
    for name, weight in cidnet_weights.items():
        (out / name).mkdir(exist_ok=True)
        groups[name] = (lambda n: lambda i: out / n / f"{i}.png")(name)
        if args.reuse and all((out / name / f"{i}.png").exists() for i in ids):
            log(f"[增强] {name}: 沿用 {out / name} 里已有的结果")
            continue
        model = load_model(weight, 1.0, device)
        t0 = time.time()
        for i in ids:
            imwrite_rgb(out / name / f"{i}.png", enhance(model, imread_rgb(find("dark", i)), 1.0, -1, 0, device)[0])
        log(f"[增强] {name}: {(time.time() - t0) / len(ids):.2f} s/张（含读写）")
        del model
        torch.cuda.empty_cache()

    # ---------- 2. 图像层面：和正常光图比 ----------
    quality = {}
    for g, path_of in groups.items():
        ps, ss, br = [], [], []
        for i in ids:
            im = imread_rgb(path_of(i))
            br.append(float(im.mean()))
            if g != "normal":
                ref = imread_rgb(find("normal", i))
                ps.append(float(calculate_psnr(im, ref)))
                ss.append(float(calculate_ssim(im, ref)))
        quality[g] = {"brightness": float(np.mean(br)),
                      "PSNR": float(np.mean(ps)) if ps else None, "SSIM": float(np.mean(ss)) if ss else None}

    # ---------- 3. 检测层面 ----------
    from ultralytics import YOLO
    det = YOLO(args.detector)
    gts = {c: {e["id"]: np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in e["boxes"] if b["category"] == c],
                                 dtype=np.float32).reshape(-1, 4) for e in entries} for c in LIS_TO_COCO}
    raw, detres = {}, {}
    for g, path_of in groups.items():
        raw[g] = {}
        for i in ids:
            r = det.predict(cv2.cvtColor(imread_rgb(path_of(i)), cv2.COLOR_RGB2BGR), imgsz=args.imgsz, conf=0.001,
                            classes=sorted(LIS_TO_COCO.values()), device=0 if device.type == "cuda" else "cpu",
                            verbose=False)[0]
            raw[g][i] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))
        per_class = {}
        for c, coco_id in LIS_TO_COCO.items():
            dets = {i: (b[k == coco_id], s[k == coco_id]) for i, (b, s, k) in raw[g].items()}
            per_class[c] = evaluate(dets, gts[c])
        tp = sum(v["TP"] for v in per_class.values())
        fp = sum(v["FP"] for v in per_class.values())
        n_gt = sum(v["n_gt"] for v in per_class.values())
        detres[g] = {"mAP50": float(np.mean([v["AP50"] for v in per_class.values()])),
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(n_gt, 1), "n_gt": n_gt,
                     "per_class_AP50": {c: v["AP50"] for c, v in per_class.items()}}

    log("\n结果（PSNR/SSIM 是和同一场景的正常光图比；mAP50 是 8 类平均，IoU 0.5；P/R 按置信度 >= %.2f）" % CONF_SHOW)
    log(f"{'输入':<24}{'亮度':<8}{'PSNR':<8}{'SSIM':<8}{'mAP50':<8}{'精确率':<8}{'召回率':<8}")
    for g in groups:
        q, d = quality[g], detres[g]
        ps = f"{q['PSNR']:.2f}" if q["PSNR"] is not None else "-"
        ss = f"{q['SSIM']:.3f}" if q["SSIM"] is not None else "-"
        log(f"{g:<26}{q['brightness']:<10.1f}{ps:<8}{ss:<8}{d['mAP50']:<8.3f}{d['precision']:<11.3f}{d['recall']:<11.3f}")
    log("\n各类 AP50")
    log(f"{'输入':<24}" + "".join(f"{c[:9]:<10}" for c in LIS_TO_COCO))
    for g in groups:
        log(f"{g:<26}" + "".join(f"{detres[g]['per_class_AP50'][c]:<10.3f}" for c in LIS_TO_COCO))

    # ---------- 4. 可视化 ----------
    picks = [ids[k] for k in np.linspace(0, len(ids) - 1, NUM_VIS).round().astype(int)]
    by_id = {e["id"]: e for e in entries}
    rows = []
    for i in picks:
        gt = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in by_id[i]["boxes"]]
        cells = []
        for g, path_of in groups.items():
            b, s, _ = raw[g][i]
            cells.append(titled(draw(imread_rgb(path_of(i)), gt, (b, s)),
                                f"{g}  det {int((s >= CONF_SHOW).sum())}/gt {len(gt)}", 420))
        rows.append(np.hstack(cells))
    save_jpg(out / "overview.jpg", np.vstack(rows), 88)

    with open(log_dir / f"pipeline_lis_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"time": stamp, "n_pairs": len(ids), "detector": Path(args.detector).name, "imgsz": args.imgsz,
                   "quality": quality, "detection": detres}, f, ensure_ascii=False, indent=2)
    log(f"\n图：{out / 'overview.jpg'}（绿框 = 标注，红框 = 检测结果）")


if __name__ == "__main__":
    main()
