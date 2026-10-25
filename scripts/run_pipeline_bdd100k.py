# -*- coding: utf-8 -*-
"""
BDD100K 小样本试跑：同一个冻结的 YOLO 分别在 白天图 / 夜间原图 / 增强后的夜间图 上检测。

BDD100K 的白天图和夜间图不是同一场景，所以这里比较的是两组图各自的 AP，而不是逐张配对比较。
另加一个不用深度学习的对照：简单 gamma 校正。

输入：data\\BDD100K_sample\\{day,night}\\（由 fetch_bdd100k_sample.py 取得）
输出：outputs\\bdd100k_sample\\，日志和结果在 logs\\
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
from run_pipeline_darkdriving import CONF_SHOW, evaluate, save_jpg, titled  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================ 默认设置 ============================
DATA_DIR = PROJECT_ROOT / "data" / "BDD100K_sample"
OUT_DIR = PROJECT_ROOT / "outputs" / "bdd100k_sample"
LLIE_WEIGHT = PROJECT_ROOT / "weights" / "generalization.pth"
DETECTOR = PROJECT_ROOT / "weights" / "yolo11s.pt"   # COCO 预训练，不训练不微调
IMGSZ = 1280
GAMMA_BASELINE = 0.5      # 对照：out = in ** 0.5
NUM_VIS = 8
# 评测的类别：名字 -> (BDD100K 类别名集合, COCO 类别编号集合)
EVAL_CLASSES = {
    "car": ({"car"}, {2}),
    "vehicle(car+truck+bus)": ({"car", "truck", "bus"}, {2, 5, 7}),
    "person": ({"person"}, {0}),
}
# ==================================================================


def gamma_correct(rgb, g):
    return (np.power(rgb.astype(np.float32) / 255.0, g) * 255.0).round().astype(np.uint8)


def draw(rgb, gt_boxes, det):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for b in gt_boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
    for b, c in zip(det[0], det[1]):
        if c >= CONF_SHOW:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--llie_weight", default=str(LLIE_WEIGHT))
    ap.add_argument("--detector", default=str(DETECTOR))
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--gamma_baseline", type=float, default=GAMMA_BASELINE, help="对照用的 gamma；1.0 = 原样")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "logs"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_bdd100k_{stamp}.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = {s: {f["name"]: f for f in json.load(open(data / s / "labels.json", encoding="utf-8"))}
              for s in ("day", "night")}
    log(f"白天 {len(labels['day'])} 张，夜间 {len(labels['night'])} 张  device={device}")
    log(f"增强权重 {args.llie_weight}\n检测器 {args.detector}  imgsz={args.imgsz}")

    # 每组输入：(组名, 用哪份标注, 取图函数)
    night_dir = data / "night" / "images"
    enh_dir, gam_dir = out / "night_cidnet", out / "night_gamma"
    enh_dir.mkdir(exist_ok=True)
    gam_dir.mkdir(exist_ok=True)

    # ---------- 1. 增强夜间图 ----------
    llie = load_model(args.llie_weight, 1.0, device)
    t0 = time.time()
    brightness = {"night": [], "night_gamma": [], "night_cidnet": []}
    for name in labels["night"]:
        rgb = imread_rgb(night_dir / name)
        enh = enhance(llie, rgb, 1.0, -1, 0, device)[0]
        gam = gamma_correct(rgb, args.gamma_baseline)
        imwrite_rgb(enh_dir / (Path(name).stem + ".png"), enh)
        imwrite_rgb(gam_dir / (Path(name).stem + ".png"), gam)
        for k, im in (("night", rgb), ("night_gamma", gam), ("night_cidnet", enh)):
            brightness[k].append(float(im.mean()))
    log(f"[增强] {(time.time() - t0) / len(labels['night']):.2f} s/张（含读写）")
    brightness["day"] = [float(imread_rgb(data / "day" / "images" / n).mean()) for n in labels["day"]]
    del llie
    torch.cuda.empty_cache()

    groups = {
        "day": ("day", lambda n: data / "day" / "images" / n),
        "night": ("night", lambda n: night_dir / n),
        "night_gamma": ("night", lambda n: gam_dir / (Path(n).stem + ".png")),
        "night_cidnet": ("night", lambda n: enh_dir / (Path(n).stem + ".png")),
    }

    # ---------- 2. 检测 + 评测 ----------
    from ultralytics import YOLO
    det = YOLO(args.detector)
    coco_ids = sorted(set().union(*(c for _, c in EVAL_CLASSES.values())))
    raw, results = {}, {}
    for g, (split, path_of) in groups.items():
        raw[g] = {}
        for name in labels[split]:
            bgr = cv2.cvtColor(imread_rgb(path_of(name)), cv2.COLOR_RGB2BGR)
            r = det.predict(bgr, imgsz=args.imgsz, conf=0.001, classes=coco_ids,
                            device=0 if device.type == "cuda" else "cpu", verbose=False)[0]
            raw[g][name] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int))
        results[g] = {"mean_brightness": float(np.mean(brightness[g]))}
        for cname, (bdd_names, coco_set) in EVAL_CLASSES.items():
            gts = {n: np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in f["boxes"]
                                if b["category"] in bdd_names], dtype=np.float32).reshape(-1, 4)
                   for n, f in labels[split].items()}
            dets = {}
            for n, (boxes, conf, cls) in raw[g].items():
                m = np.isin(cls, list(coco_set))
                dets[n] = (boxes[m], conf[m])
            results[g][cname] = evaluate(dets, gts)

    log("\n检测结果（IoU 0.5；P/R 按置信度 >= %.2f 统计）" % CONF_SHOW)
    for cname in EVAL_CLASSES:
        log(f"\n类别：{cname}")
        log(f"{'输入':<16}{'平均亮度':<8}{'AP50':<8}{'精确率':<8}{'召回率':<8}{'GT 框数':<6}")
        for g in groups:
            r = results[g][cname]
            p = r["TP"] / max(r["TP"] + r["FP"], 1)
            rec = r["TP"] / max(r["n_gt"], 1)
            log(f"{g:<18}{results[g]['mean_brightness']:<12.1f}{r['AP50']:<8.3f}{p:<11.3f}{rec:<11.3f}{r['n_gt']:<6}")

    # ---------- 3. 可视化（夜间：原图 | gamma | CIDNet；绿框标注，红框检测到的车辆） ----------
    names = sorted(labels["night"])
    picks = [names[i] for i in np.linspace(0, len(names) - 1, NUM_VIS).round().astype(int)]
    rows = []
    for n in picks:
        gt = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in labels["night"][n]["boxes"]
              if b["category"] in {"car", "truck", "bus"}]
        cells = []
        for g in ("night", "night_gamma", "night_cidnet"):
            boxes, conf, cls = raw[g][n]
            m = np.isin(cls, [2, 5, 7])
            hit = int((conf[m] >= CONF_SHOW).sum())
            img = draw(imread_rgb(groups[g][1](n)), gt, (boxes[m], conf[m]))
            cells.append(titled(img, f"{g}  det {hit} / gt {len(gt)}", 640))
        rows.append(np.hstack(cells))
    save_jpg(out / "overview_night.jpg", np.vstack(rows), 88)

    with open(log_dir / f"pipeline_bdd100k_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"time": stamp, "detector": Path(args.detector).name, "imgsz": args.imgsz,
                   "llie_weight": Path(args.llie_weight).name, "gamma_baseline": args.gamma_baseline,
                   "results": results}, f, ensure_ascii=False, indent=2)
    log(f"\n图：{out / 'overview_night.jpg'}（绿框 = 车辆标注，红框 = 检测到的车辆）")


if __name__ == "__main__":
    main()
