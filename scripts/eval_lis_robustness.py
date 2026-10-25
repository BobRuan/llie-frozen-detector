# -*- coding: utf-8 -*-
"""
LIS 测试集上的两项稳健性检查（不训练，用 outputs\\lis_test\\ 里已有的增强结果）：

  1) 换检测器：除了训练时当损失用的 YOLO11s，再用训练中没见过的 YOLOv8s 和 RT-DETR-L
     检测同一批图，看各方法的排序是否不变（检查增强是否只对 YOLO11s 过拟合）。
  2) 自助抽样（bootstrap）：对 669 张测试图有放回抽样 2000 次，给出每个方法 mAP50 的 95% 置信区间，
     以及方法之间差值的配对置信区间。

评测口径和 run_pipeline_lis.py 相同：8 类、IoU 0.5、全点插值 AP。每张图内按置信度从高到低贪心匹配，
所以每个检测框是 TP 还是 FP 和抽样无关，抽样时只需要按图重新加权，速度很快。
脚本会先检查“不抽样”时算出的 mAP50 和 run_pipeline_lis.py 的实现完全一致。

    python scripts\\eval_lis_robustness.py
输出：logs\\robustness_lis_<时间>.log / .json；原始检测结果缓存在 outputs\\lis_test\\dets\\
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline_darkdriving import IOU_THR, evaluate, iou_one_to_many  # noqa: E402
from run_pipeline_lis import LIS_TO_COCO, NAME_FIX, imread_rgb  # noqa: E402

# ============================ 默认设置 ============================
DATA_DIR = PROJECT_ROOT / "data" / "LIS" / "test"
ENH_DIR = PROJECT_ROOT / "outputs" / "lis_test"
GROUPS = ["normal", "dark", "cidnet_lolv2real", "ft_image_only", "ft_yolo_feat", "ft_yolo_feat_w10"]
# 名字 -> ultralytics 权重；没有的会由 ultralytics 自动下载到 weights\\
DETECTORS = {"yolo11s": "yolo11s.pt", "yolov8s": "yolov8s.pt", "rtdetr-l": "rtdetr-l.pt"}
IMGSZ = 1280
N_BOOT = 2000
SEED = 0
# 要算配对差值置信区间的方法对 (A, B)，报告 A - B
PAIRS = [("ft_yolo_feat", "ft_image_only"), ("ft_yolo_feat_w10", "ft_image_only"),
         ("ft_yolo_feat_w10", "ft_yolo_feat"), ("ft_image_only", "cidnet_lolv2real"),
         ("ft_yolo_feat_w10", "dark")]
# ==================================================================


def match_flags(boxes, scores, gt):
    """单张图、单个类别：按置信度从高到低贪心匹配，返回每个检测框是否 TP（和 evaluate() 的规则相同）。"""
    order = np.argsort(-scores, kind="stable")
    tp = np.zeros(len(scores), dtype=bool)
    used = np.zeros(len(gt), dtype=bool)
    for k in order:
        if len(gt):
            ious = iou_one_to_many(boxes[k], gt)
            ious[used] = 0
            j = int(ious.argmax())
            if ious[j] >= IOU_THR:
                used[j], tp[k] = True, True
    return tp


def ap_weighted(conf, tp, img_idx, n_gt_img, w):
    """按图加权的全点插值 AP。conf 已按降序排好；w[i] 是第 i 张图在本次抽样里出现的次数。"""
    wd = w[img_idx]
    tp_c = np.cumsum(wd * tp)
    fp_c = np.cumsum(wd * ~tp)
    n_gt = float(np.dot(w, n_gt_img))
    recall = tp_c / max(n_gt, 1e-9)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[1], precision, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    ap.add_argument("--groups", nargs="*", default=GROUPS)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    log_dir = PROJECT_ROOT / "logs"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"robustness_lis_{stamp}.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    entries = json.load(open(DATA_DIR / "labels.json", encoding="utf-8"))
    ids = [e["id"] for e in entries]
    gts = {c: [np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in e["boxes"] if NAME_FIX[b["category"]] == c],
                        dtype=np.float32).reshape(-1, 4) for e in entries] for c in LIS_TO_COCO}
    path_of = {"normal": lambda i: next((DATA_DIR / "normal").glob(f"{i}.*")),
               "dark": lambda i: next((DATA_DIR / "dark").glob(f"{i}.*"))}
    for g in args.groups:
        path_of.setdefault(g, (lambda g: lambda i: ENH_DIR / g / f"{i}.png")(g))

    # ---------- 1. 检测（结果缓存到 npz，重跑时直接读） ----------
    from ultralytics import RTDETR, YOLO
    cache = ENH_DIR / "dets"
    cache.mkdir(exist_ok=True)
    raw = {}
    for d in args.detectors:
        model = None
        for g in args.groups:
            f = cache / f"{d}__{g}.npz"
            if not f.exists():
                if model is None:
                    w = PROJECT_ROOT / "weights" / DETECTORS[d]
                    src = str(w) if w.exists() else DETECTORS[d]
                    model = (RTDETR if d.startswith("rtdetr") else YOLO)(src)
                t0 = time.time()
                b, s, k, n = [], [], [], []
                for i in ids:
                    r = model.predict(cv2.cvtColor(imread_rgb(path_of[g](i)), cv2.COLOR_RGB2BGR), imgsz=IMGSZ,
                                      conf=0.001, classes=sorted(LIS_TO_COCO.values()), device=0, verbose=False)[0]
                    b.append(r.boxes.xyxy.cpu().numpy())
                    s.append(r.boxes.conf.cpu().numpy())
                    k.append(r.boxes.cls.cpu().numpy().astype(int))
                    n.append(len(r.boxes))
                np.savez(f, boxes=np.concatenate(b), conf=np.concatenate(s), cls=np.concatenate(k), n=np.array(n))
                log(f"[检测] {d} / {g}: {(time.time() - t0) / len(ids):.3f} s/张")
            z = np.load(f)
            splits = np.cumsum(z["n"])[:-1]
            raw[d, g] = list(zip(np.split(z["boxes"], splits), np.split(z["conf"], splits), np.split(z["cls"], splits)))

    # ---------- 2. 每个检测框的 TP/FP（和抽样无关） ----------
    prepared = {}
    n_gt_img = {c: np.array([len(x) for x in gts[c]], dtype=np.float64) for c in LIS_TO_COCO}
    for (d, g), per_img in raw.items():
        for c, coco_id in LIS_TO_COCO.items():
            confs, tps, idxs = [], [], []
            for i, (b, s, k) in enumerate(per_img):
                m = k == coco_id
                confs.append(s[m])
                tps.append(match_flags(b[m], s[m], gts[c][i]))
                idxs.append(np.full(m.sum(), i))
            conf, tp, idx = np.concatenate(confs), np.concatenate(tps), np.concatenate(idxs)
            order = np.argsort(-conf, kind="stable")
            prepared[d, g, c] = (conf[order], tp[order], idx[order])

    def map50(d, g, w):
        return float(np.mean([ap_weighted(*prepared[d, g, c], n_gt_img[c], w) for c in LIS_TO_COCO]))

    # 自检：不抽样（每张图权重 1）时，结果要和 evaluate() 完全一致
    ones = np.ones(len(ids))
    d0, g0 = args.detectors[0], args.groups[-1]
    ref = np.mean([evaluate({i: (b[k == LIS_TO_COCO[c]], s[k == LIS_TO_COCO[c]]) for i, (b, s, k) in enumerate(raw[d0, g0])},
                            {i: gts[c][i] for i in range(len(ids))})["AP50"] for c in LIS_TO_COCO])
    assert abs(ref - map50(d0, g0, ones)) < 1e-9, (ref, map50(d0, g0, ones))
    log(f"自检通过：{d0} / {g0} 的 mAP50 与 run_pipeline_lis.py 的实现一致（{ref:.4f}）")

    # ---------- 3. 自助抽样 ----------
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=args.n_boot).astype(np.float64)
    t0 = time.time()
    boot = {(d, g): np.array([map50(d, g, w) for w in weights]) for d in args.detectors for g in args.groups}
    log(f"自助抽样 {args.n_boot} 次，用时 {time.time() - t0:.0f} s")

    result = {"time": stamp, "n_images": len(ids), "n_boot": args.n_boot, "imgsz": IMGSZ, "detectors": {}}
    for d in args.detectors:
        log(f"\n===== 检测器 {d}{'（训练时当损失用的检测器）' if d == 'yolo11s' else '（训练中没见过）'} =====")
        log(f"{'输入':<22}{'mAP50':<8}{'95% 置信区间':<18}")
        res = {"mAP50": {}, "pairs": {}}
        for g in args.groups:
            point = map50(d, g, ones)
            lo, hi = np.percentile(boot[d, g], [2.5, 97.5])
            res["mAP50"][g] = {"point": point, "ci95": [float(lo), float(hi)]}
            log(f"{g:<24}{point:<8.3f}[{lo:.3f}, {hi:.3f}]")
        log("差值（配对抽样）")
        for a, b in PAIRS:
            if a in args.groups and b in args.groups:
                diff = boot[d, a] - boot[d, b]
                point = map50(d, a, ones) - map50(d, b, ones)
                lo, hi = np.percentile(diff, [2.5, 97.5])
                p_le0 = float(np.mean(diff <= 0))
                res["pairs"][f"{a} - {b}"] = {"point": point, "ci95": [float(lo), float(hi)], "frac_le_0": p_le0}
                log(f"  {a} - {b}: {point:+.3f}  [{lo:+.3f}, {hi:+.3f}]  抽样中差值 <= 0 的比例 {p_le0:.3f}")
        result["detectors"][d] = res

    with open(log_dir / f"robustness_lis_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
