# -*- coding: utf-8 -*-
"""
用 YOLA（NeurIPS 2024，夜间检测方法）的官方权重，在 LIS 测试集上做检测，不训练。
评测代码和 run_pipeline_lis.py 相同（AP50，IoU 0.5），结果可以和我们的方法直接比。

YOLA 官方权重是在 ExDark 上训练的，ExDark 的 12 类里能对上 LIS 的有 7 类（没有 tvmonitor），
所以只比这 7 类；我们方法的 7 类 mAP50 从 run_pipeline_lis.py 的结果文件里取同样 7 类的 AP 再平均。
在 LIS 上微调过的 YOLA 权重（train_yola_lis.py）本身就是 LIS 的 8 类，这时 8 类全比。

运行环境：conda 环境 yola（E:\\Anaconda\\envs\\yola，mmdet 3.3.0 + mmcv 2.1.0），不是 cidnet。
官方代码在 code\\YOLA\\（未修改）；code\\YOLA\\_base_ 是指向 mmdet 自带 configs\\_base_ 的目录联接，
这样官方 config 里的 '../../_base_/...' 不用改。

    E:\\Anaconda\\envs\\yola\\python.exe scripts\\run_yola_lis.py --checkpoint weights\\yola\\<权重>.pth
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import glob
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline_darkdriving import CONF_SHOW, evaluate  # noqa: E402

YOLA_DIR = PROJECT_ROOT / "code" / "YOLA"
# ============================ 默认设置 ============================
CONFIG = YOLA_DIR / "configs" / "tood" / "tood_yola_exdark.py"
DATA_DIR = PROJECT_ROOT / "data" / "LIS" / "test"
# 官方 json 的类别名顺序是错的，和 run_pipeline_lis.py 里的 NAME_FIX 相同
NAME_FIX = {"bicycle": "bicycle", "chair": "car", "diningtable": "motorbike", "bottle": "bus",
            "motorbike": "bottle", "car": "chair", "tvmonitor": "diningtable", "bus": "tvmonitor"}
# ExDark 类别名 -> LIS 类别名（ExDark 没有 tvmonitor）
EXDARK_TO_LIS = {"Bicycle": "bicycle", "Bottle": "bottle", "Bus": "bus", "Car": "car",
                 "Chair": "chair", "Motorbike": "motorbike", "Table": "diningtable"}
# ==================================================================


def register_yola():
    """让 mmdet 的包也能找到官方仓库里的 yola.py / yola_utils.py / exdark_voc.py（不拷贝、不改官方文件）。"""
    import mmdet
    # 官方 config 里写的是 '../../_base_/...'，指向 code/YOLA/_base_；把它链接到 mmdet 自带的 configs/_base_
    base = YOLA_DIR / "_base_"
    if not base.exists():
        target = Path(mmdet.__file__).parent / ".mim" / "configs" / "_base_"
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(base))
        else:
            os.symlink(str(target), str(base), target_is_directory=True)
    import mmdet.datasets as data_pkg
    import mmdet.models.detectors as det_pkg
    det_pkg.__path__.append(str(YOLA_DIR / "mmdet" / "models" / "detectors"))
    data_pkg.__path__.append(str(YOLA_DIR / "mmdet" / "datasets"))
    importlib.import_module("mmdet.models.detectors.yola")
    importlib.import_module("mmdet.datasets.exdark_voc")   # 官方权重里没存类别表，类别名从这个数据集类读
    from mmdet.utils import register_all_modules
    register_all_modules()


def load_yola(config, checkpoint, device, scale=0):
    from mmdet.apis import init_detector
    from mmengine.config import Config
    register_yola()
    cfg = Config.fromfile(str(config))
    # 推理时没有标注，去掉测试流程里的 LoadAnnotations
    cfg.test_dataloader.dataset.pipeline = [t for t in cfg.test_dataloader.dataset.pipeline
                                            if t["type"] != "LoadAnnotations"]
    if scale:   # 改测试时的缩放尺寸（官方是 608，保持长宽比）
        for t in cfg.test_dataloader.dataset.pipeline:
            if t["type"] == "Resize":
                t["scale"] = (scale, scale)
    return init_detector(cfg, str(checkpoint), device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--input", default="dark", choices=["dark", "normal"], help="检测暗图还是正常光图")
    ap.add_argument("--ours", default="", help="我们方法的结果 json；默认取 logs 里最新的 pipeline_lis_*.json")
    ap.add_argument("--scale", type=int, default=0, help="测试缩放尺寸；0 = 官方设置（608）")
    args = ap.parse_args()

    log_dir = PROJECT_ROOT / "logs"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"yola_lis_{args.input}_{stamp}.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    from mmdet.apis import inference_detector
    model = load_yola(args.config, args.checkpoint, "cuda:0", args.scale)
    classes = list(model.dataset_meta["classes"])
    log(f"YOLA  config {Path(args.config).name}  权重 {args.checkpoint}  测试尺寸 {args.scale or '官方 608'}")
    log(f"权重里的类别：{classes}")
    # 官方权重是 ExDark 类别，要映射；在 LIS 上微调过的权重本身就是 LIS 类别
    to_lis = {c: EXDARK_TO_LIS.get(c, c) for c in classes if c in EXDARK_TO_LIS or c in NAME_FIX}
    common = list(to_lis.values())

    data = Path(args.data)
    entries = json.load(open(data / "labels.json", encoding="utf-8"))
    for e in entries:
        for b in e["boxes"]:
            b["category"] = NAME_FIX[b["category"]]
    gts = {c: {e["id"]: np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in e["boxes"] if b["category"] == c],
                                 dtype=np.float32).reshape(-1, 4) for e in entries} for c in common}

    raw, t0 = {}, time.time()
    for e in entries:
        path = next((data / args.input).glob(f"{e['id']}.*"))
        p = inference_detector(model, str(path)).pred_instances
        raw[e["id"]] = (p.bboxes.cpu().numpy(), p.scores.cpu().numpy(),
                        np.array([classes[k] for k in p.labels.cpu().numpy()]))
    log(f"{len(entries)} 张 {args.input} 图，{(time.time() - t0) / len(entries):.2f} s/张")

    per_class = {}
    for c in common:
        name = next(k for k, v in to_lis.items() if v == c)
        dets = {i: (b[k == name], s[k == name]) for i, (b, s, k) in raw.items()}
        per_class[c] = evaluate(dets, gts[c])
    tp = sum(v["TP"] for v in per_class.values())
    fp = sum(v["FP"] for v in per_class.values())
    n_gt = sum(v["n_gt"] for v in per_class.values())
    yola = {"mAP50": float(np.mean([v["AP50"] for v in per_class.values()])),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(n_gt, 1),
            "per_class_AP50": {c: v["AP50"] for c, v in per_class.items()}}

    ours_file = args.ours or max(glob.glob(str(log_dir / "pipeline_lis_*.json")), key=os.path.getmtime)
    ours = json.load(open(ours_file, encoding="utf-8"))["detection"]
    log(f"\n同样 {len(common)} 类的 mAP50（{', '.join(common)}；IoU 0.5）")
    log(f"我们的方法取自 {Path(ours_file).name}；YOLA 的 P/R 按置信度 >= {CONF_SHOW}")
    log(f"{'方法':<28}{'mAP50':<8}" + "".join(f"{c[:9]:<10}" for c in common))
    for g, d in ours.items():
        aps = [d["per_class_AP50"][c] for c in common]
        log(f"{g:<30}{np.mean(aps):<8.3f}" + "".join(f"{a:<10.3f}" for a in aps))
    label = f"YOLA ({args.input})"
    log(f"{label:<30}{yola['mAP50']:<8.3f}" + "".join(f"{yola['per_class_AP50'][c]:<10.3f}" for c in common))
    log(f"YOLA 精确率 {yola['precision']:.3f}  召回率 {yola['recall']:.3f}")

    with open(log_dir / f"yola_lis_{args.input}_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"time": stamp, "checkpoint": str(args.checkpoint), "config": str(args.config),
                   "input": args.input, "classes": common, "yola": yola, "ours_file": ours_file},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
