# -*- coding: utf-8 -*-
"""
对比方法 YOLA（NeurIPS 2024）在 LIS 训练集上微调。按通行做法：官方配置只改数据集和类别数，
其余设置不动，从官方 ExDark 权重出发。配置在 scripts\\yola_configs\\tood_yola_lis.py。

运行环境：conda 环境 yola（不是 cidnet）。先跑一次 make_lis_coco.py 生成 COCO 格式标注。

    E:\\Anaconda\\envs\\yola\\python.exe scripts\\train_yola_lis.py
    E:\\Anaconda\\envs\\yola\\python.exe scripts\\train_yola_lis.py --resume          # 中断后接着训
    E:\\Anaconda\\envs\\yola\\python.exe scripts\\train_yola_lis.py --smoke           # 试跑 40 张图、1 个 epoch

输出：weights\\yola_lis_ft\\<name>\\（mmengine 的 checkpoint 和日志），训练完用 run_yola_lis.py 评测。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_yola_lis import register_yola  # noqa: E402

CONFIG = PROJECT_ROOT / "scripts" / "yola_configs" / "tood_yola_lis.py"
OUT_ROOT = PROJECT_ROOT / "weights" / "yola_lis_ft"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--name", default="tood_lis")
    ap.add_argument("--batch", type=int, default=0, help="batch size；0 = 官方设置（4）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="只用 40 张图跑 1 个 epoch，检查能否运行和显存")
    args = ap.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner
    register_yola()
    os.chdir(Path(args.config).parent)   # 配置里的数据和权重路径是相对配置文件所在目录写的
    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(OUT_ROOT / (args.name + ("_smoke" if args.smoke else "")))
    if args.batch:
        cfg.train_dataloader.batch_size = args.batch
    if args.smoke:
        cfg.train_dataloader.dataset.indices = 40
        cfg.train_cfg.max_epochs = 1
        cfg.default_hooks.logger = dict(type="LoggerHook", interval=1)
    cfg.resume = args.resume
    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
