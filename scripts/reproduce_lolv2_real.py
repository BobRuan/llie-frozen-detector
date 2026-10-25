# -*- coding: utf-8 -*-
"""
复现 HVI-CIDNet 在 LOL-v2-Real 测试集上的官方结果。

做法：不改官方代码，直接调用官方 eval.py（推理）和 measure.py（算 PSNR/SSIM/LPIPS）。
官方脚本用的是相对路径（./datasets ./weights ./output），所以这里在官方仓库目录下建
三个目录联接（junction）指到本项目的 data / weights / outputs\\official_eval。

运行：PyCharm 里右键 Run，或
    E:\\Anaconda\\envs\\cidnet\\python.exe scripts\\reproduce_lolv2_real.py
日志：logs\\repro_lolv2_real.log，结果表：logs\\repro_lolv2_real_results.json
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_ROOT / "code" / "HVI-CIDNet"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "repro_lolv2_real.log"
RESULT_FILE = LOG_DIR / "repro_lolv2_real_results.json"
OUT_ROOT = PROJECT_ROOT / "outputs"

# 官方仓库里的相对目录 -> 本项目里的真实目录
JUNCTIONS = {
    "datasets": PROJECT_ROOT / "data",
    "weights": PROJECT_ROOT / "weights",
    "output": OUT_ROOT / "official_eval",
}

# eval.py 的权重开关 -> 官方 README 报告值 (PSNR, SSIM, LPIPS)；None 表示 README 没给
RUNS = [
    {"flag": "best_GT_mean", "weight": "LOLv2_real/w_perc.pth",
     "readme": {"normal": None, "gt_mean": (28.1387, 0.8920, 0.1008)}},
    {"flag": "best_PSNR", "weight": "LOLv2_real/best_PSNR.pth",
     "readme": {"normal": (23.9040, 0.8656, 0.1219), "gt_mean": None}},
    {"flag": "best_SSIM", "weight": "LOLv2_real/best_SSIM.pth",
     "readme": {"normal": (23.8975, 0.8705, 0.1185), "gt_mean": (28.3926, 0.8873, 0.1136)}},
]


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def make_dir_link(link, target):
    """目录链接：Windows 上用目录联接（不需要管理员权限），其他系统用符号链接。"""
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(str(target), str(link), target_is_directory=True)


def ensure_junctions():
    for name, target in JUNCTIONS.items():
        link = REPO_DIR / name
        target.mkdir(parents=True, exist_ok=True)
        if link.exists():
            continue
        make_dir_link(link, target)
        log(f"[link] {link} -> {target}")


def run(args):
    """在官方仓库目录下运行一个官方脚本，返回它的全部输出。"""
    cmd = [sys.executable] + args
    log(f"\n$ {' '.join(cmd)}")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, cwd=str(REPO_DIR), env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    # tqdm 的进度条用 \r 刷新，日志里只留每段的最后一行
    lines = [seg.rstrip("\r").split("\r")[-1] for seg in p.stdout.split("\n")]
    out = "\n".join(l for l in lines if l.strip())
    log(out)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败（退出码 {p.returncode}）：{' '.join(args)}")
    return out


def parse_metrics(text):
    vals = []
    for key in ("PSNR", "SSIM", "LPIPS"):
        m = re.search(rf"Avg\.{key}:\s*([0-9.]+)", text)
        if not m:
            raise RuntimeError(f"measure.py 输出里找不到 {key}")
        vals.append(float(m.group(1)))
    return tuple(vals)


def main():
    LOG_DIR.mkdir(exist_ok=True)
    log("=" * 70)
    log(f"开始复现  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    ensure_junctions()

    low_dir = PROJECT_ROOT / "data" / "LOLv2" / "Real_captured" / "Test" / "Low"
    gt_dir = PROJECT_ROOT / "data" / "LOLv2" / "Real_captured" / "Test" / "Normal"
    n_low, n_gt = len(list(low_dir.glob("*.png"))), len(list(gt_dir.glob("*.png")))
    log(f"测试集：Low {n_low} 张，Normal {n_gt} 张（官方应为各 100 张）")
    if n_low == 0 or n_low != n_gt:
        raise RuntimeError(f"测试集不完整：{low_dir}")

    results = []
    for r in RUNS:
        official_out = JUNCTIONS["output"] / "LOLv2_real"
        if official_out.exists():
            shutil.rmtree(official_out)  # 清掉上一个权重的输出，避免混用
        run(["eval.py", "--lol_v2_real", f"--{r['flag']}"])
        ours = {
            "normal": parse_metrics(run(["measure.py", "--lol_v2_real"])),
            "gt_mean": parse_metrics(run(["measure.py", "--lol_v2_real", "--use_GT_mean"])),
        }
        keep = OUT_ROOT / "lolv2_real" / r["flag"]
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(official_out, keep)
        results.append({"flag": r["flag"], "weight": r["weight"], "readme": r["readme"], "ours": ours})

    log("\n" + "=" * 70)
    log("结果汇总（README 值 vs 我们的值）")
    log(f"{'权重':<14}{'GT均值':<8}{'README PSNR/SSIM/LPIPS':<28}{'我们 PSNR/SSIM/LPIPS':<28}{'ΔPSNR':<8}")
    for r in results:
        for mode, tag in (("normal", "否"), ("gt_mean", "是")):
            ref, ours = r["readme"][mode], r["ours"][mode]
            ref_s = "{:.4f}/{:.4f}/{:.4f}".format(*ref) if ref else "(未报告)"
            ours_s = "{:.4f}/{:.4f}/{:.4f}".format(*ours)
            d = "{:+.4f}".format(ours[0] - ref[0]) if ref else "-"
            log(f"{r['flag']:<16}{tag:<9}{ref_s:<30}{ours_s:<30}{d:<8}")

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"\n结果已写入 {RESULT_FILE}")


if __name__ == "__main__":
    main()
