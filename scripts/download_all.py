# -*- coding: utf-8 -*-
"""
在新机器上一次性下载本项目需要的数据和权重（git 仓库里不放这些大文件）。

    python scripts/download_all.py                  # 主线实验需要的全部：CIDNet 官方权重、检测器、LIS、LOL-v2
    python scripts/download_all.py --yola           # 另外下载 YOLA 官方权重和我们在 LIS 上微调的 YOLA
    python scripts/download_all.py --only ours      # 只下某几步；可选 weights / ours / detectors / lis / lolv2 / yola / yola_lis

已经存在的文件会跳过，可以反复运行。我们自己训练的权重放在 Hugging Face（见 OURS_REPO），
由 scripts/upload_to_hf.py 上传；私有仓库需要先 huggingface-cli login。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
WEIGHTS = PROJECT_ROOT / "weights"
DATA = PROJECT_ROOT / "data"

# HVI-CIDNet 官方权重（HuggingFace Fediory/HVI-CIDNet 的 weights/ 目录），SHA256 记录在 logs/04_download_weights.log
CIDNET_WEIGHTS = {
    "LOLv2_real/w_perc.pth": "1e0affc8ce89cceeb283dfa3ce7108c767cefc6f59826da37ec4c39482839db8",
    "LOLv2_real/best_PSNR.pth": "44a7f92cf02a1da322c1a34b3aa940341dc93fd53840b7f19b49f461200b00a6",
    "LOLv2_real/best_SSIM.pth": "3d7ad3e93d65effaeb937cf833b1f2b3fb78c7529a4db95c589ff0ca569347c3",
    "generalization.pth": "6395370762ec1fa0f6fe92a38c62382b1ebe21edc012f597023bfcec8b95cd27",
}
DETECTORS = ["yolo11s.pt", "yolov8s.pt", "rtdetr-l.pt"]   # ultralytics 官方发布的 COCO 预训练权重
# 我们训练的权重（Hugging Face 仓库名在上传后填进来）
OURS_REPO = "<用户名>/llie-frozen-detector"
OURS_FILES = [f"lis_ft/{n}/epoch_050.pth" for n in ("image_only", "yolo_feat", "yolo_feat_w10")]
YOLA_LIS_REPO = "<用户名>/llie-yola-lis"          # 私有
YOLA_LIS_FILES = ["yola_lis_ft/tood_lis/epoch_24.pth"]
# YOLA 官方权重（README 里的 Google Drive 文件夹）
YOLA_FOLDERS = {"tood_exdark": "1-NyMRIjztyc2RYz0i3nqRuLc2BG953HK",
                "yolov3_exdark": "1ZfmRcRCYdXvrCYM1d77sN67CmyNRi-GZ"}
# LOL-v2（Retinexformer 仓库提供的 Google Drive 镜像）
LOLV2_ID = "1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(2 ** 20), b""):
            h.update(buf)
    return h.hexdigest()


def get_cidnet_weights():
    from huggingface_hub import hf_hub_download
    for name, digest in CIDNET_WEIGHTS.items():
        dst = WEIGHTS / name
        if not dst.exists():
            hf_hub_download("Fediory/HVI-CIDNet", f"weights/{name}", local_dir=str(PROJECT_ROOT))
        ok = sha256(dst) == digest
        print(f"[CIDNet] {name}: {'SHA256 一致' if ok else 'SHA256 不一致！'}")
        if not ok:
            raise IOError(f"{dst} 的 SHA256 和记录不一致")


def get_from_hf(repo, files):
    from huggingface_hub import hf_hub_download
    if repo.startswith("<"):
        raise SystemExit(f"download_all.py 里的 Hugging Face 仓库名还没填：{repo}")
    for f in files:
        if not (WEIGHTS / f).exists():
            hf_hub_download(repo, f, local_dir=str(WEIGHTS))
        print(f"[{repo}] {f}: {(WEIGHTS / f).stat().st_size / 2 ** 20:.1f} MB")


def get_ours():
    get_from_hf(OURS_REPO, OURS_FILES)


def get_yola_lis():
    get_from_hf(YOLA_LIS_REPO, YOLA_LIS_FILES)


def get_detectors():
    from ultralytics.utils.downloads import attempt_download_asset
    for name in DETECTORS:
        attempt_download_asset(str(WEIGHTS / name))
        print(f"[检测器] {name}: {(WEIGHTS / name).stat().st_size / 2 ** 20:.1f} MB")


def get_yola():
    import gdown
    for sub, folder in YOLA_FOLDERS.items():
        out = WEIGHTS / "yola" / sub
        if out.exists() and any(out.glob("*.pth")):
            print(f"[YOLA] {sub}: 已存在，跳过")
            continue
        gdown.download_folder(id=folder, output=str(out), quiet=False)


def get_lis():
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "fetch_lis_full.py")], check=True)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "make_lis_coco.py")], check=True)


def get_lolv2():
    from fetch_lis_full import GDRIVE, download
    if (DATA / "LOLv2" / "Real_captured" / "Test" / "Low").exists():
        print("[LOL-v2] 已存在，跳过")
        return
    DATA.mkdir(exist_ok=True)
    zip_path = DATA / "LOLv2.zip"
    download(GDRIVE.format(LOLV2_ID), zip_path, print)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)
    print(f"[LOL-v2] 解压到 {DATA / 'LOLv2'}")


STEPS = {"weights": get_cidnet_weights, "ours": get_ours, "detectors": get_detectors, "lis": get_lis,
         "lolv2": get_lolv2, "yola": get_yola, "yola_lis": get_yola_lis}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yola", action="store_true", help="另外下载 YOLA 官方权重和我们在 LIS 上微调的 YOLA")
    ap.add_argument("--only", nargs="*", choices=list(STEPS), help="只做其中几步")
    args = ap.parse_args()
    steps = args.only or ["weights", "ours", "detectors", "lis", "lolv2"] + (["yola", "yola_lis"] if args.yola else [])
    WEIGHTS.mkdir(exist_ok=True)
    for s in steps:
        print(f"\n===== {s} =====", flush=True)
        STEPS[s]()
    print("\n全部完成")


if __name__ == "__main__":
    main()
