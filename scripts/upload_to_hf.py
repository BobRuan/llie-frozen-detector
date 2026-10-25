# -*- coding: utf-8 -*-
"""
把本项目训练出的权重上传到 Hugging Face，供新机器用 download_all.py 下载。

    huggingface-cli login                                   # 先登录（只需一次，用自己的 token）
    python scripts/upload_to_hf.py --repo <用户名>/llie-frozen-detector --public
    python scripts/upload_to_hf.py --repo <用户名>/llie-yola-lis --yola     # YOLA 在 LIS 上微调的权重，默认私有

上传内容（在仓库里的路径和本地 weights/ 下的路径相同）：
    默认     lis_ft/{image_only,yolo_feat,yolo_feat_w10}/epoch_050.pth   我们微调的 HVI-CIDNet（MIT 许可的衍生权重）
    --yola   yola_lis_ft/tood_lis/epoch_24.pth                            YOLA 原仓库没有许可证，建议只放私有仓库
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "weights"
OURS = [f"lis_ft/{n}/epoch_050.pth" for n in ("image_only", "yolo_feat", "yolo_feat_w10")]
YOLA = ["yola_lis_ft/tood_lis/epoch_24.pth"]

CARD = """---
license: mit
tags:
- low-light-image-enhancement
- object-detection
---

# HVI-CIDNet fine-tuned on LIS with a frozen-detector loss

Weights for an ENGR 501 course project. Code: {github}

HVI-CIDNet (Yan et al., CVPR 2025, https://github.com/Fediory/HVI-CIDNet, MIT License) fine-tuned on the
LIS dataset (Chen et al., IJCV 2023) starting from the official `LOLv2_real/w_perc.pth`. Loss = official
image loss + lambda x L1 distance between frozen YOLO11s neck features (layers 16/19/22) of the enhanced
image and of the paired normal-light image. The detector is never updated.

| File | lambda | LIS test mAP50 (frozen YOLO11s, 8 classes) | PSNR |
|---|---|---|---|
| `lis_ft/image_only/epoch_050.pth` | 0 | 0.594 | 21.45 |
| `lis_ft/yolo_feat/epoch_050.pth` | 1 | 0.616 | 21.50 |
| `lis_ft/yolo_feat_w10/epoch_050.pth` | 10 | 0.625 | 21.45 |

For reference: dark input 0.411, normal-light input 0.719, off-the-shelf HVI-CIDNet 0.466.
Load with the official `net.CIDNet.CIDNet` class (`model.load_state_dict(torch.load(path))`).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face 仓库名，例如 用户名/llie-frozen-detector")
    ap.add_argument("--public", action="store_true", help="公开仓库；默认私有")
    ap.add_argument("--yola", action="store_true", help="上传 YOLA 在 LIS 上微调的权重，而不是我们的 CIDNet 权重")
    ap.add_argument("--github", default="(GitHub link)", help="写进说明页的代码仓库地址")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"登录用户：{api.whoami()['name']}")
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    files = YOLA if args.yola else OURS
    for f in files:
        api.upload_file(path_or_fileobj=str(WEIGHTS / f), path_in_repo=f, repo_id=args.repo)
        print(f"已上传 {f}")
    if not args.yola:
        api.upload_file(path_or_fileobj=CARD.format(github=args.github).encode("utf-8"),
                        path_in_repo="README.md", repo_id=args.repo)
        print("已上传说明页 README.md")
    print(f"完成：https://huggingface.co/{args.repo}（{'公开' if args.public else '私有'}）")


if __name__ == "__main__":
    main()
