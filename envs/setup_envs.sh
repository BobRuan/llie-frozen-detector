#!/usr/bin/env bash
# 在 Linux 服务器上建两个 conda 环境：cidnet（主线实验）和 yola（对比方法）。
# 需要已经装好 conda，显卡驱动支持 CUDA 11.8（驱动版本 >= 520）。
#   bash envs/setup_envs.sh            # 两个都建
#   bash envs/setup_envs.sh cidnet     # 只建主环境
set -euo pipefail
cd "$(dirname "$0")/.."
eval "$(conda shell.bash hook)"

TORCH_INDEX=https://download.pytorch.org/whl/cu118
MMCV_INDEX=https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

if [[ "${1:-all}" == "all" || "${1:-}" == "cidnet" ]]; then
  conda create -y -n cidnet python=3.8
  conda activate cidnet
  pip install torch==2.0.1 torchvision==0.15.2 --index-url "$TORCH_INDEX"
  pip install -r envs/requirements-cidnet.txt
  python -c "import torch; print('cidnet OK, CUDA', torch.cuda.is_available())"
  conda deactivate
fi

if [[ "${1:-all}" == "all" || "${1:-}" == "yola" ]]; then
  conda create -y -n yola --clone cidnet
  conda activate yola
  pip install mmengine==0.10.4
  pip install mmcv==2.1.0 -f "$MMCV_INDEX"
  pip install -r envs/requirements-yola.txt
  python -c "import mmcv, mmdet; from mmcv.ops import nms; print('yola OK, mmcv', mmcv.__version__, 'mmdet', mmdet.__version__)"
  conda deactivate
fi
