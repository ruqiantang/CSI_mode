#!/usr/bin/env bash
# WiFo UPA 服务器一键配置（适用于阿里云 A10/A100 等 Ubuntu 22.04 GPU 服务器）
# 用法：在仓库根目录执行  bash scripts/setup_server.sh
set -euo pipefail

echo "[1/6] 检查 GPU ..."
nvidia-smi || { echo "未检测到 GPU，请确认用了预装 NVIDIA 驱动的镜像"; exit 1; }

echo "[2/6] 配置国内 pip 镜像（阿里云服务器用阿里云镜像最快）..."
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ || true
pip install --upgrade pip setuptools wheel

echo "[3/6] 安装 CUDA 版 PyTorch（关键：默认 pip 装的是 CPU 版，会跑不起来）..."
# 注意：cu121 需与服务器 CUDA 版本匹配；若装的是 CUDA 11.8 则改成 cu118
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "[4/6] 安装项目依赖 + 本仓库（editable）..."
pip install -e ".[dev]"

echo "[5/6] 安装监控工具 wandb ..."
pip install wandb

echo "[6/6] 建目录结构 ..."
mkdir -p data checkpoints experiments logs

echo "=== 验证 ==="
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
python scripts/verify_cuda.py --config configs/little.yaml --dataset D5 --samples 2 --batch-size 1 --amp

echo "=== 配置完成，可以开始传数据和训练 ==="
