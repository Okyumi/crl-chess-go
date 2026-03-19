#!/bin/bash
# One-time setup: create conda environment on NYUAD HPC
#
# Run this interactively (NOT via sbatch):
#   bash scripts/setup_env.sh

set -euo pipefail

module purge
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"

export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
mkdir -p "$PIP_CACHE_DIR"

echo "Creating conda environment 'ccrl'..."
conda create -n ccrl python=3.10 -y
conda activate ccrl

echo "Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Installing other dependencies..."
pip install numpy matplotlib scikit-learn tqdm python-chess pyyaml

echo ""
echo "Setup complete. Test with:"
echo "  conda activate ccrl"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "Then clone the repo to scratch:"
echo "  cd /scratch/yd2247"
echo "  git clone https://github.com/Okyumi/crl-chess-go.git"
echo "  cd crl-chess-go"
echo "  python run_experiments.py --preset debug --device cuda"
