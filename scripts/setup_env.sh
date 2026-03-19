#!/bin/bash
# One-time setup: create conda environment on NYUAD HPC
#
# Run this interactively (NOT via sbatch):
#   bash scripts/setup_env.sh
#
# IMPORTANT (disk quota): Default conda puts envs under ~/.conda/envs/ (small HOME
# quota). PyTorch+CUDA is multi-GB and must live on scratch. This script uses:
#   - CONDA_PKGS_DIRS on scratch (conda download/extract cache)
#   - env at /scratch/$USER/.conda/envs/ccrl (pip installs go here)
# If you already created ccrl in HOME and hit quota, free space first:
#   conda deactivate 2>/dev/null || true
#   conda env remove -n ccrl -y
#   rm -rf ~/.conda/pkgs/*   # optional: reclaim conda package cache in HOME

set -eo pipefail

# Conda hooks reference vars that may be unset; avoid "unbound variable" with nounset
set +u
module purge
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
# Leave nounset off: conda/pip sometimes touch unset vars on HPC modules.

SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
export CONDA_PKGS_DIRS="${SCRATCH_ROOT}/.conda/pkgs"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/.cache/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

ENV_PREFIX="${SCRATCH_ROOT}/.conda/envs/ccrl"

echo "Creating conda environment on scratch: $ENV_PREFIX"
if [[ -d "$ENV_PREFIX" ]]; then
  echo "Environment already exists at $ENV_PREFIX — remove it first if you want a clean install."
else
  conda create --prefix "$ENV_PREFIX" python=3.10 -y
fi
conda activate "$ENV_PREFIX"

echo "Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Installing other dependencies..."
pip install numpy matplotlib scikit-learn tqdm python-chess pyyaml

echo ""
echo "Setup complete. Test with:"
echo "  conda activate $ENV_PREFIX"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "Then clone the repo to scratch:"
echo "  cd /scratch/yd2247"
echo "  git clone https://github.com/Okyumi/crl-chess-go.git"
echo "  cd crl-chess-go"
echo "  python run_experiments.py --preset debug --device cuda"
