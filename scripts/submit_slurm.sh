#!/bin/bash
#SBATCH --job-name=crl_boardgame
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=nvidia
#SBATCH --output=/scratch/yd2247/crl-chess-go/logs/%j_%x.out
#SBATCH --error=/scratch/yd2247/crl-chess-go/logs/%j_%x.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL

# ============================================================
# Usage:
#   sbatch scripts/submit_slurm.sh connect4
#   sbatch scripts/submit_slurm.sh chess
#   sbatch scripts/submit_slurm.sh chess_conv
#   sbatch scripts/submit_slurm.sh go9
#   sbatch scripts/submit_slurm.sh go13
#   sbatch scripts/submit_slurm.sh all
#   sbatch scripts/submit_slurm.sh debug
#
#   Custom config:
#   sbatch scripts/submit_slurm.sh config configs/chess_large.yaml
# ============================================================

set -euo pipefail

ENV_NAME=${1:-connect4}
CONFIG_FILE=${2:-}

echo "=== CRL Board Game Experiments ==="
echo "Environment: $ENV_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"

# --- NYUAD HPC setup ---
module purge
module load cuda/11.8.0

# Use scratch for caches/tmp (avoid home quota)
export XDG_CACHE_HOME=/scratch/yd2247/.cache
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# Avoid user site-packages conflicts
export PYTHONNOUSERSITE=1

# Initialize conda
export MKL_INTERFACE_LAYER=LP64,GNU
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"

# Activate conda environment
# Option 1: Use your existing contrastive_rl env (if it has torch, numpy, matplotlib, scikit-learn, python-chess)
# conda activate contrastive_rl
# Option 2: Create a dedicated env (see README)
conda activate ccrl

# Ensure conda Python is first
export PATH="${CONDA_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# CUDA libs
[ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"

# --- Run ---
WORK_DIR=/scratch/yd2247/crl-chess-go
cd "$WORK_DIR"
mkdir -p logs results

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())' 2>/dev/null || echo 'not found')"
echo "=================================="

if [ "$ENV_NAME" == "all" ]; then
    python run_all.py --device cuda --output-dir results
elif [ "$ENV_NAME" == "config" ] && [ -n "$CONFIG_FILE" ]; then
    python run_experiments.py --config "$CONFIG_FILE" --device cuda --output-dir results
else
    python run_experiments.py --preset "$ENV_NAME" --device cuda --output-dir results
fi

echo ""
echo "Done at $(date)"
