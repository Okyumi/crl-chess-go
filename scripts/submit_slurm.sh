#!/bin/bash
#SBATCH --job-name=crl-chess-go
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu         # <-- Adjust to your HPC partition name

# ============================================================
# Usage:
#   sbatch scripts/submit_slurm.sh connect4
#   sbatch scripts/submit_slurm.sh chess
#   sbatch scripts/submit_slurm.sh go9
#   sbatch scripts/submit_slurm.sh all
# ============================================================

ENV_NAME=${1:-connect4}

echo "=== CRL Board Game Experiments ==="
echo "Environment: $ENV_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=================================="

mkdir -p logs results

# Activate your environment (adjust as needed)
# source activate ccrl
# conda activate ccrl
# module load python/3.10 cuda/12.1

if [ "$ENV_NAME" == "all" ]; then
    python run_all.py --device cuda --output-dir results
else
    python run_experiments.py --preset $ENV_NAME --device cuda --output-dir results
fi

echo "Done at $(date)"
