#!/bin/bash
# Submit all experiments as separate SLURM jobs (parallel)

mkdir -p logs

echo "Submitting all CRL experiments..."

JOB_C4=$(sbatch --parsable scripts/submit_slurm.sh connect4)
echo "Connect-4: Job $JOB_C4"

JOB_CHESS=$(sbatch --parsable scripts/submit_slurm.sh chess)
echo "Chess: Job $JOB_CHESS"

JOB_GO9=$(sbatch --parsable scripts/submit_slurm.sh go9)
echo "Go 9x9: Job $JOB_GO9"

echo ""
echo "Monitor with: squeue -u $USER"
echo "Logs in: logs/"
echo "Results in: results/"
