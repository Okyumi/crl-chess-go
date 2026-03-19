#!/bin/bash
# Submit all CRL experiments as separate SLURM jobs (run in parallel on NYUAD HPC)

WORK_DIR=/scratch/yd2247/crl-chess-go
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/results"

echo "Submitting CRL Board Game experiments to NYUAD HPC..."
echo ""

JOB_C4=$(sbatch --parsable --job-name=crl_c4 scripts/submit_slurm.sh connect4)
echo "Connect-4:   Job $JOB_C4"

JOB_CHESS=$(sbatch --parsable --job-name=crl_chess scripts/submit_slurm.sh chess_conv)
echo "Chess (conv): Job $JOB_CHESS"

JOB_GO9=$(sbatch --parsable --job-name=crl_go9 scripts/submit_slurm.sh go9)
echo "Go 9x9:      Job $JOB_GO9"

echo ""
echo "Monitor:  squeue -u yd2247"
echo "Logs:     ls $WORK_DIR/logs/"
echo "Results:  ls $WORK_DIR/results/"
echo ""
echo "Cancel all: scancel $JOB_C4 $JOB_CHESS $JOB_GO9"
