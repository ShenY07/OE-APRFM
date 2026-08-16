#!/bin/sh
#SBATCH --job-name=mmp4-fair
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%6
#SBATCH --output=results/logs/mmp4_%A_%a.out
#SBATCH --error=results/logs/mmp4_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/mm_aprfm
seeds="11 23 37 53 71"
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then epsilon=1; else epsilon=1e-3; fi
seed=$(printf '%s\n' $seeds | sed -n "$((SLURM_ARRAY_TASK_ID % 5 + 1))p")
python scripts/run_mm_aprfm_2d.py --problem p4 --epsilon "$epsilon" --seed "$seed" --partitions 1 1 1 --features 256 --rcond 1e-12 --collocation 16 16 25
