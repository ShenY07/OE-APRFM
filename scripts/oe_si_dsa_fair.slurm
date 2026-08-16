#!/bin/sh
#SBATCH --job-name=oe-si-dsa
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%2
#SBATCH --output=results/logs/oe_si_dsa_%A_%a.out
#SBATCH --error=results/logs/oe_si_dsa_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/oe_si_dsa
problem="p$((SLURM_ARRAY_TASK_ID / 2 + 1))"
if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ]; then epsilon=1; else epsilon=1e-3; fi
/usr/bin/python -u scripts/run_oe_si_dsa.py \
  --problem "$problem" --epsilon "$epsilon" \
  --grid-1d 256 128 --grid-2d 24 24 6 \
  --tol 1e-9 --max-iter 500
