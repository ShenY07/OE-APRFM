#!/bin/sh
#SBATCH --job-name=si-dsa-p1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=results/logs/si_dsa_p1_%A_%a.out
#SBATCH --error=results/logs/si_dsa_p1_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/si_dsa
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then epsilon=1; else epsilon=1e-3; fi
/usr/bin/python -u scripts/run_oe_si_dsa.py \
  --problem p1 --epsilon "$epsilon" --grid-1d 256 128 \
  --tol 1e-9 --max-iter 500 --method-label si_dsa \
  --output-dir results/baselines/si_dsa
