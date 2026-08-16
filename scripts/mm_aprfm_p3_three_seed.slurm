#!/bin/sh
#SBATCH --job-name=mm-p3-3seed
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%1
#SBATCH --output=results/logs/mm_p3_3s_%A_%a.out
#SBATCH --error=results/logs/mm_p3_3s_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/mm_aprfm
case $((SLURM_ARRAY_TASK_ID % 3)) in 0) seed=7;; 1) seed=11;; 2) seed=17;; esac
if [ $((SLURM_ARRAY_TASK_ID / 3)) -eq 0 ]; then epsilon=1; else epsilon=1e-3; fi
python scripts/run_mm_aprfm_2d.py --problem p3 --epsilon "$epsilon" --seed "$seed" \
  --partitions 1 1 1 --features 256 --rcond 1e-12 --collocation 16 16 24
