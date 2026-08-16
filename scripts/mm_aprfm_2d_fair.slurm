#!/bin/sh
#SBATCH --job-name=mm2d-fair
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%8
#SBATCH --output=results/logs/mm2d_%A_%a.out
#SBATCH --error=results/logs/mm2d_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/mm_aprfm
seeds="11 23 37 53 71"
group=$((SLURM_ARRAY_TASK_ID / 5))
seed=$(printf '%s\n' $seeds | sed -n "$((SLURM_ARRAY_TASK_ID % 5 + 1))p")
case "$group" in
  0) problem=p3; epsilon=1; partitions="1 1 1"; features=256 ;;
  1) problem=p3; epsilon=1e-3; partitions="1 1 1"; features=256 ;;
  2) problem=p5; epsilon=1; partitions="2 2 1"; features=96 ;;
  3) problem=p5; epsilon=1e-3; partitions="2 2 1"; features=96 ;;
esac
python scripts/run_mm_aprfm_2d.py --problem "$problem" --epsilon "$epsilon" --seed "$seed" --partitions $partitions --features "$features" --rcond 1e-12 --collocation 16 16 24

