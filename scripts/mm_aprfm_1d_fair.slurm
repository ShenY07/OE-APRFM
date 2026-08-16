#!/bin/sh
#SBATCH --job-name=mm1d-fair
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-19%10
#SBATCH --output=results/logs/mm1d_%A_%a.out
#SBATCH --error=results/logs/mm1d_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/mm_aprfm
seeds="11 23 37 53 71"
group=$((SLURM_ARRAY_TASK_ID / 5))
seed=$(printf '%s\n' $seeds | sed -n "$((SLURM_ARRAY_TASK_ID % 5 + 1))p")
case "$group" in
  0) problem=p1; epsilon=1; partitions="1 1"; extra="--features 64 --rcond 1e-12" ;;
  1) problem=p1; epsilon=1e-3; partitions="1 1"; extra="--features 64 --rcond 1e-12" ;;
  2) problem=p2; epsilon=1; partitions="2 4"; extra="--rho-features 128 --g-features 96 --rcond 1e-6" ;;
  3) problem=p2; epsilon=1e-3; partitions="2 4"; extra="--rho-features 128 --g-features 96 --rcond 1e-6" ;;
esac
python scripts/run_mm_aprfm_1d.py --problem "$problem" --epsilon "$epsilon" --seed "$seed" --partitions $partitions --collocation 32 34 $extra

