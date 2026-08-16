#!/bin/sh
#SBATCH --job-name=mm-single
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%4
#SBATCH --output=results/logs/mm_single_%A_%a.out
#SBATCH --error=results/logs/mm_single_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/mm_aprfm
problem="p$((SLURM_ARRAY_TASK_ID / 2 + 1))"
if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ]; then epsilon=1; else epsilon=1e-3; fi
case "$problem" in
  p1)
    python scripts/run_mm_aprfm_1d.py --problem p1 --epsilon "$epsilon" --seed 11 \
      --partitions 1 1 --features 64 --rcond 1e-12 --collocation 32 34 ;;
  p2)
    python scripts/run_mm_aprfm_1d.py --problem p2 --epsilon "$epsilon" --seed 11 \
      --partitions 2 4 --rho-features 128 --g-features 96 --rcond 1e-6 --collocation 32 34 ;;
  p3)
    python scripts/run_mm_aprfm_2d.py --problem p3 --epsilon "$epsilon" --seed 11 \
      --partitions 1 1 1 --features 256 --rcond 1e-12 --collocation 16 16 24 ;;
  p4)
    python scripts/run_mm_aprfm_2d.py --problem p4 --epsilon "$epsilon" --seed 11 \
      --partitions 1 1 1 --features 256 --rcond 1e-12 --collocation 16 16 25 ;;
  p5)
    python scripts/run_mm_aprfm_2d.py --problem p5 --epsilon "$epsilon" --seed 11 \
      --partitions 2 2 1 --features 96 --rcond 1e-12 --collocation 16 16 24 ;;
esac
