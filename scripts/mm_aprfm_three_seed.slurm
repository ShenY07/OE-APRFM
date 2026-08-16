#!/bin/sh
#SBATCH --job-name=mm-3seed
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-29%16
#SBATCH --output=results/logs/mm3_%A_%a.out
#SBATCH --error=results/logs/mm3_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/mm_aprfm
case $((SLURM_ARRAY_TASK_ID % 3)) in 0) seed=7;; 1) seed=11;; 2) seed=17;; esac
case $((SLURM_ARRAY_TASK_ID / 3)) in
  0) p=p1; e=1;; 1) p=p1; e=1e-3;;
  2) p=p2; e=1;; 3) p=p2; e=1e-3;;
  4) p=p3; e=1;; 5) p=p3; e=1e-3;;
  6) p=p4; e=1;; 7) p=p4; e=1e-3;;
  8) p=p5; e=1;; 9) p=p5; e=1e-3;;
esac
case "$p" in
  p1) python scripts/run_mm_aprfm_1d.py --problem p1 --epsilon "$e" --seed "$seed" --partitions 1 1 --features 64 --rcond 1e-12 --collocation 32 34 ;;
  p2) python scripts/run_mm_aprfm_1d.py --problem p2 --epsilon "$e" --seed "$seed" --partitions 2 4 --rho-features 128 --g-features 96 --rcond 1e-6 --collocation 32 34 ;;
  p3) python scripts/run_mm_aprfm_2d.py --problem p3 --epsilon "$e" --seed "$seed" --partitions 1 1 1 --features 256 --rcond 1e-12 --collocation 16 16 24 ;;
  p4) python scripts/run_mm_aprfm_2d.py --problem p4 --epsilon "$e" --seed "$seed" --partitions 1 1 1 --features 256 --rcond 1e-12 --collocation 16 16 25 ;;
  p5) python scripts/run_mm_aprfm_2d.py --problem p5 --epsilon "$e" --seed "$seed" --partitions 2 2 1 --features 96 --rcond 1e-12 --collocation 16 16 24 ;;
esac
