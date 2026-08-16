#!/bin/sh
#SBATCH --job-name=oe-3seed
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --output=results/logs/oe3_%A_%a.out
#SBATCH --error=results/logs/oe3_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
mkdir -p results/logs results/baselines/oe_aprfm_3seed
case $((SLURM_ARRAY_TASK_ID % 2)) in 0) seed=7;; 1) seed=17;; esac
case $((SLURM_ARRAY_TASK_ID / 2)) in
  0) p=p1; e=1;; 1) p=p1; e=1e-3;;
  2) p=p2; e=1;; 3) p=p2; e=1e-3;;
  4) p=p3; e=1;; 5) p=p3; e=1e-3;;
  6) p=p4; e=1;; 7) p=p4; e=1e-3;;
  8) p=p5; e=1;; 9) p=p5; e=1e-3;;
esac
out=results/baselines/oe_aprfm_3seed
case "$p" in
  p1) python scripts/run_p1_oe_aprfm.py --epsilon "$e" --seed "$seed" --output-dir "$out" --partitions 1 1 --features 64 --rcond 1e-12 --tag 3seed ;;
  p2) python scripts/run_p2_oe_aprfm.py --epsilon "$e" --seed "$seed" --output-dir "$out" --partitions 2 4 --features 64 --rcond 1e-6 --tag 3seed ;;
  p3) python scripts/run_p3_oe_aprfm.py --problem p3 --epsilon "$e" --seed "$seed" --output-dir "$out" --partitions 1 1 1 --features 256 --rcond 1e-12 --collocation 16 16 16 --tag 3seed ;;
  p4) python scripts/run_p3_oe_aprfm.py --problem p4 --epsilon "$e" --seed "$seed" --output-dir "$out" --partitions 1 1 1 --features 256 --rcond 1e-12 --collocation 16 16 16 --tag 3seed ;;
  p5) python scripts/run_p3_oe_aprfm.py --problem p5 --epsilon "$e" --seed "$seed" --output-dir "$out" --partitions 2 2 1 --features 96 --rcond 1e-12 --collocation 16 16 16 --tag 3seed ;;
esac
