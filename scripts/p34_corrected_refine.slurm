#!/bin/sh
#SBATCH --job-name=p34-refine
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-23%8
#SBATCH --output=results/logs/p34refine_%A_%a.out
#SBATCH --error=results/logs/p34refine_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"

group=$((SLURM_ARRAY_TASK_ID / 6))
candidate=$((SLURM_ARRAY_TASK_ID % 6))
case "$group" in
  0) problem=p3; epsilon=1 ;;
  1) problem=p3; epsilon=1e-3 ;;
  2) problem=p4; epsilon=1 ;;
  3) problem=p4; epsilon=1e-3 ;;
esac
case "$candidate" in
  0) features=128; scale=0.5; partitions="1 1 1"; weights="" ;;
  1) features=128; scale=1.0; partitions="1 1 1"; weights="" ;;
  2) features=128; scale=2.0; partitions="1 1 1"; weights="" ;;
  3) features=256; scale=1.0; partitions="1 1 1"; weights="" ;;
  4) features=96; scale=1.0; partitions="2 1 1"; weights="" ;;
  5) features=128; scale=1.0; partitions="1 1 1"; weights="--block-weights 4 1 1 1" ;;
esac
python scripts/run_p3_oe_aprfm.py \
  --problem "$problem" --epsilon "$epsilon" --seed 11 \
  --partitions $partitions --features "$features" --scale "$scale" \
  --collocation 16 16 16 $weights --tag "corrected_c${candidate}" \
  --output-dir "results/tuning/$problem"
