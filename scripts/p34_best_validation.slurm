#!/bin/sh
#SBATCH --job-name=p34-best
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%8
#SBATCH --output=results/logs/p34best_%A_%a.out
#SBATCH --error=results/logs/p34best_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
seeds="11 23 37 53 71"
group=$((SLURM_ARRAY_TASK_ID / 5))
seed=$(printf '%s\n' $seeds | sed -n "$((SLURM_ARRAY_TASK_ID % 5 + 1))p")
case "$group" in
  0) problem=p3; epsilon=1; features=128 ;;
  1) problem=p3; epsilon=1e-3; features=128 ;;
  2) problem=p4; epsilon=1; features=256 ;;
  3) problem=p4; epsilon=1e-3; features=256 ;;
esac
python scripts/run_p3_oe_aprfm.py \
  --problem "$problem" --epsilon "$epsilon" --seed "$seed" \
  --partitions 1 1 1 --features "$features" --scale 1 --rcond 1e-12 \
  --collocation 16 16 16 --tag tuned --output-dir results/final
