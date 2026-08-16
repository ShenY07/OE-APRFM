#!/bin/sh
#SBATCH --job-name=p2-best
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-9%5
#SBATCH --output=results/logs/p2best_%A_%a.out
#SBATCH --error=results/logs/p2best_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
seeds="11 23 37 53 71"
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then
  epsilon=1
  rcond=1e-6
  partitions="2 4"
else
  epsilon=1e-3
  rcond=1e-12
  partitions="2 2"
fi
seed=$(printf '%s\n' $seeds | sed -n "$((SLURM_ARRAY_TASK_ID % 5 + 1))p")
python scripts/run_p2_oe_aprfm.py \
  --epsilon "$epsilon" --seed "$seed" \
  --partitions $partitions --features 64 --scale 1 --rcond "$rcond" \
  --tag tuned --output-dir results/final
