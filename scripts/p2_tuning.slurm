#!/bin/sh
#SBATCH --job-name=p2-tune
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-107%16
#SBATCH --output=results/logs/p2tune_%A_%a.out
#SBATCH --error=results/logs/p2tune_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src:$PWD/scripts"
if [ "$SLURM_ARRAY_TASK_ID" -lt 54 ]; then
  epsilon=1
  index=$SLURM_ARRAY_TASK_ID
else
  epsilon=1e-3
  index=$((SLURM_ARRAY_TASK_ID - 54))
fi
python scripts/tune_p2.py --epsilon "$epsilon" --seed 11 --quick --index "$index"
