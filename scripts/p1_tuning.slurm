#!/bin/sh
#SBATCH --job-name=p1-tune
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-23%12
#SBATCH --output=results/logs/p1tune_%A_%a.out
#SBATCH --error=results/logs/p1tune_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src:$PWD/scripts"
if [ "$SLURM_ARRAY_TASK_ID" -lt 12 ]; then
  epsilon=1
  index=$SLURM_ARRAY_TASK_ID
else
  epsilon=1e-8
  index=$((SLURM_ARRAY_TASK_ID - 12))
fi
python scripts/tune_p1.py --epsilon "$epsilon" --index "$index"
