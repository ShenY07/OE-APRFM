#!/bin/sh
#SBATCH --job-name=p5-ref-stiff
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=results/logs/p5stiff_%A_%a.out
#SBATCH --error=results/logs/p5stiff_%A_%a.err

set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then level=A; else level=B; fi
python scripts/generate_reference.py \
  --problem p5 --epsilon 1e-3 --level "$level" --max-iter 500
