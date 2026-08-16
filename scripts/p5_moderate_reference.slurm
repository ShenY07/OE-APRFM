#!/bin/sh
#SBATCH --job-name=p5-ref-1e-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=results/logs/p5moderate_%A_%a.out
#SBATCH --error=results/logs/p5moderate_%A_%a.err

set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then level=A; else level=B; fi
epsilon=${P5_EPSILON:-1e-1}
python scripts/generate_reference.py --problem p5 --epsilon "$epsilon" --level "$level"
