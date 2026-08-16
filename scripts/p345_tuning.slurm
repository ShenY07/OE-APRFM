#!/bin/sh
#SBATCH --job-name=p345-tune
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-53%12
#SBATCH --output=results/logs/p345tune_%A_%a.out
#SBATCH --error=results/logs/p345tune_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src:$PWD/scripts"
group=$((SLURM_ARRAY_TASK_ID / 9)); index=$((SLURM_ARRAY_TASK_ID % 9))
case "$group" in 0) problem=p3; epsilon=1;; 1) problem=p3; epsilon=1e-3;; 2) problem=p4; epsilon=1;; 3) problem=p4; epsilon=1e-3;; 4) problem=p5; epsilon=1;; 5) problem=p5; epsilon=1e-3;; esac
python scripts/tune_2d.py --problem "$problem" --epsilon "$epsilon" --index "$index"
