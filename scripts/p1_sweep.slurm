#!/bin/sh
#SBATCH --job-name=p1-oe-aprfm
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-24
#SBATCH --output=results/logs/p1_%A_%a.out
#SBATCH --error=results/logs/p1_%A_%a.err

set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/raw results/logs
epsilons="1 1e-2 1e-4 1e-8 1e-16"
seeds="11 23 37 53 71"
epsilon=$(printf '%s\n' $epsilons | sed -n "$((SLURM_ARRAY_TASK_ID / 5 + 1))p")
seed=$(printf '%s\n' $seeds | sed -n "$((SLURM_ARRAY_TASK_ID % 5 + 1))p")
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
python scripts/run_p1_oe_aprfm.py --epsilon "$epsilon" --seed "$seed"
