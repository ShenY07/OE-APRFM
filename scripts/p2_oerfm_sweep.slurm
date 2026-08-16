#!/bin/sh
#SBATCH --job-name=p2-oerfm
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=results/logs/p2_%A_%a.out
#SBATCH --error=results/logs/p2_%A_%a.err

set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
epsilons="1 1e-3"
seeds="11 23 37 53 71"
epsilon=$(printf '%s\n' $epsilons | sed -n "$((SLURM_ARRAY_TASK_ID / 5 + 1))p")
seed=$(printf '%s\n' $seeds | sed -n "$((SLURM_ARRAY_TASK_ID % 5 + 1))p")
python scripts/run_p2_oe_aprfm.py --epsilon "$epsilon" --seed "$seed"
