#!/bin/sh
#SBATCH --job-name=oe-eps-p2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-6%7
#SBATCH --output=results/logs/oeeps2_%A_%a.out
#SBATCH --error=results/logs/oeeps2_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
out=results/epsilon_scan
mkdir -p "$out" results/logs
epsilon=$(printf '%s\n' 1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
python scripts/run_p2_oe_aprfm.py --epsilon "$epsilon" --seed 11 --output-dir "$out" --partitions 2 4 --features 64 --rcond 1e-6 --collocation 32 32 --tag epsscan
