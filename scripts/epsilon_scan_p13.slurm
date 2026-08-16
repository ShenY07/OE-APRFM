#!/bin/sh
#SBATCH --job-name=oe-eps-p13
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-13%16
#SBATCH --output=results/logs/oeeps13_%A_%a.out
#SBATCH --error=results/logs/oeeps13_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
out=results/epsilon_scan
mkdir -p "$out" results/logs
index=$((SLURM_ARRAY_TASK_ID % 7))
problem_index=$((SLURM_ARRAY_TASK_ID / 7))
epsilon=$(printf '%s\n' 1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 | sed -n "$((index+1))p")
if [ "$problem_index" -eq 0 ]; then
  python scripts/run_p1_oe_aprfm.py --epsilon "$epsilon" --seed 11 --output-dir "$out" --partitions 1 1 --features 64 --rcond 1e-12 --collocation 32 32 --tag epsscan
else
  python scripts/run_p3_oe_aprfm.py --problem p3 --epsilon "$epsilon" --seed 11 --output-dir "$out" --partitions 1 1 1 --features 256 --rcond 1e-12 --collocation 16 16 16 --tag epsscan
fi
