#!/bin/sh
#SBATCH --job-name=oe-jconv
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-71%16
#SBATCH --output=results/logs/oejconv_%A_%a.out
#SBATCH --error=results/logs/oejconv_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
out=results/feature_convergence
mkdir -p "$out" results/logs
seed=$(printf '%s\n' 7 11 17 | sed -n "$((SLURM_ARRAY_TASK_ID%3+1))p")
group=$((SLURM_ARRAY_TASK_ID/3))
j_index=$((group%4)); epsilon_index=$(((group/4)%3)); problem_index=$((group/12))
features=$(printf '%s\n' 32 64 128 256 | sed -n "$((j_index+1))p")
epsilon=$(printf '%s\n' 1 1e-3 1e-6 | sed -n "$((epsilon_index+1))p")
if [ "$problem_index" -eq 0 ]; then
  case "$features" in 32) c='10 10';; 64) c='14 14';; 128) c='20 20';; 256) c='28 28';; esac
  python scripts/run_p1_oe_aprfm.py --epsilon "$epsilon" --seed "$seed" --output-dir "$out" --partitions 1 1 --features "$features" --rcond 1e-12 --collocation $c --tag jconv
else
  case "$features" in 32) c='4 4 4';; 64) c='5 5 5';; 128) c='6 6 6';; 256) c='8 8 8';; esac
  python scripts/run_p3_oe_aprfm.py --problem p3 --epsilon "$epsilon" --seed "$seed" --output-dir "$out" --partitions 1 1 1 --features "$features" --rcond 1e-12 --collocation $c --tag jconv
fi
