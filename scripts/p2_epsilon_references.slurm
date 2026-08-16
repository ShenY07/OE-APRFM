#!/bin/sh
#SBATCH --job-name=p2-eps-ref
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%5
#SBATCH --output=results/logs/p2epsref_%A_%a.out
#SBATCH --error=results/logs/p2epsref_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
mkdir -p results/references results/logs
case "$SLURM_ARRAY_TASK_ID" in
  0) epsilon=1e-1;; 1) epsilon=1e-2;; 2) epsilon=1e-4;;
  3) epsilon=1e-5;; 4) epsilon=1e-6;;
esac
python scripts/generate_reference.py --problem p2 --epsilon "$epsilon" --level B
