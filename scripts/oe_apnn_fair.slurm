#!/bin/sh
#SBATCH --job-name=oe-apnn
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-9%5
#SBATCH --output=results/logs/oe_apnn_%A_%a.out
#SBATCH --error=results/logs/oe_apnn_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/oe_apnn

problem_index=$((SLURM_ARRAY_TASK_ID / 2 + 1))
epsilon_index=$((SLURM_ARRAY_TASK_ID % 2))
problem="p${problem_index}"
if [ "$epsilon_index" -eq 0 ]; then
  epsilon=1
else
  epsilon=1e-3
fi
output="results/baselines/oe_apnn/${problem}_eps_${epsilon}_seed_11"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
.venv/bin/python -u baselines/oe-apnn/train.py \
  --problem "$problem" --epsilon "$epsilon" --seed 11 \
  --output-dir "$output" --log-every 100
