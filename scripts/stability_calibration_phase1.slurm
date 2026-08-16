#!/bin/sh
#SBATCH --job-name=oe-calib1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-95%16
#SBATCH --output=results/logs/oecal1_%A_%a.out
#SBATCH --error=results/logs/oecal1_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
out=results/stability_calibration/phase1
mkdir -p "$out" results/logs
problem_index=$((SLURM_ARRAY_TASK_ID / 48))
candidate=$((SLURM_ARRAY_TASK_ID % 48))
ratio_index=$((candidate % 4))
rcond_index=$(((candidate / 4) % 4))
scale_index=$((candidate / 16))
scale=$(printf '%s\n' 0.5 1 2 | sed -n "$((scale_index+1))p")
rcond=$(printf '%s\n' 1e-8 1e-10 1e-12 1e-14 | sed -n "$((rcond_index+1))p")
ratio=$(printf '%s\n' 4 6 8 10 | sed -n "$((ratio_index+1))p")
tag="cal1_s${scale}_r${rcond}_o${ratio}"
for epsilon in 1 1e-3 1e-6; do
  if [ "$problem_index" -eq 0 ]; then
    case "$ratio" in 4) c='14 14';; 6) c='17 17';; 8) c='19 19';; 10) c='21 21';; esac
    python scripts/run_p1_oe_aprfm.py --epsilon "$epsilon" --seed 11 --output-dir "$out" --partitions 1 1 --features 64 --scale "$scale" --rcond "$rcond" --collocation $c --block-weights 1 1 1 1 --tag "$tag"
  else
    case "$ratio" in 4) c='8 8 7';; 6) c='8 8 9';; 8) c='8 8 10';; 10) c='8 8 12';; esac
    python scripts/run_p3_oe_aprfm.py --problem p3 --epsilon "$epsilon" --seed 11 --output-dir "$out" --partitions 1 1 1 --features 128 --scale "$scale" --rcond "$rcond" --collocation $c --block-weights 1 1 1 1 --tag "$tag"
  fi
done
