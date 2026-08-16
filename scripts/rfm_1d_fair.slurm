#!/bin/sh
#SBATCH --job-name=rfm1d-fair
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-19%10
#SBATCH --output=results/logs/rfm1d_%A_%a.out
#SBATCH --error=results/logs/rfm1d_%A_%a.err
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/rfm
seeds="11 23 37 53 71"; group=$((SLURM_ARRAY_TASK_ID/5)); seed=$(printf '%s\n' $seeds|sed -n "$((SLURM_ARRAY_TASK_ID%5+1))p")
case "$group" in 0) p=p1;e=1;parts="1 1";f=128;r=1e-12;;1) p=p1;e=1e-3;parts="1 1";f=128;r=1e-12;;2) p=p2;e=1;parts="2 4";f=128;r=1e-6;;3) p=p2;e=1e-3;parts="2 4";f=128;r=1e-6;;esac
python scripts/run_rfm_1d.py --problem "$p" --epsilon "$e" --seed "$seed" --partitions $parts --features "$f" --rcond "$r"
