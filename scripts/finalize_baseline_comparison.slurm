#!/bin/sh
#SBATCH --job-name=finish-base
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/finish_base_%j.out
#SBATCH --error=results/logs/finish_base_%j.err
set -eu
cd /home/sheny-y23/rte_oe
python3 scripts/summarize_mm_aprfm.py
python3 scripts/build_baseline_comparison.py
