#!/bin/sh
#SBATCH --job-name=oe-stab-tab
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/oestabtab_%j.out
#SBATCH --error=results/logs/oestabtab_%j.err
set -eu
cd /home/sheny-y23/rte_oe
export PYTHONPATH="$PWD/src"
python scripts/summarize_stability_experiments.py
