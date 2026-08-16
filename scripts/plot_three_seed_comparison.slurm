#!/bin/sh
#SBATCH --job-name=plot-3seed
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/plot3_%j.out
#SBATCH --error=results/logs/plot3_%j.err
set -eu
cd /home/sheny-y23/rte_oe
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg /usr/bin/python3 scripts/plot_three_seed_method_comparison.py
