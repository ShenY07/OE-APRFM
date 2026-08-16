#!/bin/sh
#SBATCH --job-name=p2-finalize
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/p2_finalize_%j.out
#SBATCH --error=results/logs/p2_finalize_%j.err
set -eu
cd /home/sheny-y23/rte_oe
python scripts/summarize_results.py --output results/tables/oerfm_summary.csv
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-oerfm python scripts/plot_oerfm_results.py
