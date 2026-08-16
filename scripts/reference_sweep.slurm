#!/bin/sh
#SBATCH --job-name=parity-refs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=results/logs/ref_%A_%a.out
#SBATCH --error=results/logs/ref_%A_%a.err

set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/references results/logs
export PYTHONPATH="$PWD/src"
case "$SLURM_ARRAY_TASK_ID" in
  0) problem=p2; epsilon=1; level=A ;;
  1) problem=p2; epsilon=1; level=B ;;
  2) problem=p2; epsilon=1e-3; level=A ;;
  3) problem=p2; epsilon=1e-3; level=B ;;
  4) problem=p5; epsilon=1; level=A ;;
  5) problem=p5; epsilon=1; level=B ;;
  6) problem=p5; epsilon=1e-3; level=A ;;
  7) problem=p5; epsilon=1e-3; level=B ;;
esac
python scripts/generate_reference.py --problem "$problem" --epsilon "$epsilon" --level "$level"
