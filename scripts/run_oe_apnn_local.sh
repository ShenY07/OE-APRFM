#!/bin/sh
set -eu
cd /home/sheny-y23/rte_oe
mkdir -p results/logs results/baselines/oe_apnn
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [ "$#" -eq 1 ]; then
    task=$1
    problem_index=$((task / 2 + 1))
    problem="p${problem_index}"
    if [ $((task % 2)) -eq 0 ]; then epsilon=1; else epsilon=1e-3; fi
    output="results/baselines/oe_apnn/${problem}_eps_${epsilon}_seed_11"
    log="results/logs/oe_apnn_local_${task}.out"
    .venv/bin/python -u baselines/oe-apnn/train.py \
        --problem "$problem" --epsilon "$epsilon" --seed 11 \
        --output-dir "$output" --log-every 100 >"$log" 2>&1
    exit 0
fi

seq 0 9 | xargs -n 1 -P 5 "$0"
