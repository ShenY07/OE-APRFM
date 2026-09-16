#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
mkdir -p results/logs results/baselines/oe_apnn
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
python=${PYTHON:-python3}

run_task() {
    task=$1
    seed_index=$((task / 10)); within=$((task % 10))
    seed=$(printf '7\n11\n17\n' | sed -n "$((seed_index + 1))p")
    problem_index=$((within / 2 + 1)); problem="p${problem_index}"
    if [ $((within % 2)) -eq 0 ]; then epsilon=1; tag=1e0; else epsilon=1e-3; tag=1e-3; fi
    output="results/baselines/oe_apnn/P${problem_index}/eps_${tag}/seed_${seed}"
    log="results/logs/oe_apnn_p${problem_index}_${tag}_seed${seed}.out"
    "$python" -u baselines/oe-apnn/train.py --problem "$problem" \
        --epsilon "$epsilon" --seed "$seed" --output-dir "$output" \
        --log-every 200 >"$log" 2>&1
}

if [ "$#" -eq 1 ]; then run_task "$1"; exit 0; fi
export PYTHON
seq 0 29 | xargs -n 1 -P "${MAX_WORKERS:-4}" "$0"
