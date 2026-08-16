#!/bin/sh
set -eu
if [ "$#" -lt 3 ]; then
    echo "usage: $0 SCRIPT FIRST LAST [PARALLEL]" >&2
    exit 2
fi
script=$1
first=$2
last=$3
parallel=${4:-4}
mkdir -p results/logs

if [ -n "${LOCAL_ARRAY_TASK_ID:-}" ]; then
    export SLURM_ARRAY_TASK_ID=$LOCAL_ARRAY_TASK_ID
    export SLURM_CPUS_PER_TASK=4
    name=$(basename "$script" .slurm)
    exec sh "$script" >"results/logs/${name}_local_${SLURM_ARRAY_TASK_ID}.out" \
        2>"results/logs/${name}_local_${SLURM_ARRAY_TASK_ID}.err"
fi

export script
seq "$first" "$last" | xargs -n 1 -P "$parallel" sh -c \
    'LOCAL_ARRAY_TASK_ID=$1 scripts/run_array_local.sh "$script" 0 0 1' _
