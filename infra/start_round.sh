#!/usr/bin/env bash
# Usage: ./infra/start_round.sh RUN_ID [RUN_ID ...] [--smoke]
# Trains each run and then benchmarks it. Runs execute in parallel, one per GPU (run i on GPU i).
# A failed run does not stop the others.

set -u

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="${PYTHON:-$REPOSITORY_ROOT/.venv/bin/python}"
BENCHMARK_DATASETS=("benchmark_datasets/flux_1024")
TRAINING_TIME_LIMIT="35m"
KILL_GRACE_PERIOD="60s"
NUM_GPUS="$(nvidia-smi --list-gpus | wc -l)"

run_ids=()
smoke_arguments=()
is_smoke=0
for argument in "$@"; do
    if [[ "$argument" == "--smoke" ]]; then
        smoke_arguments=(--smoke)
        is_smoke=1
    elif [[ "$argument" =~ ^[0-9]{4}$ ]]; then
        run_ids+=("$argument")
    else
        echo "Unknown argument: $argument" >&2
        echo "Usage: $0 RUN_ID [RUN_ID ...] [--smoke]" >&2
        exit 2
    fi
done

if [[ ${#run_ids[@]} -eq 0 ]]; then
    echo "Usage: $0 RUN_ID [RUN_ID ...] [--smoke]" >&2
    exit 2
fi

if [[ ${#run_ids[@]} -gt $NUM_GPUS ]]; then
    echo "Got ${#run_ids[@]} runs but only ${NUM_GPUS} GPUs" >&2
    exit 2
fi

for run_id in "${run_ids[@]}"; do
    if [[ ! -f "src/stdev/upscalers/run_${run_id}.py" ]]; then
        echo "Missing run file: src/stdev/upscalers/run_${run_id}.py" >&2
        exit 2
    fi
done

status_directory="$(mktemp -d)"
trap 'rm -rf "$status_directory"' EXIT

# Usage: log_output RUN_NAME LOG_PATH; passes stdin through with a run prefix, saving it to LOG_PATH
# unless in smoke mode.
log_output() {
    if [[ $is_smoke -eq 1 ]]; then
        sed -u "s/^/[$1] /"
    else
        tee "$2" | sed -u "s/^/[$1] /"
    fi
}

# Usage: execute_run RUN_ID GPU_INDEX; writes the one-line outcome to the status directory.
execute_run() {
    local run_id="$1"
    local run_name="run_${run_id}"
    local run_file="src/stdev/upscalers/${run_name}.py"
    local log_directory="train_logs/${run_name}"
    export CUDA_VISIBLE_DEVICES="$2"
    [[ $is_smoke -eq 0 ]] && mkdir -p "$log_directory"

    echo "=== ${run_name}: training on GPU $2 ==="
    timeout --kill-after="$KILL_GRACE_PERIOD" "$TRAINING_TIME_LIMIT" \
        "$PYTHON" -u infra/run_training.py "$run_id" "${smoke_arguments[@]}" \
        2>&1 | log_output "$run_name" "${log_directory}/train.log"
    local training_status=${PIPESTATUS[0]}

    if [[ $training_status -ne 0 ]]; then
        if [[ $training_status -eq 124 || $training_status -eq 137 ]]; then
            echo "${run_name}: training TIMEOUT (exit ${training_status}), benchmark skipped" >"$status_directory/$run_id"
        else
            echo "${run_name}: training FAILED (exit ${training_status}), benchmark skipped" >"$status_directory/$run_id"
        fi
        return
    fi

    echo "=== ${run_name}: benchmark ==="
    "$PYTHON" -u scripts/run_benchmark.py \
        --datasets "${BENCHMARK_DATASETS[@]}" --upscaler "$run_file" "${smoke_arguments[@]}" \
        2>&1 | log_output "$run_name" "${log_directory}/benchmark.log"
    local benchmark_status=${PIPESTATUS[0]}

    if [[ $benchmark_status -ne 0 ]]; then
        echo "${run_name}: training ok, benchmark FAILED (exit ${benchmark_status})" >"$status_directory/$run_id"
    else
        echo "${run_name}: ok" >"$status_directory/$run_id"
    fi
}

# Fill the shared benchmark cache once, so parallel benchmarks only read it.
if [[ $is_smoke -eq 0 ]]; then
    echo "=== Preparing benchmark cache ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u scripts/run_benchmark.py \
        --datasets "${BENCHMARK_DATASETS[@]}" --prepare-cache || exit 1
fi

for gpu_index in "${!run_ids[@]}"; do
    execute_run "${run_ids[$gpu_index]}" "$gpu_index" &
done
wait

round_failed=0
echo "=== Round summary ==="
for run_id in "${run_ids[@]}"; do
    outcome="$(cat "$status_directory/$run_id" 2>/dev/null || echo "run_${run_id}: no status (crashed)")"
    echo "$outcome"
    [[ "$outcome" == *": ok" ]] || round_failed=1
done
exit $round_failed
