#!/usr/bin/env bash
# Usage: ./infra/start_round.sh RUN_ID [RUN_ID ...] [--smoke]
# Trains each run and then benchmarks it, sequentially. A failed run does not stop the round.

set -u

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="${PYTHON:-$REPOSITORY_ROOT/.venv/bin/python}"
BENCHMARK_DATASET="benchmark_datasets/axis_1024"
TRAINING_TIME_LIMIT="35m"
KILL_GRACE_PERIOD="60s"

run_ids=()
smoke_arguments=()
log_suffix=""
for argument in "$@"; do
    if [[ "$argument" == "--smoke" ]]; then
        smoke_arguments=(--smoke)
        log_suffix="_smoke"
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

for run_id in "${run_ids[@]}"; do
    if [[ ! -f "src/stdev/upscalers/run_${run_id}.py" ]]; then
        echo "Missing run file: src/stdev/upscalers/run_${run_id}.py" >&2
        exit 2
    fi
done

summary=()
round_failed=0

for run_id in "${run_ids[@]}"; do
    run_name="run_${run_id}"
    run_file="src/stdev/upscalers/${run_name}.py"
    log_directory="trained_models/${run_name}"
    mkdir -p "$log_directory"

    echo "=== ${run_name}: training ==="
    timeout --kill-after="$KILL_GRACE_PERIOD" "$TRAINING_TIME_LIMIT" \
        "$PYTHON" -u infra/run_training.py "$run_id" "${smoke_arguments[@]}" \
        2>&1 | tee "${log_directory}/train${log_suffix}.log"
    training_status=${PIPESTATUS[0]}

    if [[ $training_status -ne 0 ]]; then
        if [[ $training_status -eq 124 || $training_status -eq 137 ]]; then
            summary+=("${run_name}: training TIMEOUT (exit ${training_status}), benchmark skipped")
        else
            summary+=("${run_name}: training FAILED (exit ${training_status}), benchmark skipped")
        fi
        round_failed=1
        continue
    fi

    echo "=== ${run_name}: benchmark ==="
    "$PYTHON" -u scripts/run_benchmark.py \
        --dataset "$BENCHMARK_DATASET" --upscaler "$run_file" "${smoke_arguments[@]}" \
        2>&1 | tee "${log_directory}/benchmark${log_suffix}.log"
    benchmark_status=${PIPESTATUS[0]}

    if [[ $benchmark_status -ne 0 ]]; then
        summary+=("${run_name}: training ok, benchmark FAILED (exit ${benchmark_status})")
        round_failed=1
    else
        summary+=("${run_name}: ok")
    fi
done

echo "=== Round summary ==="
printf '%s\n' "${summary[@]}"
exit $round_failed
