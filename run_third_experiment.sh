#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_binary=${PYTHON_BINARY:-/root/nvfp4-validation/venv/bin/python}
result_directory=$(realpath -m "${1:-"$script_directory/third-repro-$(date -u +%Y%m%dT%H%M%SZ)"}")
baseline_directory=$(realpath -m "${2:-"$script_directory/host_results"}")
comparison_directory=$(realpath -m "${3:-"$result_directory-comparison"}")
for output_directory in "$result_directory" "$comparison_directory"; do
    if [[ -e "$output_directory" ]]; then
        printf 'Refusing to overwrite existing output: %s\n' "$output_directory" >&2
        exit 1
    fi
done
for required_file in requests.json validation.json artifacts.sha256; do
    if [[ ! -f "$baseline_directory/$required_file" ]]; then
        printf 'Missing validated baseline artifact: %s/%s\n' "$baseline_directory" "$required_file" >&2
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLEXKV_ENABLE_MPS=1
export PYTHONPATH="${SGLANG_DIRECTORY:-/root/sglang}/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$result_directory/code"
cp "$baseline_directory/requests.json" "$result_directory/requests.json"
cp "$script_directory"/*.py "$script_directory"/*.sh \
    "$script_directory/flexkv_host.yaml" "$script_directory/compatibility.patch" \
    "$script_directory/THIRD_EXPERIMENT.md" "$result_directory/code/"
if printf 'get_server_list\n' | nvidia-cuda-mps-control > "$result_directory/mps_before.txt" 2>&1; then
    printf 'Using an existing MPS daemon.\n' | tee "$result_directory/mps_ownership.txt"
else
    nvidia-cuda-mps-control -d
    printf 'Started MPS at %s; shared daemon is not stopped automatically.\n' \
        "$(date -u --iso-8601=seconds)" | tee "$result_directory/mps_ownership.txt"
fi
"$python_binary" "$script_directory/collect_environment.py" "$result_directory/environment"

server_started=false
cleanup() {
    if [[ "$server_started" == true ]]; then
        "$python_binary" "$script_directory/manage_server.py" stop
    fi
}
trap cleanup EXIT
for block in 01_bounded_on_flexkv 02_bounded_on_flexkv; do
    block_directory="$result_directory/$block"
    "$python_binary" "$script_directory/manage_server.py" start bounded_on_flexkv "$block_directory"
    server_started=true
    "$python_binary" "$script_directory/manage_server.py" wait --timeout 1800
    nvidia-smi -q > "$block_directory/gpu_before.txt"
    "$python_binary" "$script_directory/benchmark_host.py" \
        --backend bounded_on_flexkv --output "$block_directory" \
        --requests "$result_directory/requests.json" \
        --base-url "http://127.0.0.1:${SERVER_PORT:-30180}" \
        --warmup-pairs 3 --pairs 10 \
        2>&1 | tee "$block_directory/benchmark.log"
    nvidia-smi -q > "$block_directory/gpu_after.txt"
    "$python_binary" "$script_directory/manage_server.py" stop
    server_started=false
done
trap - EXIT
"$python_binary" "$script_directory/collect_environment.py" "$result_directory/environment_after"
"$python_binary" "$script_directory/summarize_three_groups.py" \
    --baseline "$baseline_directory" --third "$result_directory" \
    --output "$comparison_directory"
printf 'Experiment servers stopped; leave any shared MPS daemon to its owner.\n'
