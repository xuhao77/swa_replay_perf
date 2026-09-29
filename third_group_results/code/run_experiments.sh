#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_binary=${PYTHON_BINARY:-/root/nvfp4-validation/venv/bin/python}
result_directory=${1:-"$script_directory/repro-$(date -u +%Y%m%dT%H%M%SZ)"}

if [[ -e "$result_directory" ]]; then
    printf 'Refusing to overwrite existing result directory: %s\n' "$result_directory" >&2
    exit 1
fi
mkdir -p "$result_directory"
if [[ ! -f "$script_directory/requests.json" ]]; then
    "$python_binary" "$script_directory/prepare_requests.py" \
        --model-directory "${MODEL_DIRECTORY:-/root/models/DeepSeek-V4.1-Flash}" \
        --sglang-directory "${SGLANG_DIRECTORY:-/root/sglang}"
fi
cp "$script_directory/requests.json" "$result_directory/requests.json"
"$python_binary" "$script_directory/collect_environment.py" "$result_directory/environment"

server_started=false
cleanup() {
    if [[ "$server_started" == true ]]; then
        "$python_binary" "$script_directory/manage_server.py" stop
    fi
}
trap cleanup EXIT

for block in 01_bounded_on 02_bounded_off 03_bounded_off 04_bounded_on; do
    mode=${block#*_}
    block_directory="$result_directory/$block"
    "$python_binary" "$script_directory/manage_server.py" start "$mode" "$block_directory"
    server_started=true
    "$python_binary" "$script_directory/manage_server.py" wait --timeout 1800
    nvidia-smi -q > "$block_directory/gpu_before.txt"
    "$python_binary" "$script_directory/benchmark.py" \
        --mode "$mode" --output "$block_directory" \
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
"$python_binary" "$script_directory/validate_artifacts.py" "$result_directory"
"$python_binary" "$script_directory/summarize.py" "$result_directory"
