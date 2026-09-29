#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_binary=${PYTHON_BINARY:-/root/nvfp4-validation/venv/bin/python}
result_directory=${1:-"$script_directory/host-repro-$(date -u +%Y%m%dT%H%M%SZ)"}
export PYTHONPATH="${SGLANG_DIRECTORY:-/root/sglang}/python${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$result_directory" ]]; then
    printf 'Refusing to overwrite existing result directory: %s\n' "$result_directory" >&2
    exit 1
fi
mkdir -p "$result_directory/code"
cp "$script_directory/requests.json" "$result_directory/requests.json"
cp "$script_directory"/*.py "$script_directory"/*.sh \
    "$script_directory/flexkv_host.yaml" "$script_directory/compatibility.patch" \
    "$script_directory/HOST_EXPERIMENT.md" "$result_directory/code/"
"$python_binary" "$script_directory/collect_environment.py" "$result_directory/environment"

server_started=false
cleanup() {
    if [[ "$server_started" == true ]]; then
        "$python_binary" "$script_directory/manage_server.py" stop
    fi
}
trap cleanup EXIT

for block in 01_bounded_on_l2 02_bounded_off_host 03_bounded_off_host 04_bounded_on_l2; do
    backend=${block#*_}
    block_directory="$result_directory/$block"
    "$python_binary" "$script_directory/manage_server.py" start "$backend" "$block_directory"
    server_started=true
    "$python_binary" "$script_directory/manage_server.py" wait --timeout 1800
    nvidia-smi -q > "$block_directory/gpu_before.txt"
    "$python_binary" "$script_directory/benchmark_host.py" \
        --backend "$backend" --output "$block_directory" \
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
"$python_binary" "$script_directory/validate_host_artifacts.py" "$result_directory"
"$python_binary" "$script_directory/summarize_host.py" "$result_directory"
