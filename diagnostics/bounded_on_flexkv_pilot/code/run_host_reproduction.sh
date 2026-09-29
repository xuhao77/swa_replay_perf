#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
result_directory=${1:-"$script_directory/host-repro-$(date -u +%Y%m%dT%H%M%SZ)"}
if [[ -e "$result_directory" ]]; then
    printf 'Refusing to overwrite existing result directory: %s\n' "$result_directory" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLEXKV_ENABLE_MPS=1
if printf 'get_server_list\n' | nvidia-cuda-mps-control >/dev/null 2>&1; then
    printf 'CUDA MPS is already available before the first group.\n'
else
    nvidia-cuda-mps-control -d
    printf 'Started CUDA MPS before the first group.\n'
fi
printf 'MPS is shared infrastructure; stop it manually only when no clients remain.\n'
exec bash "$script_directory/run_host_experiments.sh" "$result_directory"
