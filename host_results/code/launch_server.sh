#!/usr/bin/env bash
set -euo pipefail

mode=${1:?Usage: launch_server.sh MODE RUN_DIRECTORY}
run_directory=${2:?A run directory is required}
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_binary=${PYTHON_BINARY:-/root/nvfp4-validation/venv/bin/python}
sglang_directory=${SGLANG_DIRECTORY:-/root/sglang}
model_directory=${MODEL_DIRECTORY:-/root/models/DeepSeek-V4.1-Flash}

if [[ "$mode" != bounded_on_l2 && "$mode" != bounded_off_host ]]; then
    printf 'Invalid mode: %s\n' "$mode" >&2
    exit 2
fi

mkdir -p "$run_directory"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$sglang_directory/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DSV4_KV_LAYOUT=v41
export SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp4
export PYTHONUNBUFFERED=1
export SWA_PERF_POOL_AUDIT_DIRECTORY="$run_directory/pool_audit"
export SWA_PERF_EXPECT_ENCODER_REPLAY="$mode"
export SWA_PERF_HOST_BACKEND="$mode"
export SWA_PERF_RUN_DIRECTORY="$run_directory"
export FLEXKV_ENABLE_LAYERWISE_TRANSFER=0
export SGLANG_FLEXKV_SWA_GRID_PAGES=31
if [[ "$mode" == bounded_on_l2 ]]; then
    export SWA_PERF_EXPECT_ENCODER_REPLAY=bounded_on
fi

arguments=(
    --model-path "$model_directory"
    --trust-remote-code
    --tp-size 8
    --ep-size 8
    --kv-cache-dtype fp8_e4m3
    --mem-fraction-static 0.85
    --context-length 16384
    --max-total-tokens 131072
    --max-running-requests 8
    --chunked-prefill-size 8192
    --cuda-graph-backend-decode disabled
    --cuda-graph-backend-prefill disabled
    --random-seed 20260929
    --host 127.0.0.1
    --port "${SERVER_PORT:-30180}"
    --enable-metrics
    --skip-server-warmup
)
if [[ "$mode" == bounded_on_l2 ]]; then
    arguments+=(--enable-encoder-swa-bounded-replay)
fi
if [[ "$mode" == bounded_on_l2 ]]; then
    arguments+=(
        --enable-hierarchical-cache
        --hicache-ratio 2
        --hicache-write-policy write_through
        --hicache-io-backend kernel
        --hicache-mem-layout page_first
    )
elif [[ "$mode" == bounded_off_host ]]; then
    arguments+=(--enable-flexkv --flexkv-config-file "$script_directory/flexkv_host.yaml")
fi
printf '%q ' "$python_binary" "$script_directory/server_entry.py" "${arguments[@]}"
printf '\n'
exec "$python_binary" "$script_directory/server_entry.py" "${arguments[@]}"
