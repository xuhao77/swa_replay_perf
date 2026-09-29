import json
import os
import sys
from pathlib import Path

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


original_pool_init = DeepSeekV4TokenToKVPool.__init__


def describe_kv_pool(pool):
    return {
        "layout": pool.kv_layout.value,
        "bytes_per_token": pool.get_bytes_per_token(),
        "layers": pool.layer_num,
        "size_tokens": pool.size,
        "page_size": pool.page_size,
        "dtype": str(pool.dtype),
        "storage_dtype": str(pool.store_dtype),
    }


def audited_pool_init(pool, *arguments, **keywords):
    original_pool_init(pool, *arguments, **keywords)
    audit_directory = Path(os.environ["SWA_PERF_POOL_AUDIT_DIRECTORY"])
    audit_directory.mkdir(parents=True, exist_ok=True)
    request_window = pool.request_window
    swa_pool = request_window.state if request_window else pool.swa_kv_pool
    audit = {
        "pid": os.getpid(),
        "device": str(swa_pool.kv_buffer[0].device),
        "encoder_bounded_replay": request_window is not None,
        "full_size": pool.full_size,
        "logical_page_size": pool.page_size,
        "sliding_window": pool.sliding_window,
        "main_kv": {
            str(ratio): describe_kv_pool(kv_pool)
            for ratio, kv_pool in pool.kv_pools.items()
        },
        "indexer_k": {
            str(ratio): {
                "fp4": index_pool.use_fp4_indexer,
                "bytes_per_token": index_pool.get_bytes_per_token(),
                "layers": index_pool.layer_num,
                "size_tokens": index_pool.size,
                "page_size": index_pool.page_size,
            }
            for ratio, index_pool in pool.index_pools.items()
        },
        "swa_kv": describe_kv_pool(swa_pool),
        "swa_storage": "request_window" if request_window else "paged_cache",
        "request_window_capacity": request_window.capacity if request_window else None,
    }
    expected_replay = os.environ["SWA_PERF_EXPECT_ENCODER_REPLAY"] == "bounded_on"
    assert audit["encoder_bounded_replay"] == expected_replay, audit
    assert set(audit["main_kv"]) == {"1", "2"}, audit
    assert all(item["layout"] == "v41_fp4" for item in audit["main_kv"].values()), audit
    assert set(audit["indexer_k"]) == {"1", "2"}, audit
    assert all(item["fp4"] for item in audit["indexer_k"].values()), audit
    assert audit["swa_kv"]["layout"] == "v41", audit
    assert audit["swa_kv"]["dtype"] == "torch.float8_e4m3fn", audit
    destination = audit_directory / f"pool-{os.getpid()}.json"
    destination.write_text(json.dumps(audit, indent=2) + "\n")
    print("SWA_PERF_POOL_AUDIT " + json.dumps(audit), flush=True)


DeepSeekV4TokenToKVPool.__init__ = audited_pool_init

if os.environ.get("SWA_PERF_HOST_BACKEND"):
    from host_control import install_host_controls

    install_host_controls()


def main():
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    arguments = prepare_server_args(sys.argv[1:])
    try:
        run_server(arguments)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
