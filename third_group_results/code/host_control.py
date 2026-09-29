import dataclasses
import json
import os
import time
from pathlib import Path

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.runtime_context import get_context, get_schedule


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def host_pool_state(cache):
    host_group = getattr(cache, "host_pool_group", None)
    if host_group is None:
        host_group = getattr(cache, "token_to_kv_pool_host", None)
    pools = {}
    for entry in getattr(host_group, "entries", []):
        host_pool = entry.host_pool
        buffers = getattr(host_pool, "kv_buffer", None)
        tensors = buffers if isinstance(buffers, (list, tuple)) else [buffers]
        tensors = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
        pools[entry.name.value] = {
            "class": type(host_pool).__name__,
            "size": host_pool.size,
            "available": int(host_pool.available_size()),
            "page_size": host_pool.page_size,
            "buffers": [
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "pinned": tensor.is_pinned(),
                }
                for tensor in tensors[:1]
            ],
        }
    return pools


def device_cache_state(scheduler):
    cache = scheduler.tree_cache
    allocator = scheduler.token_to_kv_pool_allocator
    state = {
        "full_evictable": int(cache.full_evictable_size()),
        "full_protected": int(cache.full_protected_size()),
        "swa_evictable": int(cache.swa_evictable_size()),
        "swa_protected": int(cache.swa_protected_size()),
    }
    for name in ("available_size", "full_available_size", "swa_available_size"):
        method = getattr(allocator, name, None)
        if callable(method):
            state[name] = int(method())
    return state


def global_max(scheduler, value):
    counter = torch.tensor(int(value), dtype=torch.int64, device="cpu")
    torch.distributed.all_reduce(
        counter, op=torch.distributed.ReduceOp.MAX, group=scheduler.tp_group.cpu_group
    )
    return int(counter.item())


def drain_host_io(scheduler):
    cache = scheduler.tree_cache
    deadline = time.monotonic() + 90
    while True:
        torch.cuda.synchronize()
        cache.check_hicache_events()
        pending = sum(
            len(getattr(cache, name, {}))
            for name in (
                "ongoing_write_through",
                "ongoing_load_back",
                "_inflight_store_nodes",
                "_pending_store_launches",
                "_pending_store_copies",
            )
        )
        if global_max(scheduler, pending) == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Host cache transfers did not drain")
        time.sleep(0.01)


def control(scheduler, action, trial_directory):
    run_directory = Path(os.environ["SWA_PERF_RUN_DIRECTORY"]).resolve()
    destination = Path(trial_directory).resolve()
    if not destination.is_relative_to(run_directory):
        raise ValueError("Control audit must remain inside this server's run directory")
    if action not in {"cold_phase", "evict_for_replay", "drain"}:
        raise ValueError(action)
    drain_host_io(scheduler)
    if global_max(scheduler, not scheduler.is_fully_idle()):
        raise RuntimeError("Experiment cache control requires an idle scheduler")
    audit = {
        "action": action,
        "tp_rank": scheduler._swa_perf_tp_rank,
        "backend": os.environ["SWA_PERF_HOST_BACKEND"],
        "device_before": device_cache_state(scheduler),
        "host_before": host_pool_state(scheduler.tree_cache),
    }
    if action == "cold_phase":
        get_context().override(
            source="swa_replay_perf.cold_phase",
            chunked_prefill_size=7936,
            prefill_max_requests=1,
        )
        scheduler.chunked_prefill_size = 7936
    elif action == "evict_for_replay":
        eviction = scheduler.tree_cache.evict(
            EvictParams(
                num_tokens=scheduler.max_total_num_tokens * 2,
                swa_num_tokens=scheduler.max_total_num_tokens * 2,
            )
        )
        audit["eviction"] = dataclasses.asdict(eviction)
        get_context().override(
            source="swa_replay_perf.replay_phase",
            chunked_prefill_size=8192,
            prefill_max_requests=8,
        )
        scheduler.chunked_prefill_size = 8192
    torch.cuda.synchronize()
    audit["device_after"] = device_cache_state(scheduler)
    audit["host_after"] = host_pool_state(scheduler.tree_cache)
    audit["chunked_prefill_size"] = scheduler.chunked_prefill_size
    audit["prefill_max_requests"] = get_schedule().prefill_max_requests
    save_json(destination / f"{action}.rank_{scheduler._swa_perf_tp_rank}.json", audit)
    if action == "evict_for_replay":
        assert audit["device_after"]["full_evictable"] == 0, audit
        assert audit["device_after"]["full_protected"] == 0, audit
        assert audit["device_after"]["swa_evictable"] == 0, audit
        assert audit["device_after"]["swa_protected"] == 0, audit
        assert audit["host_before"] == audit["host_after"], audit


def install_host_controls():
    original_init = Scheduler.__init__

    def audited_scheduler_init(
        scheduler, server_args, port_args, tp_rank, pp_rank, dp_rank
    ):
        original_init(scheduler, server_args, port_args, tp_rank, pp_rank, dp_rank)
        scheduler._swa_perf_tp_rank = tp_rank
        run_directory = Path(os.environ["SWA_PERF_RUN_DIRECTORY"])
        cache = scheduler.tree_cache
        audit = {
            "tp_rank": scheduler._swa_perf_tp_rank,
            "pid": os.getpid(),
            "backend": os.environ["SWA_PERF_HOST_BACKEND"],
            "cache_class": type(cache).__name__,
            "host_pools": host_pool_state(cache),
            "initial_device_state": device_cache_state(scheduler),
            "flexkv_layerwise": os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER"),
            "flexkv_swa_grid_pages": os.environ.get("SGLANG_FLEXKV_SWA_GRID_PAGES"),
        }
        connector = getattr(cache, "flexkv_connector", None)
        if connector is not None:
            audit["flexkv_groups"] = [
                {
                    "name": group["name"],
                    "compress_ratio": group.get("compress_ratio"),
                    "buffer_dtype": str(group["buffers"][0].dtype),
                    "buffer_shape": list(group["buffers"][0].shape),
                }
                for group in connector._dsv4_layer_groups
            ]
            audit["flexkv_cache_config"] = dataclasses.asdict(connector.cache_config)
            audit["flexkv_swa_pool"] = connector._swa_kv_pool is not None
        save_json(
            run_directory
            / "host_backend_audit"
            / f"rank_{scheduler._swa_perf_tp_rank}.json",
            audit,
        )
        if scheduler._swa_perf_tp_rank == 0:
            save_json(
                run_directory / "scheduler_control.json",
                {"rpc_ipc_name": port_args.rpc_ipc_name},
            )

    Scheduler.__init__ = audited_scheduler_init
    Scheduler.swa_perf_control = control
