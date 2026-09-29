import argparse
import copy
import hashlib
import json
import shlex
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import zmq
from prometheus_client.parser import text_string_to_metric_families

from benchmark import checked_server_info, generate_batch, write_json


def hicache_loads_by_rank(metrics_text):
    return {
        sample.labels["tp_rank"]: sample.value
        for family in text_string_to_metric_families(metrics_text)
        for sample in family.samples
        if sample.name == "sglang:load_back_tokens_total"
        and sample.labels.get("pool") == "kv"
    }


def parse_flexkv_operations(text):
    operations = []
    marker = "[FlexKV-SGLang] "
    for line in text.splitlines():
        if marker not in line:
            continue
        fields = {}
        for field in shlex.split(line.split(marker, 1)[1]):
            if "=" in field:
                name, value = field.split("=", 1)
                fields[name] = value
        operations.append(fields)
    return operations


class CacheControl:
    def __init__(self, output_directory):
        from sglang.srt.utils.network import get_zmq_socket

        address = json.loads((output_directory / "scheduler_control.json").read_text())[
            "rpc_ipc_name"
        ]
        self.context = zmq.Context()
        self.socket = get_zmq_socket(self.context, zmq.DEALER, address, True)
        self.socket.setsockopt(zmq.RCVTIMEO, 120000)
        self.socket.setsockopt(zmq.SNDTIMEO, 120000)

    def call(self, action, trial_directory):
        from sglang.srt.managers.io_struct import (
            RpcReqInput,
            RpcReqOutput,
            sock_recv,
            sock_send,
        )

        for attempt in range(100):
            sock_send(
                self.socket,
                RpcReqInput(
                    method="swa_perf_control",
                    parameters={
                        "action": action,
                        "trial_directory": str(trial_directory.resolve()),
                    },
                ),
            )
            response = sock_recv(self.socket)
            assert isinstance(response, RpcReqOutput), response
            if response.success:
                paths = sorted(trial_directory.glob(f"{action}.rank_*.json"))
                assert len(paths) == 8, paths
                return [json.loads(path.read_text()) for path in paths]
            if "requires an idle scheduler" not in response.message:
                raise RuntimeError(response.message)
            time.sleep(0.1)
        raise TimeoutError("Cache control could not obtain an idle scheduler")

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


def check_host_configuration(configuration, arguments):
    expected_hicache = arguments.backend == "bounded_on_l2"
    expected_swa_pool = arguments.backend == "bounded_off_host"
    assert configuration["enable_hierarchical_cache"] == expected_hicache
    assert configuration["enable_flexkv"] == (not expected_hicache)
    assert configuration["hicache_storage_backend"] is None
    assert configuration["cpu_offload_gb"] == 0
    audits = [
        json.loads(path.read_text())
        for path in (arguments.output / "host_backend_audit").glob("*.json")
    ]
    assert len(audits) == 8
    for audit in audits:
        assert audit["backend"] == arguments.backend
        if expected_hicache:
            names = set(audit["host_pools"])
            assert any("c1" in name and "indexer" not in name for name in names), names
            assert any("c2" in name and "indexer" not in name for name in names), names
            assert any("c1" in name and "indexer" in name for name in names), names
            assert any("c2" in name and "indexer" in name for name in names), names
            assert not any("swa" in name for name in names), names
        else:
            assert {item["name"] for item in audit["flexkv_groups"]} == {
                "c1",
                "c2",
                "c1_indexer",
                "c2_indexer",
            }
            assert audit["flexkv_swa_pool"] == expected_swa_pool
            assert audit["flexkv_layerwise"] == "0"
            assert audit["flexkv_swa_grid_pages"] == "31"
            cache_config = audit["flexkv_cache_config"]
            assert cache_config["enable_cpu"]
            assert cache_config["_user_cpu_cache_gb"] == 16
            assert not cache_config["enable_ssd"]
            assert not cache_config["enable_remote"]
            assert cache_config["enable_swa_transfer"] == expected_swa_pool
            if not expected_swa_pool:
                assert cache_config["swa"] is None
                assert audit["cache_class"] == "FlexKVRadixCache"
            assert all(
                group["buffer_dtype"] == "torch.uint8"
                for group in audit["flexkv_groups"]
            )


def benchmark_host(arguments):
    arguments.output = arguments.output.resolve()
    arguments.mode = (
        "bounded_off" if arguments.backend == "bounded_off_host" else "bounded_on"
    )
    dataset = json.loads(arguments.requests.read_text())
    assert dataset["batch_size"] == 8
    assert all(len(item["input_ids"]) == 8192 for item in dataset["requests"])
    result_path = arguments.output / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    experiment = {
        "protocol": "host_l2_v1",
        "mode": arguments.mode,
        "backend": arguments.backend,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(arguments.requests.read_bytes()).hexdigest(),
        "batch_size": 8,
        "input_tokens_per_request": 8192,
        "output_tokens_per_request": 2,
        "warmup_pairs": arguments.warmup_pairs,
        "measured_pairs": arguments.pairs,
        "trials": [],
        "complete": False,
    }
    control = CacheControl(arguments.output)
    try:
        with requests.Session() as session:
            session.trust_env = False
            configuration = checked_server_info(
                session, arguments.base_url, arguments.mode, arguments.output
            )
            check_host_configuration(configuration, arguments)
            arguments.page_size = configuration["page_size"]
            experiment["page_size"] = arguments.page_size
            for trial_index in range(arguments.warmup_pairs + arguments.pairs):
                measured = trial_index >= arguments.warmup_pairs
                trial_name = (
                    f"measure_{trial_index - arguments.warmup_pairs:02d}"
                    if measured
                    else f"warmup_{trial_index:02d}"
                )
                trial_directory = arguments.output / trial_name
                trial_directory.mkdir()
                trial_dataset = copy.deepcopy(dataset)
                for item in trial_dataset["requests"]:
                    item["cache_salt"] += (
                        f":host:{arguments.output.parent.name}:{trial_name}"
                    )
                write_json(
                    trial_directory / "cache_salts.json",
                    [item["cache_salt"] for item in trial_dataset["requests"]],
                )
                response = session.post(
                    arguments.base_url + "/flush_cache?timeout=60", timeout=70
                )
                (trial_directory / "flush_response.txt").write_text(response.text)
                response.raise_for_status()
                control.call("cold_phase", trial_directory)
                cold = generate_batch(
                    session, arguments, trial_dataset, trial_directory, "cold"
                )
                eviction = control.call("evict_for_replay", trial_directory)
                for audit in eviction:
                    assert audit["chunked_prefill_size"] == 8192
                    assert audit["prefill_max_requests"] == 8
                replay = generate_batch(
                    session, arguments, trial_dataset, trial_directory, "replay"
                )
                control.call("drain", trial_directory)
                trial = {
                    "name": trial_name,
                    "measured": measured,
                    "cold": cold,
                    "replay": replay,
                    "cold_replay_output_ids_equal": cold["output_ids"]
                    == replay["output_ids"],
                    "cold_replay_text_equal": cold["output_text"]
                    == replay["output_text"],
                }
                assert trial["cold_replay_output_ids_equal"], trial
                assert all(cold["semantic_correct"]) and all(replay["semantic_correct"])
                if arguments.backend != "bounded_on_l2":
                    operations = parse_flexkv_operations(
                        (trial_directory / "replay.server.log").read_text()
                    )
                    restored = {
                        operation["sglang_req_id"]
                        for operation in operations
                        if operation.get("direction") == "H2D"
                        and operation.get("act") == "complete"
                        and operation.get("status") == "success"
                    }
                    assert set(replay["request_ids"]) <= restored, operations
                    launches = {
                        operation["sglang_req_id"]: operation
                        for operation in operations
                        if operation.get("direction") == "H2D"
                        and operation.get("act") == "launch"
                    }
                    for request_id in replay["request_ids"]:
                        assert launches[request_id]["slots"] == "7936", launches
                        expected_swa_slots = (
                            "256" if arguments.backend == "bounded_off_host" else "0"
                        )
                        assert (
                            launches[request_id]["swa_slots"] == expected_swa_slots
                        ), launches
                        assert launches[request_id]["mode"] == "no-layerwise", launches
                    write_json(
                        trial_directory / "flexkv_restore_operations.json", operations
                    )
                else:
                    before = hicache_loads_by_rank(
                        (trial_directory / "replay.metrics_before.txt").read_text()
                    )
                    after = hicache_loads_by_rank(
                        (trial_directory / "replay.metrics_after.txt").read_text()
                    )
                    restored = {
                        rank: value - before.get(rank, 0)
                        for rank, value in after.items()
                    }
                    assert restored == {str(rank): 63488 for rank in range(8)}, restored
                    write_json(
                        trial_directory / "hicache_restore_counters.json", restored
                    )
                experiment["trials"].append(trial)
                write_json(result_path, experiment)
                print(
                    f"{arguments.backend} {trial_name}: "
                    f"first_token={replay['client_batch_first_token_seconds']:.6f}s "
                    f"complete={replay['client_batch_latency_seconds']:.6f}s "
                    f"host={replay['cached_tokens']} "
                    f"replay_tokens={replay['replay_tokens_inferred']:.0f}",
                    flush=True,
                )
    finally:
        control.close()
    measured = [trial for trial in experiment["trials"] if trial["measured"]]
    durations = [trial["replay"]["client_batch_latency_seconds"] for trial in measured]
    first_token_durations = [
        trial["replay"]["client_batch_first_token_seconds"] for trial in measured
    ]
    experiment["summary"] = {
        "measured_pairs": len(measured),
        "mean_replay_seconds": statistics.mean(durations),
        "mean_batch_first_token_seconds": statistics.mean(first_token_durations),
        "aggregate_logical_input_tokens_per_second": len(measured)
        * 65536
        / sum(durations),
        "aggregate_prefill_logical_input_tokens_per_second": len(measured)
        * 65536
        / sum(first_token_durations),
    }
    experiment["complete"] = True
    experiment["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(result_path, experiment)
    print(json.dumps(experiment["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("bounded_on_l2", "bounded_off_host", "bounded_on_flexkv"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--requests", type=Path, default=Path(__file__).with_name("requests.json")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30180")
    parser.add_argument("--warmup-pairs", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=10)
    benchmark_host(parser.parse_args())


if __name__ == "__main__":
    main()
