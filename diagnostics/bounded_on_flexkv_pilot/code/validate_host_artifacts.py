import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmark import parse_prefill_log, read_counters, write_json
from benchmark_host import hicache_loads_by_rank, parse_flexkv_operations


def load_json(path):
    return json.loads(path.read_text())


def validate_host_results(directory):
    blocks = sorted(directory.glob("*/results.json"))
    expected_backends = [
        "bounded_on_l2",
        "bounded_off_host",
        "bounded_off_host",
        "bounded_on_l2",
    ]
    assert len(blocks) == 4
    dataset = load_json(directory / "requests.json")
    dataset_hash = hashlib.sha256(
        (directory / "requests.json").read_bytes()
    ).hexdigest()
    all_outputs = set()
    stage_count = measured_count = request_count = 0
    per_backend_measured = dict.fromkeys(expected_backends, 0)
    for block_path, backend in zip(blocks, expected_backends):
        block = load_json(block_path)
        block_directory = block_path.parent
        assert (
            "nvidia-cuda-mps-server" in (block_directory / "gpu_before.txt").read_text()
        )
        assert block["complete"] and block["protocol"] == "host_l2_v1"
        assert block["backend"] == backend
        assert block["dataset_sha256"] == dataset_hash
        assert block["batch_size"] == 8 and block["input_tokens_per_request"] == 8192
        assert block["output_tokens_per_request"] == 2
        assert block["warmup_pairs"] == 3 and block["measured_pairs"] == 10
        assert len(block["trials"]) == 13
        bounded = backend == "bounded_on_l2"
        configuration = load_json(block_directory / "server_info.json")
        configuration = configuration.get("server_args", configuration)
        assert configuration["tp_size"] == configuration["ep_size"] == 8
        assert configuration["dp_size"] == configuration["attn_cp_size"] == 1
        assert configuration["enable_encoder_swa_bounded_replay"] == bounded
        assert configuration["enable_hierarchical_cache"] == bounded
        assert configuration["enable_flexkv"] == (not bounded)
        assert not configuration["enable_decoder_swa_bounded_replay"]
        assert configuration["hicache_storage_backend"] is None
        assert configuration["swa_prefix_tails"] is None
        assert not configuration.get("_swa_full_tokens_ratio_explicitly_set", False)
        assert configuration["chunked_prefill_size"] == 8192
        assert configuration["max_running_requests"] == 8
        assert configuration["max_total_tokens"] == 131072
        audits = [
            load_json(path) for path in (block_directory / "pool_audit").glob("*.json")
        ]
        assert len(audits) == 8
        for audit in audits:
            assert audit["encoder_bounded_replay"] == bounded
            assert all(
                pool["layout"] == "v41_fp4" and pool["bytes_per_token"] == 288
                for pool in audit["main_kv"].values()
            )
            assert all(
                pool["fp4"] and pool["bytes_per_token"] == 68
                for pool in audit["indexer_k"].values()
            )
            assert audit["swa_kv"]["layout"] == "v41"
            assert audit["swa_kv"]["bytes_per_token"] == 528
            if not bounded:
                assert audit["swa_kv"]["size_tokens"] == 35072
        audit_pids = {audit["pid"] for audit in audits}
        placements = {
            int(row[0].strip()): row[1].strip()
            for row in csv.reader(
                (block_directory / "gpu_process_placement.csv").read_text().splitlines()
            )
            if int(row[0].strip()) in audit_pids
        }
        assert set(placements) == audit_pids
        assert len(set(placements.values())) == 8
        host_audits = [
            load_json(path)
            for path in (block_directory / "host_backend_audit").glob("*.json")
        ]
        assert len(host_audits) == 8
        for audit in host_audits:
            assert audit["backend"] == backend
            if bounded:
                names = set(audit["host_pools"])
                for ratio in ("c1", "c2"):
                    assert any(
                        ratio in name and "indexer" not in name for name in names
                    )
                    assert any(ratio in name and "indexer" in name for name in names)
                assert not any("swa" in name for name in names)
                for pool in audit["host_pools"].values():
                    for buffer in pool["buffers"]:
                        assert buffer["device"] == "cpu" and buffer["pinned"]
            else:
                assert {group["name"] for group in audit["flexkv_groups"]} == {
                    "c1",
                    "c2",
                    "c1_indexer",
                    "c2_indexer",
                }
                assert audit["flexkv_swa_pool"]
                assert audit["flexkv_layerwise"] == "0"
                assert audit["flexkv_cache_config"]["swa"]["enabled"]
                assert audit["flexkv_cache_config"]["swa"]["pin_memory"]
                assert audit["flexkv_cache_config"]["swa"]["num_ssd_slots"] == 0
                assert audit["flexkv_cache_config"]["swa"]["num_remote_slots"] == 0
        salts_seen = set()
        for trial in block["trials"]:
            trial_directory = block_directory / trial["name"]
            salts = load_json(trial_directory / "cache_salts.json")
            assert len(set(salts)) == 8 and not (set(salts) & salts_seen)
            salts_seen.update(salts)
            measured_count += int(trial["measured"])
            per_backend_measured[backend] += int(trial["measured"])
            for action in ("cold_phase", "evict_for_replay", "drain"):
                controls = [
                    load_json(path)
                    for path in trial_directory.glob(f"{action}.rank_*.json")
                ]
                assert len(controls) == 8
                for control in controls:
                    assert control["action"] == action and control["backend"] == backend
                    if action == "cold_phase":
                        assert control["chunked_prefill_size"] == 7936
                        assert control["prefill_max_requests"] == 1
                    else:
                        assert control["chunked_prefill_size"] == 8192
                        assert control["prefill_max_requests"] == 8
                    if action == "evict_for_replay":
                        assert all(
                            control["device_after"][key] == 0
                            for key in (
                                "full_evictable",
                                "full_protected",
                                "swa_evictable",
                                "swa_protected",
                            )
                        )
                        assert control["host_before"] == control["host_after"]
            payloads = {}
            stage_outputs = {}
            for stage in ("cold", "replay"):
                stage_count += 1
                summary = load_json(trial_directory / f"{stage}.summary.json")
                assert summary == trial[stage]
                payload_bytes = (trial_directory / f"{stage}.request.json").read_bytes()
                assert (
                    hashlib.sha256(payload_bytes).hexdigest()
                    == summary["request_payload_sha256"]
                )
                payload = json.loads(payload_bytes)
                assert payload["input_ids"] == [
                    request["input_ids"] for request in dataset["requests"]
                ]
                assert payload["cache_salt"] == salts
                assert payload["sampling_params"] == {
                    "temperature": 0,
                    "max_new_tokens": 2,
                    "ignore_eos": True,
                }
                assert payload["stream"]
                assert payload.pop("rid") == summary["request_ids"]
                payloads[stage] = payload
                responses = load_json(trial_directory / f"{stage}.response.json")
                assert len(responses) == 8
                request_count += len(responses)
                cached = 0 if stage == "cold" else 7936
                replay_tokens = 1024 if stage == "replay" and bounded else 0
                for request, response in zip(dataset["requests"], responses):
                    metadata = response["meta_info"]
                    assert metadata["prompt_tokens"] == 8192
                    assert metadata["cached_tokens"] == cached
                    assert metadata["completion_tokens"] == 2
                    assert metadata.get("num_retractions", 0) == 0
                    details = metadata.get("cached_tokens_details")
                    if stage == "replay":
                        assert details["device"] == 0 and details["host"] == cached
                    else:
                        assert details is None or all(
                            details.get(tier, 0) == 0
                            for tier in ("device", "host", "storage")
                        )
                    assert len(response["output_ids"]) == 2
                    assert response["text"].strip() == request["expected_answer"]
                stage_outputs[stage] = [item["output_ids"] for item in responses]
                streamed = {}
                for event in (
                    (trial_directory / f"{stage}.response.sse")
                    .read_bytes()
                    .split(b"\n\n")
                ):
                    if event.startswith(b"data: ") and event != b"data: [DONE]":
                        response = json.loads(event[6:])
                        streamed[response["meta_info"]["id"]] = response
                assert streamed == {item["meta_info"]["id"]: item for item in responses}
                events = load_json(trial_directory / f"{stage}.stream_events.json")
                first_events = {}
                for event in events:
                    first_events.setdefault(event["request_id"], event)
                assert len(first_events) == 8
                assert all(
                    event["completion_tokens"] == 1 for event in first_events.values()
                )
                first_token_duration = max(
                    event["received_seconds"] for event in first_events.values()
                )
                assert (
                    first_token_duration == summary["client_batch_first_token_seconds"]
                )
                duration = summary["client_batch_latency_seconds"]
                assert 0 < first_token_duration < duration
                assert (
                    abs(summary["logical_input_tokens_per_second"] * duration - 65536)
                    < 1e-6
                )
                assert (
                    abs(
                        summary["prefill_logical_input_tokens_per_second"]
                        * first_token_duration
                        - 65536
                    )
                    < 1e-6
                )
                before = read_counters(
                    (trial_directory / f"{stage}.metrics_before.txt").read_text()
                )
                after = read_counters(
                    (trial_directory / f"{stage}.metrics_after.txt").read_text()
                )
                delta = {key: after[key] - before[key] for key in before}
                assert delta == summary["counter_delta"]
                assert delta["prefill_cache"] == cached * 8
                assert delta["prefill_compute"] == (8192 - cached) * 8 + replay_tokens
                logs = parse_prefill_log(
                    (trial_directory / f"{stage}.server.log").read_text()
                )
                assert logs == summary["prefill_log_records"]
                if stage == "replay":
                    assert len(logs) == 1 and logs[0]["new-seq"] == 8
                    assert logs[0]["cached-device"] == logs[0]["cached-storage"] == 0
                    assert logs[0]["cached-host"] == 63488
                    assert logs[0]["new-token"] == 2048
                    assert logs[0]["replay-token"] == replay_tokens
                    if not bounded:
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
                        assert set(summary["request_ids"]) <= restored
                        launches = {
                            operation["sglang_req_id"]: operation
                            for operation in operations
                            if operation.get("direction") == "H2D"
                            and operation.get("act") == "launch"
                        }
                        for request_id in summary["request_ids"]:
                            assert launches[request_id]["slots"] == "7936"
                            assert launches[request_id]["swa_slots"] == "256"
                            assert launches[request_id]["mode"] == "no-layerwise"
                    else:
                        before_loads = hicache_loads_by_rank(
                            (trial_directory / "replay.metrics_before.txt").read_text()
                        )
                        after_loads = hicache_loads_by_rank(
                            (trial_directory / "replay.metrics_after.txt").read_text()
                        )
                        restored = {
                            rank: value - before_loads.get(rank, 0)
                            for rank, value in after_loads.items()
                        }
                        assert restored == {str(rank): 63488 for rank in range(8)}
                        assert restored == load_json(
                            trial_directory / "hicache_restore_counters.json"
                        )
            assert payloads["cold"] == payloads["replay"]
            assert stage_outputs["cold"] == stage_outputs["replay"]
            assert trial["cold_replay_output_ids_equal"]
            all_outputs.add(json.dumps(stage_outputs["replay"]))
    assert measured_count == 40 and set(per_backend_measured.values()) == {20}
    assert len(all_outputs) == 1
    verdict = {
        "passed": True,
        "protocol": "host_l2_v1",
        "server_blocks": 4,
        "stages_including_warmup": stage_count,
        "requests_including_cold_and_warmup": request_count,
        "measured_replay_batches": measured_count,
        "measured_batches_per_backend": per_backend_measured,
        "dataset_sha256": dataset_hash,
        "all_device_hits_zero": True,
        "host_hits_per_replay_request": 7936,
        "all_cold_replay_and_cross_backend_output_ids_equal": True,
        "paged_swa_capacity_not_enlarged": True,
        "mps_available_before_all_four_blocks": True,
    }
    write_json(directory / "validation.json", verdict)
    print(json.dumps(verdict, indent=2))
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    validate_host_results(parser.parse_args().directory)


if __name__ == "__main__":
    main()
