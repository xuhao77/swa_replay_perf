import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmark import parse_prefill_log, read_counters


def load_json(path):
    return json.loads(path.read_text())


def validate(results_directory):
    expected_modes = ["bounded_on", "bounded_off", "bounded_off", "bounded_on"]
    result_paths = sorted(results_directory.glob("*/results.json"))
    assert len(result_paths) == 4, result_paths
    dataset_path = results_directory / "requests.json"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    dataset = load_json(dataset_path)
    assert len(dataset["requests"]) == 8
    assert all(len(item["input_ids"]) == 8192 for item in dataset["requests"])
    checked_stages = 0
    checked_requests = 0
    checked_measured_batches = 0
    for result_path, expected_mode in zip(result_paths, expected_modes):
        block_directory = result_path.parent
        block = load_json(result_path)
        assert block["complete"], result_path
        assert block["mode"] == expected_mode, block["mode"]
        assert block["dataset_sha256"] == dataset_hash, result_path
        assert block["warmup_pairs"] == 3 and block["measured_pairs"] == 10, result_path
        assert len(block["trials"]) == 13, result_path
        assert block["output_tokens_per_request"] == 2
        configuration = load_json(block_directory / "server_info.json")
        configuration = configuration.get("server_args", configuration)
        assert configuration["tp_size"] == configuration["ep_size"] == 8
        assert configuration["dp_size"] == configuration["attn_cp_size"] == 1
        assert not configuration.get("_swa_full_tokens_ratio_explicitly_set", False)
        assert configuration.get("swa_prefix_tails") is None
        assert configuration["enable_encoder_swa_bounded_replay"] == (
            expected_mode == "bounded_on"
        )
        assert not configuration["enable_decoder_swa_bounded_replay"]
        audits = [
            load_json(path) for path in (block_directory / "pool_audit").glob("*.json")
        ]
        assert len(audits) == 8
        for audit in audits:
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
            assert audit["swa_storage"] == (
                "request_window" if expected_mode == "bounded_on" else "paged_cache"
            )
            if expected_mode == "bounded_off":
                assert audit["swa_kv"]["size_tokens"] == 35072
        placement_rows = csv.reader(
            (block_directory / "gpu_process_placement.csv").read_text().splitlines()
        )
        audit_pids = {audit["pid"] for audit in audits}
        placements = {
            int(row[0].strip()): row[1].strip()
            for row in placement_rows
            if int(row[0].strip()) in audit_pids
        }
        assert set(placements) == audit_pids
        assert len(set(placements.values())) == 8
        for trial in block["trials"]:
            trial_directory = block_directory / trial["name"]
            checked_measured_batches += int(trial["measured"])
            for stage in ("cold", "replay"):
                stage_summary = load_json(trial_directory / f"{stage}.summary.json")
                assert stage_summary == trial[stage]
                responses = load_json(trial_directory / f"{stage}.response.json")
                assert len(responses) == 8
                assert {item["meta_info"]["id"] for item in responses} == set(
                    stage_summary["request_ids"]
                )
                cached = 0 if stage == "cold" else 7936
                replayed = (
                    1024 if stage == "replay" and expected_mode == "bounded_on" else 0
                )
                for response in responses:
                    metadata = response["meta_info"]
                    assert metadata["prompt_tokens"] == 8192
                    assert metadata["cached_tokens"] == cached
                    assert metadata["completion_tokens"] == 2
                    assert metadata.get("num_retractions", 0) == 0
                    assert len(response["output_ids"]) == 2
                before = read_counters(
                    (trial_directory / f"{stage}.metrics_before.txt").read_text()
                )
                after = read_counters(
                    (trial_directory / f"{stage}.metrics_after.txt").read_text()
                )
                delta = {key: after[key] - before[key] for key in before}
                assert delta == stage_summary["counter_delta"]
                assert delta["prefill_cache"] == 8 * cached
                assert delta["prefill_compute"] == 8 * (8192 - cached) + replayed
                logs = parse_prefill_log(
                    (trial_directory / f"{stage}.server.log").read_text()
                )
                assert logs == stage_summary["prefill_log_records"]
                assert sum(item["replay-token"] for item in logs) == replayed
                if stage == "replay":
                    assert len(logs) == 1 and logs[0]["new-seq"] == 8
                    assert logs[0]["cached-device"] == 8 * cached
                    assert logs[0]["cached-host"] == logs[0]["cached-storage"] == 0
                    assert logs[0]["new-token"] == 2048
                duration = stage_summary["client_batch_latency_seconds"]
                assert duration > 0
                stream_responses = {}
                for event in (
                    (trial_directory / f"{stage}.response.sse")
                    .read_bytes()
                    .split(b"\n\n")
                ):
                    if event.startswith(b"data: ") and event != b"data: [DONE]":
                        decoded = json.loads(event[6:])
                        stream_responses[decoded["meta_info"]["id"]] = decoded
                assert stream_responses == {
                    item["meta_info"]["id"]: item for item in responses
                }
                stream_events = load_json(
                    trial_directory / f"{stage}.stream_events.json"
                )
                first_events = {}
                for event in stream_events:
                    first_events.setdefault(event["request_id"], event)
                assert len(first_events) == 8
                assert all(
                    event["completion_tokens"] == 1 for event in first_events.values()
                )
                prefill_duration = max(
                    event["received_seconds"] for event in first_events.values()
                )
                assert (
                    prefill_duration
                    == stage_summary["client_batch_first_token_seconds"]
                )
                assert 0 < prefill_duration < duration
                assert (
                    abs(
                        stage_summary["prefill_logical_input_tokens_per_second"]
                        * prefill_duration
                        - 65536
                    )
                    < 1e-6
                )
                assert (
                    abs(
                        stage_summary["logical_input_tokens_per_second"] * duration
                        - 65536
                    )
                    < 1e-6
                )
                assert abs(stage_summary["requests_per_second"] * duration - 8) < 1e-9
                checked_stages += 1
                checked_requests += len(responses)
    assert checked_measured_batches == 40
    report = {
        "passed": True,
        "checked_server_blocks": len(result_paths),
        "checked_stages_including_warmup": checked_stages,
        "checked_requests_including_cold_and_warmup": checked_requests,
        "checked_measured_replay_batches": checked_measured_batches,
        "dataset_sha256": dataset_hash,
        "checks": [
            "TP8/EP8 and eight distinct physical GPUs",
            "actual FP4 Main KV and Indexer K, FP8 SWA on all ranks",
            "default paged SWA capacity with no ratio/headroom override",
            "raw responses, summaries, metrics and server logs agree",
            "each cold pass has zero cached tokens",
            "each second pass hits 7936 tokens/request and has actual batch=8",
            "bounded replay computes 8x128 extra tokens; disabled computes zero",
            "only second-pass client wall times enter throughput calculation",
        ],
    }
    (results_directory / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directory", type=Path)
    arguments = parser.parse_args()
    validate(arguments.results_directory)


if __name__ == "__main__":
    main()
