import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_counters(metrics_text):
    counters = {"prefill_compute": 0.0, "prefill_cache": 0.0, "decode": 0.0}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name == "sglang:realtime_tokens_total":
                mode = sample.labels.get("mode")
                if mode in counters:
                    counters[mode] += sample.value
    return counters


def metrics_snapshot(session, base_url, path):
    response = session.get(base_url + "/metrics", timeout=30)
    response.raise_for_status()
    path.write_text(response.text)
    return read_counters(response.text)


def parse_prefill_log(text):
    records = []
    for line in text.splitlines():
        if "Prefill batch" not in line:
            continue
        record = {}
        for key in (
            "new-seq",
            "new-token",
            "cached-token",
            "cached-device",
            "cached-host",
            "cached-storage",
            "replay-token",
        ):
            match = re.search(r"#" + key + r": (\d+)", line)
            record[key] = int(match.group(1)) if match else 0
        records.append(record)
    return records


def checked_server_info(session, base_url, mode, output_directory):
    response = session.get(base_url + "/get_server_info", timeout=30)
    response.raise_for_status()
    server_info = response.json()
    write_json(output_directory / "server_info.json", server_info)
    arguments = server_info.get("server_args", server_info)
    expected = {
        "tp_size": 8,
        "ep_size": 8,
        "dp_size": 1,
        "max_running_requests": 8,
        "chunked_prefill_size": 8192,
        "max_total_tokens": 131072,
        "kv_cache_dtype": "fp8_e4m3",
        "enable_encoder_swa_bounded_replay": mode == "bounded_on",
        "enable_decoder_swa_bounded_replay": False,
        "disable_radix_cache": False,
    }
    for key, value in expected.items():
        assert arguments.get(key) == value, (key, arguments.get(key), value)
    assert arguments.get("speculative_algorithm") is None, arguments
    assert not arguments.get("_swa_full_tokens_ratio_explicitly_set", False), arguments
    assert arguments.get("swa_prefix_tails") is None, arguments
    audits = [
        json.loads(path.read_text())
        for path in sorted((output_directory / "pool_audit").glob("*.json"))
    ]
    assert len(audits) == 8, f"Expected eight worker pool audits, got {len(audits)}"
    placement_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    (output_directory / "gpu_process_placement.csv").write_text(placement_output)
    audit_pids = {audit["pid"] for audit in audits}
    placement = {
        int(row[0].strip()): row[1].strip()
        for row in csv.reader(placement_output.splitlines())
        if len(row) >= 2 and int(row[0].strip()) in audit_pids
    }
    assert set(placement) == audit_pids, (placement, audit_pids)
    assert len(set(placement.values())) == 8, placement
    for audit in audits:
        assert audit["encoder_bounded_replay"] == (mode == "bounded_on"), audit
        assert all(item["layout"] == "v41_fp4" for item in audit["main_kv"].values())
        assert all(item["fp4"] for item in audit["indexer_k"].values())
        assert audit["swa_kv"]["layout"] == "v41", audit
    return arguments


def consume_stream(response, started_ns, raw_path):
    chunks = []
    pending = b""
    responses = {}
    events = []
    received_done = False
    try:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024):
            received_ns = time.perf_counter_ns()
            chunks.append(chunk)
            pending += chunk
            while b"\n\n" in pending:
                event, pending = pending.split(b"\n\n", 1)
                data_lines = [
                    line[5:].lstrip()
                    for line in event.splitlines()
                    if line.startswith(b"data:")
                ]
                if not data_lines:
                    continue
                data = b"\n".join(data_lines)
                if data == b"[DONE]":
                    received_done = True
                    continue
                item = json.loads(data)
                metadata = item["meta_info"]
                request_id = metadata["id"]
                responses[request_id] = item
                events.append(
                    {
                        "request_id": request_id,
                        "received_seconds": (received_ns - started_ns) / 1e9,
                        "completion_tokens": metadata["completion_tokens"],
                        "finish_reason": metadata.get("finish_reason"),
                    }
                )
        completed_ns = time.perf_counter_ns()
        assert received_done and not pending.strip(), pending
        return responses, events, completed_ns
    finally:
        raw_path.write_bytes(b"".join(chunks))


def generate_batch(session, arguments, dataset, trial_directory, stage):
    base_url = arguments.base_url
    request_ids = [
        f"{arguments.mode}-{trial_directory.name}-{stage}-{request_index}"
        for request_index in range(dataset["batch_size"])
    ]
    payload = {
        "input_ids": [item["input_ids"] for item in dataset["requests"]],
        "cache_salt": [item["cache_salt"] for item in dataset["requests"]],
        "rid": request_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2,
            "ignore_eos": True,
        },
        "return_logprob": False,
        "stream": True,
    }
    serialized = json.dumps(payload, separators=(",", ":")).encode()
    if getattr(arguments, "cache_tier", "device") == "host":
        (trial_directory / f"{stage}.request.json").write_bytes(serialized)
    before = metrics_snapshot(
        session, base_url, trial_directory / f"{stage}.metrics_before.txt"
    )
    server_log = arguments.output / "server.log"
    log_start = server_log.stat().st_size
    started_utc = datetime.now(timezone.utc).isoformat()
    started_ns = time.perf_counter_ns()
    with session.post(
        base_url + "/generate",
        data=serialized,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=900,
    ) as response:
        by_id, stream_events, completed_ns = consume_stream(
            response, started_ns, trial_directory / f"{stage}.response.sse"
        )
    latency_seconds = (completed_ns - started_ns) / 1e9
    assert set(by_id) == set(request_ids), by_id.keys()
    responses = [by_id[request_id] for request_id in request_ids]
    write_json(trial_directory / f"{stage}.response.json", responses)
    write_json(trial_directory / f"{stage}.stream_events.json", stream_events)
    first_events = {}
    for event in stream_events:
        first_events.setdefault(event["request_id"], event)
    assert all(event["completion_tokens"] == 1 for event in first_events.values())
    first_token_seconds = [
        first_events[request_id]["received_seconds"] for request_id in request_ids
    ]
    batch_first_token_seconds = max(first_token_seconds)
    assert 0 < batch_first_token_seconds < latency_seconds
    assert all(
        isinstance(item.get("output_ids"), list) and len(item["output_ids"]) == 2
        for item in responses
    ), responses
    after = metrics_snapshot(
        session, base_url, trial_directory / f"{stage}.metrics_after.txt"
    )
    with server_log.open("rb") as logfile:
        logfile.seek(log_start)
        log_text = logfile.read().decode(errors="replace")
    (trial_directory / f"{stage}.server.log").write_text(log_text)
    counter_delta = {key: after[key] - before[key] for key in before}
    metadata = [item["meta_info"] for item in responses]
    cached_tokens = [item["cached_tokens"] for item in metadata]
    prompt_tokens = [item["prompt_tokens"] for item in metadata]
    completion_tokens = [item["completion_tokens"] for item in metadata]
    assert prompt_tokens == [8192] * 8, prompt_tokens
    assert completion_tokens == [2] * 8, completion_tokens
    assert all(item.get("num_retractions", 0) == 0 for item in metadata), metadata
    uncached_tokens = sum(prompt_tokens) - sum(cached_tokens)
    result = {
        "stage": stage,
        "started_utc": started_utc,
        "request_ids": request_ids,
        "request_payload_sha256": hashlib.sha256(serialized).hexdigest(),
        "request_payload_bytes": len(serialized),
        "client_batch_latency_seconds": latency_seconds,
        "client_first_token_seconds": first_token_seconds,
        "client_batch_first_token_seconds": batch_first_token_seconds,
        "prefill_logical_input_tokens_per_second": sum(prompt_tokens)
        / batch_first_token_seconds,
        "logical_input_tokens_per_second": sum(prompt_tokens) / latency_seconds,
        "requests_per_second": len(responses) / latency_seconds,
        "output_tokens_per_second": sum(completion_tokens) / latency_seconds,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cached_tokens_details": [
            item.get("cached_tokens_details") for item in metadata
        ],
        "uncached_tokens_total": uncached_tokens,
        "completion_tokens": completion_tokens,
        "counter_delta": counter_delta,
        "replay_tokens_inferred": counter_delta["prefill_compute"] - uncached_tokens,
        "prefill_log_records": parse_prefill_log(log_text),
        "output_ids": [item.get("output_ids") for item in responses],
        "output_text": [item["text"] for item in responses],
        "expected_answers": [item["expected_answer"] for item in dataset["requests"]],
        "semantic_correct": [
            response_item["text"].strip() == request_item["expected_answer"]
            for response_item, request_item in zip(responses, dataset["requests"])
        ],
        "server_first_token_latency_seconds": [
            item.get("first_token_latency") for item in metadata
        ],
        "server_e2e_latency_seconds": [item.get("e2e_latency") for item in metadata],
        "queue_time_seconds": [item.get("queue_time") for item in metadata],
    }
    write_json(trial_directory / f"{stage}.summary.json", result)
    assert counter_delta["prefill_cache"] == sum(cached_tokens), result
    if stage == "cold":
        assert cached_tokens == [0] * 8, result
        assert result["replay_tokens_inferred"] == 0, result
    else:
        page_size = arguments.page_size
        expected_cached = ((8192 - 1) // page_size) * page_size
        assert cached_tokens == [expected_cached] * 8, result
        expected_replay = 8 * 128 if arguments.mode == "bounded_on" else 0
        assert result["replay_tokens_inferred"] == expected_replay, result
        assert len(result["prefill_log_records"]) == 1, result
        prefill = result["prefill_log_records"][0]
        assert prefill["new-seq"] == 8, result
        tier = getattr(arguments, "cache_tier", "device")
        other_tier = "host" if tier == "device" else "device"
        assert prefill[f"cached-{tier}"] == expected_cached * 8, result
        assert prefill[f"cached-{other_tier}"] == prefill["cached-storage"] == 0, result
        if tier == "host":
            for detail in result["cached_tokens_details"]:
                assert detail["host"] == expected_cached, result
                assert detail["device"] == 0, result
        assert prefill["replay-token"] == expected_replay, result
    return result


def benchmark(arguments):
    arguments.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(arguments.requests.read_text())
    assert dataset["batch_size"] == 8, dataset["batch_size"]
    assert all(len(item["input_ids"]) == 8192 for item in dataset["requests"])
    result_path = arguments.output / "results.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    experiment = {
        "mode": arguments.mode,
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
    with requests.Session() as session:
        session.trust_env = False
        info = checked_server_info(
            session, arguments.base_url, arguments.mode, arguments.output
        )
        arguments.page_size = info["page_size"]
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
            flush_response = session.post(
                arguments.base_url + "/flush_cache?timeout=30", timeout=40
            )
            (trial_directory / "flush_response.txt").write_text(flush_response.text)
            flush_response.raise_for_status()
            cold = generate_batch(session, arguments, dataset, trial_directory, "cold")
            replay = generate_batch(
                session, arguments, dataset, trial_directory, "replay"
            )
            trial = {
                "name": trial_name,
                "measured": measured,
                "cold": cold,
                "replay": replay,
                "cold_replay_output_ids_equal": cold["output_ids"]
                == replay["output_ids"],
                "cold_replay_text_equal": cold["output_text"] == replay["output_text"],
            }
            experiment["trials"].append(trial)
            write_json(result_path, experiment)
            print(
                f"{arguments.mode} {trial_name}: cold={cold['client_batch_latency_seconds']:.4f}s "
                f"replay={replay['client_batch_latency_seconds']:.4f}s "
                f"batch_ttft={replay['client_batch_first_token_seconds']:.4f}s "
                f"effective_input={replay['logical_input_tokens_per_second']:.1f} token/s "
                f"cached={replay['cached_tokens']} replay_tokens={replay['replay_tokens_inferred']:.0f} "
                f"output_equal={trial['cold_replay_output_ids_equal']}",
                flush=True,
            )
        flush_response = session.post(
            arguments.base_url + "/flush_cache?timeout=30", timeout=40
        )
        flush_response.raise_for_status()
    measured_trials = [trial for trial in experiment["trials"] if trial["measured"]]
    durations = [
        trial["replay"]["client_batch_latency_seconds"] for trial in measured_trials
    ]
    prefill_durations = [
        trial["replay"]["client_batch_first_token_seconds"] for trial in measured_trials
    ]
    experiment["summary"] = {
        "measured_pairs": len(durations),
        "total_replay_seconds": sum(durations),
        "mean_replay_seconds": statistics.mean(durations),
        "median_replay_seconds": statistics.median(durations),
        "aggregate_logical_input_tokens_per_second": len(durations)
        * 8
        * 8192
        / sum(durations),
        "aggregate_requests_per_second": len(durations) * 8 / sum(durations),
        "mean_batch_first_token_seconds": statistics.mean(prefill_durations),
        "aggregate_prefill_logical_input_tokens_per_second": len(durations)
        * 8
        * 8192
        / sum(prefill_durations),
        "all_cold_replay_output_ids_equal": all(
            trial["cold_replay_output_ids_equal"] for trial in measured_trials
        ),
        "all_semantically_correct": all(
            all(trial["replay"]["semantic_correct"]) for trial in measured_trials
        ),
    }
    experiment["complete"] = True
    experiment["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(result_path, experiment)
    print(json.dumps(experiment["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bounded_on", "bounded_off"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--requests", type=Path, default=Path(__file__).parent / "requests.json"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30180")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--warmup-pairs", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.pairs < 1 or arguments.warmup_pairs < 1:
        parser.error("At least one measured pair and one warmup pair are required")
    benchmark(arguments)


if __name__ == "__main__":
    main()
