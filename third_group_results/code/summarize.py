import statistics


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def summarize_group(trials):
    durations = [trial["replay"]["client_batch_latency_seconds"] for trial in trials]
    total_seconds = sum(durations)
    prefill_durations = [
        trial["replay"]["client_batch_first_token_seconds"] for trial in trials
    ]
    server_latencies = [
        latency
        for trial in trials
        for latency in trial["replay"]["server_first_token_latency_seconds"]
        if latency is not None
    ]
    return {
        "samples": len(trials),
        "total_seconds": total_seconds,
        "mean_batch_seconds": statistics.mean(durations),
        "median_batch_seconds": statistics.median(durations),
        "p95_batch_seconds": percentile(durations, 0.95),
        "min_batch_seconds": min(durations),
        "max_batch_seconds": max(durations),
        "stdev_batch_seconds": statistics.stdev(durations) if len(durations) > 1 else 0,
        "aggregate_logical_input_tokens_per_second": len(trials)
        * 8
        * 8192
        / total_seconds,
        "aggregate_requests_per_second": len(trials) * 8 / total_seconds,
        "aggregate_output_tokens_per_second": len(trials) * 8 * 2 / total_seconds,
        "mean_batch_first_token_seconds": statistics.mean(prefill_durations),
        "median_batch_first_token_seconds": statistics.median(prefill_durations),
        "p95_batch_first_token_seconds": percentile(prefill_durations, 0.95),
        "aggregate_prefill_logical_input_tokens_per_second": len(trials)
        * 8
        * 8192
        / sum(prefill_durations),
        "aggregate_prefill_requests_per_second": len(trials)
        * 8
        / sum(prefill_durations),
        "mean_server_first_token_seconds": statistics.mean(server_latencies),
        "cached_tokens_per_request": sorted(
            {value for trial in trials for value in trial["replay"]["cached_tokens"]}
        ),
        "replay_tokens_per_batch": sorted(
            {trial["replay"]["replay_tokens_inferred"] for trial in trials}
        ),
        "all_cold_cache_zero": all(
            trial["cold"]["cached_tokens"] == [0] * 8 for trial in trials
        ),
        "all_cold_replay_output_ids_equal": all(
            trial["cold_replay_output_ids_equal"] for trial in trials
        ),
        "all_semantically_correct": all(
            all(trial["cold"]["semantic_correct"])
            and all(trial["replay"]["semantic_correct"])
            for trial in trials
        ),
    }
