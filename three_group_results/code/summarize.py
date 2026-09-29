import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


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


def make_report(summary):
    groups = summary["groups"]
    enabled = groups["bounded_on"]
    disabled = groups["bounded_off"]
    rows = []
    prefill_rows = []
    for mode, title in (
        ("bounded_on", "开启 encoder bounded replay"),
        ("bounded_off", "关闭，三类 KV 命中"),
    ):
        group = groups[mode]
        prefill_rows.append(
            f"| {title} | {group['samples']} | {group['mean_batch_first_token_seconds'] * 1000:.3f} "
            f"| {group['median_batch_first_token_seconds'] * 1000:.3f} "
            f"| {group['p95_batch_first_token_seconds'] * 1000:.3f} "
            f"| {group['aggregate_prefill_logical_input_tokens_per_second']:,.1f} "
            f"| {group['aggregate_prefill_requests_per_second']:.3f} |"
        )
        rows.append(
            f"| {title} | {group['samples']} | {group['mean_batch_seconds'] * 1000:.3f} "
            f"| {group['median_batch_seconds'] * 1000:.3f} "
            f"| {group['p95_batch_seconds'] * 1000:.3f} "
            f"| {group['aggregate_logical_input_tokens_per_second']:,.1f} "
            f"| {group['aggregate_requests_per_second']:.3f} |"
        )
    block_rows = []
    for block in summary["blocks"]:
        statistics_block = block["summary"]
        block_rows.append(
            f"| {block['name']} | {block['mode']} | {statistics_block['measured_pairs']} "
            f"| {statistics_block['mean_batch_first_token_seconds'] * 1000:.3f} "
            f"| {statistics_block['mean_replay_seconds'] * 1000:.3f} "
            f"| {statistics_block['aggregate_logical_input_tokens_per_second']:,.1f} |"
        )
    return f"""# DeepSeek-V4.1 SWA bounded replay 吞吐对照实验

生成时间：{summary["generated_utc"]}。所有表格仅使用每次独立填充后的**第二次请求**。

## 1. 结论

- **首 token / prefill 阶段**：关闭 encoder bounded replay 的有效输入吞吐是开启时的 **{summary["off_vs_on_prefill_throughput_ratio"]:.3f} 倍**。
- **完整 2-token 请求**：关闭后的有效输入吞吐是开启时的 **{summary["off_vs_on_throughput_ratio"]:.3f} 倍**；开启时吞吐相对下降 **{summary["on_throughput_reduction_percent"]:.2f}%**。
- 这里的吞吐是命中缓存后的**逻辑输入 token 吞吐**，不是重新计算了 8192 token 的算力吞吐。
- 未指定输出长度，最终两次请求均固定为 **2 token/request**。流式记录每条请求首 token，用 batch 中最后一条首 token 的到达时间分离 prefill / SWA 重放成本；不代表长输出 decode 的稳态吞吐。

## 2. 固定配置

- 单机 8×NVIDIA B200；TP=8、EP=8、DP=1、CP=1；实际模型为本地 `DeepSeek-V4.1-Flash`。
- batch=8，每条输入严格为 8192 token；通过单个 `/generate` 原生批请求一起提交，计时日志验证 batch 确实为 8。
- Main KV：C1/C2 `v41_fp4`，512 维 E2M1，256 字节 payload + 32 字节 scale/token。
- Indexer K：C1/C2 packed FP4；SWA KV：`v41` FP8 E4M3，512 字节 payload + 16 字节 scale/token。
- 两组仅改变 `--enable-encoder-swa-bounded-replay`；decoder bounded replay 关闭，不使用 speculative decoding。
- `--max-running-requests 8 --chunked-prefill-size 8192 --max-total-tokens 131072`；**paged SWA 使用默认容量**，不设置 `--swa-full-tokens-ratio` 或 `--swa-prefix-tails` 扩容。
- 关闭组实际 paged SWA 容量为 **35072 token/worker**，在 `pool_audit/` 中核验；两组输入、输出预算和流式协议完全一致。
- 两组的 prefill / decode CUDA graphs 均关闭；温度 0，固定随机种子；不启用 FlexKV / HiCache / CPU offload。
- `pool_audit/` 保存全部 8 个 worker 的真实布局、字节数和容量断言；只在初始化检查，没有给 timed forward 加 hook。

## 3. 实验方法与计时

1. 为每个请求使用独立 `cache_salt`，防止 batch 内共有前缀造成额外命中；两组共享同一 `requests.json`。
2. 每个样本先 `/flush_cache`，然后首次推理 8 条请求填充缓存，断言这次 8 条请求的 `cached_tokens` 全部为 0。
3. 原样重放相同的 8 条输入和 cache salt；只更换追踪用 request ID。计时只涵盖第二次批请求。
4. 每个 server block 先丢弃 3 对完整 cold→replay 预热，再测 10 对独立 cold→replay。
5. 按 A-B-B-A 顺序重启服务测试，A=开启、B=关闭；每组共 20 个有效 batch、160 条重放请求。
6. 使用 `time.perf_counter_ns()` 记录本机持久 HTTP 连接发送前的时间、每条请求首 token 的 SSE 到达时间，以及完整响应收齐时间。请求 JSON 预先构建；流式接收/解析在计时内，写盘、metrics 查询、冷填充和 flush 不计入。逐条验证第一个 SSE 事件的 completion_tokens=1，最后一个为 2。
7. **首 token 阶段吞吐** = `样本数 × 8 × 8192 / batch 首 token 阶段耗时之和`，其中每个 batch 的阶段耗时取 8 条请求首 token 延迟的最大值。**完整请求吞吐**的分母换成完整 batch 耗时；request/s = `样本数 × 8 / 完整耗时之和`；output token/s = `样本数 × 8 × 2 / 完整耗时之和`。
8. 保留所有样本，不删除慢样本；报告均值、中位数、P95。服务端 `input throughput` 的分母是相邻日志的时间间隔，且分子不含 cache-hit token，**不拿它充当本实验主指标**。

**为什么是 2 token 而非 1 token？** 预实验发现，1-token 请求直接进入 finished-request 缓存路径，跳过 unfinished-request 路径的 out-of-window SWA 回收；默认 SWA 池在 8×8192 的整段缓存之间逐出/抖动，关闭组第二次请求全部 miss。两次均生成 2 token 会正常经过 unfinished-request 路径，保留必要的窗口，默认容量即能满足全部可复用前缀命中。未修改缓存回收逻辑、未扩大 SWA 池。1-token 的无效对照及 2-token 验证保存在 `../diagnostics/`，不混入正式统计。

## 4. 必须说明的命中边界

SGLang 为取得 next-token logits 会保留至少一个 token 重算，而本模型逻辑 page size 为 256。8192-token 相同请求第二次最多匹配 `floor((8192-1)/256)×256 = 7936` token。

因此，两组每请求均为 **7936 个 prefix token 命中 + 256 个尾部 token 重算**，不是 8192/8192 token 全部命中；输入 token 命中率为 **96.875%**。这里“全命中”指可复用前缀的 Main KV、Indexer K、SWA KV 三类缓存都可用。没有改动模型的最后一页重算语义，也没有把 cache miss 冒充 hit。

| 项目 | 开启 | 关闭 |
|---|---:|---:|
| 每 batch 逻辑输入 token | 65536 | 65536 |
| 每 batch GPU cached prefix token | 63488 | 63488 |
| 每 batch 常规新 prefill token | 2048 | 2048 |
| 每 batch 额外 SWA 重放 token | 1024（8×128） | 0 |
| Main KV / Indexer K | 命中 | 命中 |
| SWA | request-owned window 有界重放重建 | paged SWA 缓存命中 |

逐样本校验 HTTP `cached_tokens`、Prometheus `prefill_cache/prefill_compute` 差值和 `#replay-token`，三者必须一致；要求 timed prefill 只有一个 batch，`#new-seq=8`，且 `cached-host=cached-storage=0`、无重调度回退。

## 5. 第二次请求性能

**首 token / prefill 阶段（不把额外 decode 混入此指标）：**

| 组别 | 有效 batch 数 | 平均 batch 首 token ms | P50 ms | P95 ms | 有效输入 token/s | 前缀请求/s |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(prefill_rows)}

**完整请求（每条 2 个输出 token）：**

| 组别 | 有效 batch 数 | 平均 batch ms | P50 ms | P95 ms | 有效输入 token/s | request/s |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

分 block 结果（便于观察预热、运行顺序和时间漂移）：

| Block | 模式 | 样本数 | 平均 batch 首 token ms | 平均完成 ms | 完整请求有效输入 token/s |
|---|---|---:|---:|---:|---:|
{chr(10).join(block_rows)}

服务端平均首 token 延迟：开启 {enabled["mean_server_first_token_seconds"] * 1000:.3f} ms，关闭 {disabled["mean_server_first_token_seconds"] * 1000:.3f} ms。

## 6. 校验与解释范围

- 所有首次请求均冷缓存：开启 `{enabled["all_cold_cache_zero"]}`，关闭 `{disabled["all_cold_cache_zero"]}`。
- 首次与第二次输出 token IDs 全部一致：开启 `{enabled["all_cold_replay_output_ids_equal"]}`，关闭 `{disabled["all_cold_replay_output_ids_equal"]}`。
- 两组的所有已测输出 token IDs 一致：`{summary["cross_mode_outputs_equal"]}`。
- 单字母检索语义检查全部通过：开启 `{enabled["all_semantically_correct"]}`，关闭 `{disabled["all_semantically_correct"]}`；这只是 smoke check，不是完整准确率评测。
- 本地 `encoder_swa_replay.py` 在主 batch 前逐请求执行 128-token replay，再执行普通尾部 prefill；因此 batch=8 会额外进行 8 次 replay forward。这个执行方式能解释开销来源，但本次并未用 GPU profiler 分解各 kernel 耗时。
- 开启模式用很小的 request-owned SWA window 替代可共享 paged SWA 缓存；关闭模式保持默认 paged SWA 容量。实验验证实际命中，不衡量内存节省能带来的更大并发收益。
- 结果仅适用于当前代码与 native FlashMLA 构建、TP8/EP8、该文本输入集、8192 输入和 2 token 输出。不能外推为其他请求长度、长 decode、CUDA graphs、多模态或高并发容量极限。
- 启动过程中发现 8K 冷填充触发 FlashMLA 元数据优化的 48 KiB shared-memory 上限；统一应用 `../compatibility.patch` 做容量保护，超限回退原生调度。2048-token timed batch 和单请求 128-token replay 不超限，仍走原有优化。失败的 warmup 没有计入数据。

## 7. 原始证据与复现

- `requests.json`：8 条真实 input IDs、请求 salt、期望输出、每条 SHA-256。
- `*/server_info.json`、`*/pool_audit/*.json`：运行参数及 8 rank 物理 cache layout。
- `*/measure_*/*.response.sse`：两次请求的原始 SSE 响应；`*.response.json`：每条请求的最终 SSE 消息；`*.stream_events.json`：每个 SSE 事件的到达时间。
- 每个样本的 `*.summary.json`、`*.server.log`、`*.metrics_before.txt`、`*.metrics_after.txt`：计时、命中和实际重放证据。
- `trials.csv`：所有有效样本；`summary.json`：机器可读汇总；`validation.json`：从原始响应和计数重新核验的结果；`environment/`、`environment_after/`：前后环境、硬件、拓扑、模型配置及源代码/native kernel 指纹（包含首次 JIT 后的 FlashInfer 模块）。
- 在当前节点运行 `bash /root/swa_replay_perf/run_experiments.sh /root/swa_replay_perf/repro-$(date -u +%Y%m%dT%H%M%SZ)`。
- 脚本只管理自己启动的服务；结束后停止该服务并释放 8 张 GPU。除已记录的容量保护补丁外，不改变 SGLang 的缓存和调度逻辑；没有提交 Git commit。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directory", type=Path)
    arguments = parser.parse_args()
    groups = {"bounded_on": [], "bounded_off": []}
    blocks = []
    csv_rows = []
    dataset_hashes = set()
    for path in sorted(arguments.results_directory.glob("*/results.json")):
        block = json.loads(path.read_text())
        assert block["complete"], f"Incomplete block: {path}"
        dataset_hashes.add(block["dataset_sha256"])
        blocks.append(
            {
                "name": path.parent.name,
                "mode": block["mode"],
                "summary": block["summary"],
            }
        )
        for trial in block["trials"]:
            if not trial["measured"]:
                continue
            groups[block["mode"]].append(trial)
            replay = trial["replay"]
            csv_rows.append(
                {
                    "block": path.parent.name,
                    "mode": block["mode"],
                    "trial": trial["name"],
                    "batch_latency_seconds": replay["client_batch_latency_seconds"],
                    "batch_first_token_seconds": replay[
                        "client_batch_first_token_seconds"
                    ],
                    "prefill_logical_input_tokens_per_second": replay[
                        "prefill_logical_input_tokens_per_second"
                    ],
                    "logical_input_tokens_per_second": replay[
                        "logical_input_tokens_per_second"
                    ],
                    "requests_per_second": replay["requests_per_second"],
                    "cached_tokens_total": sum(replay["cached_tokens"]),
                    "uncached_tokens_total": replay["uncached_tokens_total"],
                    "replay_tokens": replay["replay_tokens_inferred"],
                    "cold_batch_seconds": trial["cold"]["client_batch_latency_seconds"],
                    "output_equal_to_cold": trial["cold_replay_output_ids_equal"],
                }
            )
    assert len(dataset_hashes) == 1, dataset_hashes
    assert all(groups.values()), "Both modes must have complete measured results"
    summary_groups = {mode: summarize_group(trials) for mode, trials in groups.items()}
    enabled = summary_groups["bounded_on"]
    disabled = summary_groups["bounded_off"]
    reference_outputs = groups["bounded_off"][0]["replay"]["output_ids"]
    cross_mode_outputs_equal = all(
        trial[stage]["output_ids"] == reference_outputs
        for trials in groups.values()
        for trial in trials
        for stage in ("cold", "replay")
    )
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": next(iter(dataset_hashes)),
        "groups": summary_groups,
        "blocks": blocks,
        "off_vs_on_throughput_ratio": disabled[
            "aggregate_logical_input_tokens_per_second"
        ]
        / enabled["aggregate_logical_input_tokens_per_second"],
        "off_vs_on_prefill_throughput_ratio": disabled[
            "aggregate_prefill_logical_input_tokens_per_second"
        ]
        / enabled["aggregate_prefill_logical_input_tokens_per_second"],
        "on_throughput_reduction_percent": (
            1
            - enabled["aggregate_logical_input_tokens_per_second"]
            / disabled["aggregate_logical_input_tokens_per_second"]
        )
        * 100,
        "on_latency_increase_percent": (
            enabled["mean_batch_seconds"] / disabled["mean_batch_seconds"] - 1
        )
        * 100,
        "cross_mode_outputs_equal": cross_mode_outputs_equal,
    }
    (arguments.results_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (arguments.results_directory / "trials.csv").open(
        "w", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (arguments.results_directory / "REPORT.md").write_text(make_report(summary))
    manifest = []
    for path in sorted(arguments.results_directory.rglob("*")):
        if path.is_file() and path.name != "artifacts.sha256":
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            manifest.append(
                f"{digest}  {path.relative_to(arguments.results_directory)}"
            )
    (arguments.results_directory / "artifacts.sha256").write_text(
        "\n".join(manifest) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
