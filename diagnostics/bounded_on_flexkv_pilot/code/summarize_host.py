import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark import write_json
from summarize import summarize_group


TITLES = {
    "bounded_on_l2": "开启：HiCache L2 Main/Indexer + SWA replay",
    "bounded_off_host": "关闭：FlexKV host Main/Indexer/SWA 回读",
}


def make_report(summary):
    tables = {"prefill": [], "complete": [], "blocks": []}
    for backend, statistics in summary["groups"].items():
        tables["prefill"].append(
            f"| {TITLES[backend]} | {statistics['samples']} "
            f"| {statistics['mean_batch_first_token_seconds'] * 1000:.3f} "
            f"| {statistics['p95_batch_first_token_seconds'] * 1000:.3f} "
            f"| {statistics['aggregate_prefill_logical_input_tokens_per_second']:,.1f} |"
        )
        tables["complete"].append(
            f"| {TITLES[backend]} | {statistics['samples']} "
            f"| {statistics['mean_batch_seconds'] * 1000:.3f} "
            f"| {statistics['p95_batch_seconds'] * 1000:.3f} "
            f"| {statistics['aggregate_logical_input_tokens_per_second']:,.1f} "
            f"| {statistics['aggregate_requests_per_second']:.3f} |"
        )
    for block in summary["blocks"]:
        statistics = block["summary"]
        tables["blocks"].append(
            f"| {block['name']} | {statistics['samples']} "
            f"| {statistics['mean_batch_first_token_seconds'] * 1000:.3f} "
            f"| {statistics['mean_batch_seconds'] * 1000:.3f} "
            f"| {statistics['aggregate_prefill_logical_input_tokens_per_second']:,.1f} "
            f"| {statistics['aggregate_logical_input_tokens_per_second']:,.1f} |"
        )
    return f"""# DeepSeek-V4.1：HiCache L2 / FlexKV host 吞吐对照

生成时间：{summary["generated_utc"]}。本报告仅使用此次重新执行的 **host/L2 命中** 数据，不复用之前 GPU-only 实验的计时。

## 1. 结果

**首 token 阶段**（直到 batch 中全部 8 条请求收到首 token）：

| 组别 | 有效 batch | 平均耗时 ms | P95 ms | 有效输入 token/s |
|---|---:|---:|---:|---:|
{chr(10).join(tables["prefill"])}

**完整请求**（每条输出 2 token）：

| 组别 | 有效 batch | 平均耗时 ms | P95 ms | 有效输入 token/s | request/s |
|---|---:|---:|---:|---:|---:|
{chr(10).join(tables["complete"])}

- 第二组 / 第一组的首 token 阶段吞吐比：**{summary["off_vs_on_prefill_throughput_ratio"]:.3f}×**。
- 第二组 / 第一组的完整请求吞吐比：**{summary["off_vs_on_throughput_ratio"]:.3f}×**。
- 输入吞吐包含命中的逻辑前缀，即 `batch数×8×8192 / 对应阶段耗时之和`，不是重新计算全部 8192 token 的算力吞吐。
- 两组的计时均从第二遍 HTTP 发送前开始，**包含 host lookup 与 H2D 回迁时间**；不包含首次填充、D2H 等待和 L1 驱逐。

## 2. 缓存来源与命中证据

| 项目 | 第一组：开启 replay | 第二组：关闭 replay |
|---|---|---|
| Host 管理器 | 原生 HiCache L2 | FlexKV CPU cache |
| Main KV / Indexer K | FP4，从 L2 回读 | FP4，从 FlexKV host 回读 |
| SWA KV | FP8 request-owned window，有界重建 | FP8 paged SWA，从 FlexKV host 回读 |
| 每请求 device cache hit | **0** | **0** |
| 每请求 host cache hit | **7936 token** | **7936 token** |
| 每 batch host hit | **63488 token** | **63488 token** |
| 每 batch storage hit | **0** | **0** |
| 每 batch 普通尾部 prefill | 2048 token | 2048 token |
| 每 batch 额外 SWA replay | **1024 token（8×128）** | **0** |

SGLang 需保留 next-token logits 的重算 token，并按 256-token page 对齐。因此 8192-token 请求的可复用前缀为 `floor((8192-1)/256)×256=7936`，其余 256 token 正常重算。这里“全命中”指这个可复用前缀需要的缓存类型全部具备，不声称 8192/8192 token 全命中。

每个样本检查原始响应的 `cached_tokens_details`、prefill 日志的 `cached-device/host/storage`、Prometheus prefill compute/cache 差值，以及实际 `#new-seq=8`。FlexKV 还逐 request ID 检查成功的 H2D 完成日志。全部验证结果见 `validation.json`。

## 3. 固定运行条件

- 本地 DeepSeek-V4.1-Flash；单机 8×NVIDIA B200，TP8/EP8，DP1/CP1。
- batch=8，严格 8192 输入 token/请求，两次均输出 2 token，temperature=0、ignore_eos=true。
- Main C1/C2 `v41_fp4`（288 bytes/token），Indexer K packed FP4（68 bytes/token），SWA `v41` FP8（528 bytes/token）；每个 worker 初始化时审计真实物理布局。
- GPU FULL cache 上限 131072 token；max-running-requests=8；context=16384；mem-fraction-static=0.85。
- **未扩大 GPU paged SWA**。第二组容量仍为默认 **35072 token/worker**，没有设置 SWA ratio/headroom 扩容参数。
- 第一组 HiCache ratio=2、write_through、kernel/page_first，host 上只建立 Main/Indexer side pools，不保存 SWA；第二组 FlexKV cpu_cache_gb=16、swa_multi_group=true，使用 bulk/no-layerwise transfer，另有 FlexKV SWA host pool。
- prefill/decode CUDA Graph 均关闭，decoder bounded replay 关闭，不使用 speculative decoding、SSD/远端存储或模型权重 CPU offload。
- CUDA MPS 服务在第一组开始前就已运行，四轮均保持可用，并非在第二组才临时引入；每个 block 的 `gpu_before.txt` 均有记录。复现入口会在第一组前确保 MPS 已启动。
- 沿用同一 FlashMLA shared-memory 容量保护补丁；native scheduling 用于超限冷填充。相关 42 项回归测试已通过，未为两组单独改动 kernel。

## 4. 首次填充与 L1 驱逐

每个样本只推理同一批输入两次，不额外推理缩短的 prefix。首次也是一次 HTTP batch=8 提交，但**仅冷填充阶段**临时使用 chunk=7936、每个 prefill batch 最多 1 条请求：每条先计算 7936 token，再计算末尾 256 token，正常完成 2-token 输出。

FlexKV 的 `SGLANG_FLEXKV_SWA_GRID_PAGES=31` 在冷填充的 7936-token chunk 边界保存可回读的 SWA 快照。只有 8192-token turn-end 快照不能保证第二遍的 7936-token 前缀命中。本控制两组相同，避免通过额外 prefix 请求、扩大 SWA 或隐藏 cache miss 来制造命中。

首次完成后，实验 RPC 在空闲 scheduler 上排空 D2H，再调用 cache 原有的 device eviction API，保留 host cache。8 个 TP rank 的 L1 evictable/protected token 均归零；HiCache 的 host pool 分配在驱逐前后保持不变。随后恢复 chunk=8192、prefill batch 上限=8，才开始计时第二遍。

`host_control.py` 的 hook 只在初始化和上述非计时控制阶段运行，不包装 timed forward/kernel/cache lookup。每个样本保留 cold_phase、evict_for_replay、drain 的 8-rank 控制记录，以及 cold/replay 原始请求体，验证两遍除 rid 外完全相同。不同样本使用不同 cache salt，首次请求必须全部 miss。

## 5. ABBA 与解释范围

四个独立服务 block 按 A-B-B-A 顺序运行。每个 block 丢弃 3 对 warmup，再保留 10 对测量；每组 20 个正式重放 batch、160 条重放请求。所有有效样本均保留，不按速度筛选。

| Block | 样本数 | 平均首 token ms | 平均完成 ms | 首 token 输入 token/s | 完整请求输入 token/s |
|---|---:|---:|---:|---:|---:|
{chr(10).join(tables["blocks"])}

两组所有冷填充/重放 output token IDs 一致，跨组也一致；单字母检索答案全部正确。这只是 smoke check，不是完整模型准确率评测。

**这不是仅切换 replay 开关的单变量消融**：按要求，第一组使用 HiCache L2，第二组使用 FlexKV，因此传输后端、实现及 SWA 恢复方式均不同。结果不能将全部差异单独归因于 bounded replay，也不能外推到长输出 decode、CUDA Graph 开启、其他请求长度或并发容量极限。未进行 GPU kernel 级 profiling。

## 6. 证据与复现

- `requests.json`、`*/measure_*/*.request.json`：精确 input IDs、盐和完整两遍请求。
- `*/measure_*/*.response.sse`、`*.response.json`、`*.stream_events.json`：原始 wire 响应及客户端事件时间。
- `*.metrics_before/after.txt`、`*.server.log`：缓存来源、实际 batch、重放及 H2D 证据。
- `*/pool_audit/`、`*/host_backend_audit/`、`*.rank_*.json`：全部 GPU/host 池布局和 L1 驱逐记录。
- `trials.csv`、`summary.json`、`validation.json`：逐样本与机器可读结论。
- `code/`、`environment/`、`environment_after/`：脚本快照、SGLang/FlexKV 版本和源码/native 指纹；`artifacts.sha256` 提供完整性校验。
- 复现：`bash /root/swa_replay_perf/run_host_reproduction.sh /root/swa_replay_perf/host-repro-$(date -u +%Y%m%dT%H%M%SZ)`。该入口先确保共享 MPS 可用，再调用原 ABBA 编排脚本。

旧的 `../results/` 是 GPU-only 历史实验，不参与本报告。开发期失败及独立 pilot 保留在 `../diagnostics/`，不计入上述样本数。
"""


def summarize_host_results(directory):
    validation = json.loads((directory / "validation.json").read_text())
    assert validation["passed"]
    groups = {backend: [] for backend in TITLES}
    block_summaries = []
    rows = []
    for block_path in sorted(directory.glob("*/results.json")):
        block = json.loads(block_path.read_text())
        assert block["complete"] and block["protocol"] == "host_l2_v1"
        measured = [trial for trial in block["trials"] if trial["measured"]]
        groups[block["backend"]].extend(measured)
        block_summaries.append(
            {
                "name": block_path.parent.name,
                "backend": block["backend"],
                "summary": summarize_group(measured),
            }
        )
        for trial in measured:
            replay = trial["replay"]
            rows.append(
                {
                    "block": block_path.parent.name,
                    "backend": block["backend"],
                    "trial": trial["name"],
                    "batch_first_token_seconds": replay[
                        "client_batch_first_token_seconds"
                    ],
                    "batch_completion_seconds": replay["client_batch_latency_seconds"],
                    "prefill_logical_input_tokens_per_second": replay[
                        "prefill_logical_input_tokens_per_second"
                    ],
                    "complete_logical_input_tokens_per_second": replay[
                        "logical_input_tokens_per_second"
                    ],
                    "requests_per_second": replay["requests_per_second"],
                    "host_cached_tokens": sum(
                        detail["host"] for detail in replay["cached_tokens_details"]
                    ),
                    "device_cached_tokens": sum(
                        detail["device"] for detail in replay["cached_tokens_details"]
                    ),
                    "swa_replay_tokens": replay["replay_tokens_inferred"],
                }
            )
    assert all(len(trials) == 20 for trials in groups.values())
    statistics = {
        backend: summarize_group(trials) for backend, trials in groups.items()
    }
    enabled = statistics["bounded_on_l2"]
    disabled = statistics["bounded_off_host"]
    summary = {
        "protocol": "host_l2_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": validation["dataset_sha256"],
        "groups": statistics,
        "blocks": block_summaries,
        "off_vs_on_prefill_throughput_ratio": disabled[
            "aggregate_prefill_logical_input_tokens_per_second"
        ]
        / enabled["aggregate_prefill_logical_input_tokens_per_second"],
        "off_vs_on_throughput_ratio": disabled[
            "aggregate_logical_input_tokens_per_second"
        ]
        / enabled["aggregate_logical_input_tokens_per_second"],
        "all_device_hits_zero": validation["all_device_hits_zero"],
        "host_cached_tokens_per_request": 7936,
    }
    write_json(directory / "summary.json", summary)
    with (directory / "trials.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "REPORT.md").write_text(make_report(summary))
    checksums = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "artifacts.sha256":
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            checksums.append(f"{digest}  {path.relative_to(directory)}")
    (directory / "artifacts.sha256").write_text("\n".join(checksums) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize_host_results(parser.parse_args().directory)


if __name__ == "__main__":
    main()
