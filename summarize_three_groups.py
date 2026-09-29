import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from benchmark import write_json
from summarize import summarize_group
from validate_host_artifacts import load_json, validate_host_results


TITLES = {
    "bounded_on_l2": "第一组：HiCache L2 Main/Indexer + SWA replay",
    "bounded_off_host": "第二组：FlexKV host Main/Indexer/SWA",
    "bounded_on_flexkv": "第三组：FlexKV host Main/Indexer + SWA replay",
}


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def artifact_files(directory):
    return {
        str(path.relative_to(directory)): path
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "artifacts.sha256"
    }


def write_manifest(directory):
    manifest = directory / "artifacts.sha256"
    if manifest.exists():
        raise FileExistsError(manifest)
    files = artifact_files(directory)
    manifest.write_text(
        "".join(f"{file_hash(files[name])}  {name}\n" for name in sorted(files))
    )


def verify_manifest(directory):
    entries = {}
    for line in (directory / "artifacts.sha256").read_text().splitlines():
        digest, relative_path = line.split("  ", 1)
        assert relative_path not in entries
        path = (directory / relative_path).resolve()
        assert path.is_relative_to(directory.resolve())
        assert file_hash(path) == digest, path
        entries[relative_path] = digest
    assert set(entries) == set(artifact_files(directory)), directory
    return {
        "files_verified": len(entries),
        "manifest_sha256": file_hash(directory / "artifacts.sha256"),
    }


def check_source_provenance(baseline, third):
    before = load_json(third / "environment/environment.json")
    after = load_json(third / "environment_after/environment.json")
    original = load_json(baseline / "environment/environment.json")
    original_after = load_json(baseline / "environment_after/environment.json")
    assert before["source_sha256"] == after["source_sha256"]
    for field in ("git_head", "git_status", "flexkv_git_head", "flexkv_git_status"):
        assert original[field]["stdout"] == before[field]["stdout"]
        assert before[field]["stdout"] == after[field]["stdout"]
    assert original["packages"] == before["packages"] == after["packages"]
    experiment_directory = Path(__file__).resolve().parent
    unchanged_helpers = {
        "benchmark.py",
        "host_control.py",
        "server_entry.py",
        "flexkv_host.yaml",
    }
    required_sources = {}
    helper_changes = []
    for source, digest in original["source_sha256"].items():
        path = Path(source)
        if path.parent == experiment_directory and path.name not in unchanged_helpers:
            if before["source_sha256"].get(source) != digest:
                helper_changes.append(source)
            continue
        assert original_after["source_sha256"][source] == digest, source
        assert before["source_sha256"][source] == digest, source
        required_sources[source] = digest
    assert len(required_sources) >= 17
    assert all(
        str(experiment_directory / name) in required_sources
        for name in unchanged_helpers
    )
    return {
        "third_group_source_hashes_unchanged_during_run": True,
        "common_model_kernel_and_timing_source_hashes_equal": True,
        "matched_source_sha256": required_sources,
        "extended_or_updated_helpers": helper_changes,
    }


def throughput_ratio(groups, numerator, denominator):
    first = (
        groups[numerator]["aggregate_prefill_logical_input_tokens_per_second"]
        / groups[denominator]["aggregate_prefill_logical_input_tokens_per_second"]
    )
    complete = (
        groups[numerator]["aggregate_logical_input_tokens_per_second"]
        / groups[denominator]["aggregate_logical_input_tokens_per_second"]
    )
    return {
        "first_token_throughput_ratio": first,
        "complete_throughput_ratio": complete,
        "first_token_change_percent": (first - 1) * 100,
        "complete_change_percent": (complete - 1) * 100,
    }


def make_report(summary):
    rows = []
    for backend, statistics in summary["groups"].items():
        rows.append(
            f"| {TITLES[backend]} | {statistics['samples']} "
            f"| {statistics['mean_batch_first_token_seconds'] * 1000:.3f} "
            f"| {statistics['p95_batch_first_token_seconds'] * 1000:.3f} "
            f"| {statistics['aggregate_prefill_logical_input_tokens_per_second']:,.1f} "
            f"| {statistics['mean_batch_seconds'] * 1000:.3f} "
            f"| {statistics['aggregate_logical_input_tokens_per_second']:,.1f} |"
        )
    block_rows = []
    for block in summary["blocks"]:
        statistics = block["summary"]
        block_rows.append(
            f"| {block['source']}/{block['name']} "
            f"| {statistics['mean_batch_first_token_seconds'] * 1000:.3f} "
            f"| {statistics['aggregate_prefill_logical_input_tokens_per_second']:,.1f} "
            f"| {statistics['aggregate_logical_input_tokens_per_second']:,.1f} "
            f"| {block['started_utc']} — {block['completed_utc']} |"
        )
    third_first = summary["ratios"]["group3_vs_group1"]
    second_third = summary["ratios"]["group2_vs_group3"]
    second_first = summary["ratios"]["group2_vs_group1"]
    return f"""# DeepSeek V4.1：三组 host-cache 重放吞吐比较

生成时间（UTC）：{summary["generated_utc"]}。本报告保留原两组测量，新增第三组，所有吞吐均只计第二遍。

## 1. 结果

固定：8×B200、TP8/EP8、batch=8、8192 输入 token/请求、2 输出 token/请求；Main KV/Indexer K 为 FP4，SWA 为 FP8，不扩大 paged SWA。

| 组别 | 有效 batch | 首 token 均值 ms | 首 token P95 ms | 首 token 输入 token/s | 完整请求均值 ms | 完整请求输入 token/s |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- **第三组 / 第一组**：首 token 吞吐 **{third_first["first_token_throughput_ratio"]:.4f}×**（{third_first["first_token_change_percent"]:+.2f}%），完整请求吞吐 **{third_first["complete_throughput_ratio"]:.4f}×**（{third_first["complete_change_percent"]:+.2f}%）。两组都重放 SWA，host 管理器及 cache wrapper 不同。
- **第二组 / 第三组**：首 token 吞吐 **{second_third["first_token_throughput_ratio"]:.3f}×**，完整请求吞吐 **{second_third["complete_throughput_ratio"]:.3f}×**。两组均使用 FlexKV，区别包含 SWA host 回迁和有界重建以及相应 cache 实现。
- 原 **第二组 / 第一组**：首 token **{second_first["first_token_throughput_ratio"]:.3f}×**，完整请求 **{second_first["complete_throughput_ratio"]:.3f}×**；原数据没有覆盖或重测。

“输入 token/s”是 `有效 batch 数×8×8192 / 对应阶段耗时之和`，包含已命中的逻辑输入，不表示重新计算全部 8192 token。首 token 阶段结束于 batch 中最后一条请求收到首 token；完整请求结束于全部两-token 响应完成。

两种计时均包含第二遍 host lookup、H2D、SWA 重放（若启用）和正常尾部 prefill；不包含冷填充、D2H 等待、L1 驱逐及外部 metrics 查询。输出仅两 token，不代表长输出 decode 稳态吞吐。

## 2. 三组缓存路径

| 项目 | 第一组 | 第二组 | 第三组 |
|---|---|---|---|
| Encoder bounded replay | ON | OFF | ON |
| Main KV/Indexer K | 原生 HiCache L2、FP4 | FlexKV host、FP4 | FlexKV host、FP4 |
| SWA KV | request-owned FP8 window，有界重建 | FlexKV host FP8 snapshot 回迁 | request-owned FP8 window，有界重建 |
| 本地 cache 类 | UnifiedRadixCache + HiCache | FlexKVHybridRadixCache | FlexKVRadixCache |
| 每请求 device / host 命中 token | 0 / 7936 | 0 / 7936 | 0 / 7936 |
| 每 batch 普通尾部 prefill token | 2048 | 2048 | 2048 |
| 每 batch 额外 SWA replay token | 1024 | 0 | 1024 |
| 每请求 FlexKV H2D swa_slots | 不适用 | 256 | **0** |
| GPU paged SWA | 不创建 | **35072 token/worker，未扩容** | 不创建 |

8192-token 请求需保留 next-token logits 重算 token，并按 256-token page 对齐，因此可复用前缀是 `floor((8192-1)/256)×256=7936`；每请求剩余 256 token 正常计算。“命中”指该可复用前缀需要的缓存类型，不声称 8192/8192 token 全命中。

第三组八个 rank 均审计到四组 packed uint8 Main/Indexer buffers（实际物理 FP4），`enable_cpu=true`、CPU cache 配置 16 GiB、SSD/remote 禁用。`flexkv_swa_pool=false`、`swa=null`、`enable_swa_transfer=false`；逐 request ID 检查 H2D `slots=7936`、`swa_slots=0` 成功完成，再检查 batch 额外 replay=1024，确认没有设备缓存命中和 SWA host 回迁。

三组 Main C1/C2 物理布局均为 `v41_fp4`（288 bytes/token），Indexer 为 packed FP4（68 bytes/token），SWA 为 `v41` FP8（528 bytes/token）。ON 的 window 物理容量按页对齐为 256，但实际 SWA replay 为每请求 128 token。通用 dtype 标签不能代替这些物理布局审计。

## 3. 相同的填充及驱逐协议

每个样本均只执行同一批完整输入两遍。非计时首次填充临时采用 chunk=7936、prefill-max-requests=1；首次仍是一次 batch=8 HTTP 提交。该协议在第二组生成 7936 前缀的 SWA snapshot，第三组虽不需要 SWA snapshot 也保持相同冷填充方式。

冷请求完成后，RPC 排空 D2H，等待 scheduler 空闲，通过原有 eviction API 仅驱逐 L1；八个 rank 的 full/SWA evictable/protected 计数均归零，host 保留。随后恢复 chunk=8192、prefill-max-requests=8，才开始计时第二遍。每个第二遍只有一次 `#new-seq=8` 的 prefill，host=63488、device/storage=0，零 retraction。

首次与第二次的原始 payload 除 request ID 外相同；每个样本使用独立 salt，首次全部 miss。验证所有原始 SSE、cache 明细、指标计数、输出 IDs 和 A–H retrieval 答案。三组共验证 **{summary["validation"]["requests_including_cold_and_warmup"]} 条请求**（含 cold/热身），其中 **60 个第二遍 batch、480 条请求**参与吞吐统计。

## 4. 采样与可比性

原第一、二组是 ABBA 四轮；第三组后续追加 CC 两轮。每轮独立重启服务，丢弃 3 对 cold/replay 热身，保留 10 对正式样本，因此每组 20 个有效 batch。不是三组随机交错实验，不能排除时间漂移；细小差异不做统计显著性断言。

| 轮次 | 首 token 均值 ms | 首 token 输入 token/s | 完整请求输入 token/s | 实验起止 UTC（含热身） |
|---|---:|---:|---:|---|
{chr(10).join(block_rows)}

MPS 在各轮开始前均可用；第三组使用新启动的 MPS 实例，原两组实验结束后原实例已关闭。保持相同硬件、模型、依赖包、推理源码和 FlashMLA 容量保护补丁，未更改模型执行或 timed client；源码指纹核对见 `validation.json`。不同 cache wrapper、调度与传输成本均包含在测得的总时间内，不把两个配置的耗时差直接称为独立 SWA kernel 耗时。

## 5. 数据与复现

- 原两组数据（只读保留）：`{summary["sources"]["baseline"]}`。
- 第三组原始数据：`{summary["sources"]["third"]}`；其中有两轮 `results.json`、原始请求/SSE、8-rank 审计、环境及脚本快照。
- 本目录 `summary.json`、`trials.csv`、`validation.json` 分别保存聚合结果、60 个有效 batch、跨组三层校验；`source_artifacts.sha256` 引用两套原始 manifest，`artifacts.sha256` 校验本报告及分析代码快照。
- 新实验入口：`../run_third_experiment.sh NEW_THIRD_DIR BASELINE_DIR NEW_COMPARISON_DIR`；完整协议见 `../THIRD_EXPERIMENT.md`。
- 原两组各自报告为 `../host_results/REPORT.md`。pilot 在 `../diagnostics/bounded_on_flexkv_pilot/`，不进入此次正式统计。

数据集 SHA-256：`{summary["dataset_sha256"]}`。
"""


def summarize_three_groups(baseline, third, output):
    directories = [path.resolve() for path in (baseline, third, output)]
    baseline, third, output = directories
    for directory in directories:
        assert not any(
            directory != other and directory.is_relative_to(other)
            for other in directories
        ), directories
    assert len(set(directories)) == 3
    if output.exists():
        raise FileExistsError(output)
    baseline_manifest = verify_manifest(baseline)
    baseline_validation = validate_host_results(baseline, write_results=False)
    assert baseline_validation == load_json(baseline / "validation.json")
    third_manifest_exists = (third / "artifacts.sha256").exists()
    if third_manifest_exists:
        verify_manifest(third)
    third_validation = validate_host_results(
        third, third_group=True, write_results=False
    )
    if (third / "validation.json").exists():
        assert third_validation == load_json(third / "validation.json")
    else:
        assert not third_manifest_exists
        write_json(third / "validation.json", third_validation)
    provenance = check_source_provenance(baseline, third)
    assert baseline_validation["dataset_sha256"] == third_validation["dataset_sha256"]
    trials = {backend: [] for backend in TITLES}
    blocks = []
    rows = []
    outputs = set()
    for source_name, directory in (("baseline", baseline), ("third", third)):
        for block_path in sorted(directory.glob("*/results.json")):
            block = load_json(block_path)
            measured = [trial for trial in block["trials"] if trial["measured"]]
            backend = block["backend"]
            trials[backend].extend(measured)
            blocks.append(
                {
                    "source": source_name,
                    "name": block_path.parent.name,
                    "backend": backend,
                    "started_utc": block["started_utc"],
                    "completed_utc": block["completed_utc"],
                    "summary": summarize_group(measured),
                }
            )
            for trial in block["trials"]:
                outputs.add(json.dumps(trial["replay"]["output_ids"]))
                if not trial["measured"]:
                    continue
                replay = trial["replay"]
                rows.append(
                    {
                        "source": source_name,
                        "block": block_path.parent.name,
                        "backend": backend,
                        "trial": trial["name"],
                        "batch_first_token_seconds": replay[
                            "client_batch_first_token_seconds"
                        ],
                        "batch_completion_seconds": replay[
                            "client_batch_latency_seconds"
                        ],
                        "first_token_input_tokens_per_second": replay[
                            "prefill_logical_input_tokens_per_second"
                        ],
                        "complete_input_tokens_per_second": replay[
                            "logical_input_tokens_per_second"
                        ],
                        "host_cached_tokens": sum(
                            detail["host"] for detail in replay["cached_tokens_details"]
                        ),
                        "device_cached_tokens": sum(
                            detail["device"]
                            for detail in replay["cached_tokens_details"]
                        ),
                        "swa_replay_tokens": replay["replay_tokens_inferred"],
                    }
                )
    assert len(outputs) == 1
    assert all(len(group_trials) == 20 for group_trials in trials.values())
    groups = {
        backend: summarize_group(group_trials)
        for backend, group_trials in trials.items()
    }
    original_summary = load_json(baseline / "summary.json")
    for backend in ("bounded_on_l2", "bounded_off_host"):
        assert groups[backend] == original_summary["groups"][backend]
    third_summary = {
        "protocol": "host_l2_v1",
        "backend": "bounded_on_flexkv",
        "dataset_sha256": third_validation["dataset_sha256"],
        "summary": groups["bounded_on_flexkv"],
        "blocks": [block for block in blocks if block["source"] == "third"],
    }
    if (third / "summary.json").exists():
        assert load_json(third / "summary.json") == third_summary
    else:
        assert not third_manifest_exists
        write_json(third / "summary.json", third_summary)
    if not third_manifest_exists:
        write_manifest(third)
    third_manifest = verify_manifest(third)
    assert verify_manifest(baseline) == baseline_manifest
    validation = {
        "passed": True,
        "original_baseline_artifacts_unchanged": True,
        "baseline": baseline_validation,
        "third": third_validation,
        "baseline_manifest": baseline_manifest,
        "third_manifest": third_manifest,
        "source_provenance": provenance,
        "all_three_groups_output_ids_equal": True,
        "measured_replay_batches": len(rows),
        "requests_including_cold_and_warmup": baseline_validation[
            "requests_including_cold_and_warmup"
        ]
        + third_validation["requests_including_cold_and_warmup"],
    }
    summary = {
        "protocol": "host_l2_v1_three_groups",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": third_validation["dataset_sha256"],
        "sources": {"baseline": str(baseline), "third": str(third)},
        "groups": groups,
        "blocks": blocks,
        "ratios": {
            "group3_vs_group1": throughput_ratio(
                groups, "bounded_on_flexkv", "bounded_on_l2"
            ),
            "group2_vs_group3": throughput_ratio(
                groups, "bounded_off_host", "bounded_on_flexkv"
            ),
            "group2_vs_group1": throughput_ratio(
                groups, "bounded_off_host", "bounded_on_l2"
            ),
        },
        "validation": validation,
    }
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    write_json(output / "validation.json", validation)
    with (output / "trials.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "REPORT.md").write_text(make_report(summary))
    (output / "source_artifacts.sha256").write_text(
        "".join(
            f"{file_hash(directory / 'artifacts.sha256')}  "
            f"{os.path.relpath(directory / 'artifacts.sha256', output)}\n"
            for directory in (baseline, third)
        )
    )
    code_directory = output / "code"
    code_directory.mkdir()
    for name in (
        "summarize_three_groups.py",
        "summarize.py",
        "validate_host_artifacts.py",
        "benchmark.py",
        "benchmark_host.py",
    ):
        (code_directory / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_manifest(output)
    verify_manifest(output)
    print(json.dumps({"groups": groups, "ratios": summary["ratios"]}, indent=2))
    print(f"Saved verified three-group comparison to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--third", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summarize_three_groups(arguments.baseline, arguments.third, arguments.output)


if __name__ == "__main__":
    main()
