# DeepSeek V4.1：三组 host-cache 重放吞吐比较

生成时间（UTC）：2026-09-29T09:19:29.165141+00:00。本报告保留原两组测量，新增第三组，所有吞吐均只计第二遍。

## 1. 结果

固定：8×B200、TP8/EP8、batch=8、8192 输入 token/请求、2 输出 token/请求；Main KV/Indexer K 为 FP4，SWA 为 FP8，不扩大 paged SWA。

| 组别 | 有效 batch | 首 token 均值 ms | 首 token P95 ms | 首 token 输入 token/s | 完整请求均值 ms | 完整请求输入 token/s |
|---|---:|---:|---:|---:|---:|---:|
| 第一组：HiCache L2 Main/Indexer + SWA replay | 20 | 1582.531 | 1588.839 | 41,412.1 | 1739.836 | 37,667.9 |
| 第二组：FlexKV host Main/Indexer/SWA | 20 | 342.952 | 346.780 | 191,093.5 | 476.065 | 137,662.0 |
| 第三组：FlexKV host Main/Indexer + SWA replay | 20 | 1618.428 | 1633.163 | 40,493.6 | 1773.492 | 36,953.1 |

- **第三组 / 第一组**：首 token 吞吐 **0.9778×**（-2.22%），完整请求吞吐 **0.9810×**（-1.90%）。两组都重放 SWA，host 管理器及 cache wrapper 不同。
- **第二组 / 第三组**：首 token 吞吐 **4.719×**，完整请求吞吐 **3.725×**。两组均使用 FlexKV，区别包含 SWA host 回迁和有界重建以及相应 cache 实现。
- 原 **第二组 / 第一组**：首 token **4.614×**，完整请求 **3.655×**；原数据没有覆盖或重测。

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

第三组八个 rank 均审计到四组 packed uint8 Main/Indexer buffers（实际物理 FP4），`enable_cpu=true`、CPU cache 配置 16 GiB、SSD/remote 禁用。`flexkv_swa_pool=false`、`swa=null`、`enable_swa_transfer=false`；逐 request ID 检查 H2D `slots=7936`、`swa_slots=0` 成功完成，再检查 batch 额外 replay=1024，排除 GPU-only 命中和 SWA host 回迁。

三组 Main C1/C2 物理布局均为 `v41_fp4`（288 bytes/token），Indexer 为 packed FP4（68 bytes/token），SWA 为 `v41` FP8（528 bytes/token）。ON 的 window 物理容量按页对齐为 256，但实际 SWA replay 为每请求 128 token。通用 dtype 标签不能代替这些物理布局审计。

## 3. 相同的填充及驱逐协议

每个样本均只执行同一批完整输入两遍。非计时首次填充临时采用 chunk=7936、prefill-max-requests=1；首次仍是一次 batch=8 HTTP 提交。该协议在第二组生成 7936 前缀的 SWA snapshot，第三组虽不需要 SWA snapshot 也保持相同冷填充方式。

冷请求完成后，RPC 排空 D2H，等待 scheduler 空闲，通过原有 eviction API 仅驱逐 L1；八个 rank 的 full/SWA evictable/protected 计数均归零，host 保留。随后恢复 chunk=8192、prefill-max-requests=8，才开始计时第二遍。每个第二遍只有一次 `#new-seq=8` 的 prefill，host=63488、device/storage=0，零 retraction。

首次与第二次的原始 payload 除 request ID 外相同；每个样本使用独立 salt，首次全部 miss。验证所有原始 SSE、cache 明细、指标计数、输出 IDs 和 A–H retrieval 答案。三组共验证 **1248 条请求**（含 cold/热身），其中 **60 个第二遍 batch、480 条请求**参与吞吐统计。

## 4. 采样与可比性

原第一、二组是 ABBA 四轮；第三组后续追加 CC 两轮。每轮独立重启服务，丢弃 3 对 cold/replay 热身，保留 10 对正式样本，因此每组 20 个有效 batch。不是三组随机交错实验，不能排除时间漂移；细小差异不做统计显著性断言。

| 轮次 | 首 token 均值 ms | 首 token 输入 token/s | 完整请求输入 token/s | 实验起止 UTC（含热身） |
|---|---:|---:|---:|---|
| baseline/01_bounded_on_l2 | 1585.418 | 41,336.7 | 37,598.1 | 2026-09-29T08:32:21.155964+00:00 — 2026-09-29T08:33:42.681597+00:00 |
| baseline/02_bounded_off_host | 341.004 | 192,185.3 | 138,232.7 | 2026-09-29T08:35:51.947736+00:00 — 2026-09-29T08:36:43.038317+00:00 |
| baseline/03_bounded_off_host | 344.901 | 190,014.1 | 137,096.0 | 2026-09-29T08:38:54.665585+00:00 — 2026-09-29T08:39:45.990202+00:00 |
| baseline/04_bounded_on_l2 | 1579.644 | 41,487.8 | 37,738.0 | 2026-09-29T08:41:26.290602+00:00 — 2026-09-29T08:42:47.581199+00:00 |
| third/01_bounded_on_flexkv | 1612.631 | 40,639.2 | 37,097.7 | 2026-09-29T09:14:18.522642+00:00 — 2026-09-29T09:15:37.627650+00:00 |
| third/02_bounded_on_flexkv | 1624.225 | 40,349.1 | 36,809.6 | 2026-09-29T09:17:41.426568+00:00 — 2026-09-29T09:19:01.155529+00:00 |

MPS 在各轮开始前均可用；第三组使用新启动的 MPS 实例，原两组实验结束后原实例已关闭。保持相同硬件、模型、依赖包、推理源码和 FlashMLA 容量保护补丁，未更改模型执行或 timed client；源码指纹核对见 `validation.json`。不同 cache wrapper、调度与传输成本均包含在测得的总时间内，不把两个配置的耗时差直接称为独立 SWA kernel 耗时。

## 5. 数据与复现

- 原两组数据（只读保留）：`/root/swa_replay_perf/host_results`。
- 第三组原始数据：`/root/swa_replay_perf/third_group_results`；其中有两轮 `results.json`、原始请求/SSE、8-rank 审计、环境及脚本快照。
- 本目录 `summary.json`、`trials.csv`、`validation.json` 分别保存聚合结果、60 个有效 batch、跨组三层校验；`source_artifacts.sha256` 引用两套原始 manifest，`artifacts.sha256` 校验本报告及分析代码快照。
- 新实验入口：`../run_third_experiment.sh NEW_THIRD_DIR BASELINE_DIR NEW_COMPARISON_DIR`；完整协议见 `../THIRD_EXPERIMENT.md`。
- 原两组各自报告为 `../host_results/REPORT.md`。`../results/` 是 GPU-only 历史结果，pilot 在 `../diagnostics/bounded_on_flexkv_pilot/`，均不进入此次正式统计。

数据集 SHA-256：`2b720466cd08e49f12ec815b63e93b3e405d28bec633f96e17302d40fe55a357`。
