# DeepSeek-V4.1 SWA bounded replay 吞吐对照实验

生成时间：2026-09-29T07:56:46.225522+00:00。所有表格仅使用每次独立填充后的**第二次请求**。

## 1. 结论

- **首 token / prefill 阶段**：关闭 encoder bounded replay 的有效输入吞吐是开启时的 **5.672 倍**。
- **完整 2-token 请求**：关闭后的有效输入吞吐是开启时的 **4.374 倍**；开启时吞吐相对下降 **77.14%**。
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
| 开启 encoder bounded replay | 20 | 1549.621 | 1552.339 | 1554.856 | 42,291.6 | 5.163 |
| 关闭，三类 KV 命中 | 20 | 273.184 | 273.082 | 274.725 | 239,896.6 | 29.284 |

**完整请求（每条 2 个输出 token）：**

| 组别 | 有效 batch 数 | 平均 batch ms | P50 ms | P95 ms | 有效输入 token/s | request/s |
|---|---:|---:|---:|---:|---:|---:|
| 开启 encoder bounded replay | 20 | 1699.930 | 1702.337 | 1705.348 | 38,552.2 | 4.706 |
| 关闭，三类 KV 命中 | 20 | 388.617 | 388.315 | 390.267 | 168,639.2 | 20.586 |

分 block 结果（便于观察预热、运行顺序和时间漂移）：

| Block | 模式 | 样本数 | 平均 batch 首 token ms | 平均完成 ms | 完整请求有效输入 token/s |
|---|---|---:|---:|---:|---:|
| 01_bounded_on | bounded_on | 10 | 1552.712 | 1703.150 | 38,479.3 |
| 02_bounded_off | bounded_off | 10 | 273.335 | 388.816 | 168,552.8 |
| 03_bounded_off | bounded_off | 10 | 273.034 | 388.418 | 168,725.6 |
| 04_bounded_on | bounded_on | 10 | 1546.530 | 1696.710 | 38,625.3 |

服务端平均首 token 延迟：开启 1546.220 ms，关闭 269.769 ms。

## 6. 校验与解释范围

- 所有首次请求均冷缓存：开启 `True`，关闭 `True`。
- 首次与第二次输出 token IDs 全部一致：开启 `True`，关闭 `True`。
- 两组的所有已测输出 token IDs 一致：`True`。
- 单字母检索语义检查全部通过：开启 `True`，关闭 `True`；这只是 smoke check，不是完整准确率评测。
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
