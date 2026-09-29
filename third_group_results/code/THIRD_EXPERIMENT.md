# 第三组：FlexKV Main/Indexer host 命中 + encoder SWA bounded replay

第三组模式名为 `bounded_on_flexkv`。新增实验以已完成的第一、二组 `host_results/` 为基线，不覆盖其原始测量数据。

## 配置

- 本地 `/root/models/DeepSeek-V4.1-Flash`，8×B200，TP8/EP8、DP1/CP1。
- 第二遍严格 batch=8，每请求 8192 输入 token，两遍均输出 2 token，temperature=0、ignore_eos=true。
- Main KV C1/C2：`v41_fp4`、288 bytes/token；Indexer K C1/C2：packed FP4、68 bytes/token；SWA：`v41` FP8、528 bytes/token。
- 开启 `--enable-encoder-swa-bounded-replay` 和 `--enable-flexkv`，不开启原生 `--enable-hierarchical-cache`。
- 沿用 `flexkv_host.yaml`：`cpu_cache_gb: 16`、`swa_multi_group: true`。因为不存在 paged SWA pool，FlexKV 自动禁用 SWA host pool；配置审计必须为 `swa: null`、`enable_swa_transfer: false`。
- FlexKV 注册的四组为 `c1`、`c2`、`c1_indexer`、`c2_indexer`，没有 SWA host 传输；CPU cache 开启，SSD/remote 关闭。
- `FLEXKV_ENABLE_LAYERWISE_TRANSFER=0`，bulk/no-layerwise H2D；MPS 在启动实验服务前可用。
- Main GPU cache 上限 131072 token、max-running-requests=8、context=16384、mem-fraction-static=0.85；prefill/decode CUDA Graph 均关闭。
- **不扩大 paged SWA**：第三组与第一组一样使用 request-owned window（物理容量按页对齐为 256 token，实际有界重放 128 token），不创建 paged SWA cache。第二组仍使用原始默认 35072 token/worker。

本地 cache builder 在 encoder replay 下选择 `FlexKVRadixCache`；第二组使用 `FlexKVHybridRadixCache`，第一组使用原生 `UnifiedRadixCache` + HiCache。三组比较的是这些端到端实际配置，不是只替换一个独立的 memcpy 函数。

## 协议与证据

复用 `host_results/requests.json` 的八条 8K retrieval 输入；每个样本首次和第二次使用相同输入与 salt，不同样本 salt 不同。

1. 每个样本开始时清理设备 cache；FlexKV host 中可能仍有旧记录，但不同 salt 保证首次全部 miss。
2. 首次仍只提交一次完整 batch=8；非计时冷填充采用 chunk=7936、prefill-max-requests=1，与已完成的两组保持一致。
3. 空闲 scheduler RPC 等待 D2H 完成，调用现有 eviction API 仅驱逐 L1；八个 rank 的 full/SWA evictable/protected 计数均归零。
4. 恢复 chunk=8192、prefill-max-requests=8，开始计时第二遍；不在两遍之间调用会清空 HiCache L2 的 flush。
5. 每请求复用 7936 token（`floor((8192-1)/256)*256`），普通尾部 prefill 256 token，额外重放 SWA 128 token。每 batch host=63488、device/storage=0、normal prefill=2048、SWA replay=1024、prefill compute=3072。
6. 对每个 request ID 校验 FlexKV H2D `slots=7936`、`swa_slots=0`、`mode=no-layerwise` 和成功完成日志。由此确认实际发生 host 回读，且未使用 SWA host snapshot。
7. 检查首次/重放/跨组三者的输出 token IDs、A–H retrieval 答案、原始 SSE、首 token 到达时间、cache 明细、指标差值、真实 `#new-seq=8` 和零 retraction。

计时从第二遍 HTTP 发送前开始，包含 host lookup、H2D、SWA replay 和普通尾部 prefill；不包含首次填充、等待 D2H、L1 驱逐、metrics 查询和文件写入。

- 首 token 阶段：直到八条请求都收到第一个输出 token。
- 完整请求：直到八条请求都输出完两个 token。
- 吞吐为 `有效 batch 数 × 8 × 8192 / 对应阶段总耗时`，是包含缓存前缀的逻辑输入吞吐，不是长输出 decode 稳态吞吐。

## 采样

第三组独立启动两次服务，每次丢弃 3 对 cold/replay 热身，采集 10 对正式样本，共 **20 个有效 batch、160 条计时请求**。pilot 只保存在 `diagnostics/bounded_on_flexkv_pilot/`，不纳入正式统计。

原先两组保留 ABBA 四轮、每组 20 个有效 batch。第三组是后续追加的两轮，不把 ABBA + CC 宣称为三组交错/随机实验；报告保留每轮统计和时间范围，提醒可能的时间漂移。

## 复现

```bash
cd /root/swa_replay_perf
baseline_directory=/root/swa_replay_perf/host-repro-$(date -u +%Y%m%dT%H%M%SZ)
bash run_host_reproduction.sh "$baseline_directory"
bash run_third_experiment.sh \
  /root/swa_replay_perf/third-repro-$(date -u +%Y%m%dT%H%M%SZ) \
  "$baseline_directory"
```

第一个参数为新第三组原始数据目录，第二个为已验证的原两组目录，可选第三个参数为新的三组对比目录，默认在第三组目录名后追加 `-comparison`。三套目录不能相互嵌套，所有输出目录必须不存在；脚本绝不覆盖原始基线。

归档脚本已裁剪无关入口，保存的源码指纹仍属于采样时内容。重新采样时必须先用 `run_host_reproduction.sh NEW_BASELINE` 生成当前脚本的基线，再将 `NEW_BASELINE` 作为第二个参数运行本脚本；不能把新运行与归档基线混用，否则源码指纹检查会失败。单独复核保存的三组数据、不要重启 GPU 服务：

```bash
/root/nvfp4-validation/venv/bin/python summarize_three_groups.py \
  --baseline host_results --third third_group_results \
  --output comparison-recheck-$(date -u +%Y%m%dT%H%M%SZ)
```

`validate_host_artifacts.py` 默认仍严格要求原来的四轮/40 个有效 batch；只有显式 `--third-group` 才校验第三组固定两轮/20 个有效 batch，不接受任意减少样本数。汇总脚本只读复核基线，并验证两套数据的 SHA-256、共同 timing/kernel 源码指纹及跨组输出一致性。

运行脚本只停止自己启动的推理服务，不自动关闭共享 MPS。若本次启动了 MPS，结束后应先检查其所有 server/client 列表与进程所有权；只有确认没有其他客户端才能关闭。
