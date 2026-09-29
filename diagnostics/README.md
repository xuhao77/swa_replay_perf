# 排除的预实验与启动记录

本目录全部排除在正式统计之外。第一、二组 host/L2 对照在 `../host_results/`，第三组在 `../third_group_results/`，三组汇总在 `../three_group_results/`；旧 GPU-only 对照在 `../results/`。

## 旧 GPU-only 预实验

`aborted_capacity_override/` 是初版启动记录。当用户补充“不需要扩大 paged SWA 容量”时，该服务仍在初始化/autotune，尚未发送实验请求，因此立即停止。

随后从启动脚本移除了 `--swa-full-tokens-ratio 1`，使用默认 paged SWA 配置重新启动。此目录不参与汇总，不属于任何有效样本。

`shared_memory_limit_8k/` 保存默认容量启动后的首次冷请求失败记录：8K prefill 的 FlashMLA 元数据预计算需要 168580 bytes shared memory，但优化 kernel 限制为 48 KiB。失败发生在第一个 warmup 的 cold 请求，没有任何有效计时样本。

修复是在 SGLang 的 `_maybe_precompute_flashmla_sched_meta` 入口检查容量；超限回退 FlashMLA 原生调度。8192-token 冷填充会回退，实测阶段的 2048-token batch 和每请求 128-token bounded replay 仍使用原有预计算优化。补丁在实验根目录 `compatibility.patch`，两组使用相同代码。没有采用临时讨论过的 4K 分块，正式配置仍为 8K 分块。

`one_token_output/` 保存最初的 1-token 输出协议：开启组首轮 10 个样本有效，但关闭组第一次 warmup 重放的 cached_tokens 全为 0，故不能作为要求的全命中对照。1-token 请求直接进入 finished 缓存路径，未经过 unfinished 缓存路径的 out-of-window SWA 回收，默认池 35072 token 无法长期缓存 8×8192 的整段 SWA，导致循环逐出。没有扩大该池。

`two_token_pilot/` 验证两次均输出 2 token 后，unfinished 缓存路径正常保留窗口，默认池下冷请求 cached_tokens=0，重放 8 条请求 cached_tokens 均为 7936。另验证流式响应每条请求先返回 completion_tokens=1，再返回 2。因此正式两组统一采用 2-token 流式协议，分别测量 batch 首 token 阶段和完成阶段。

## Host/L2 预实验

- `host_l2_pilot/`：初始化审计使用了本分支已移除的 scheduler rank 属性，启动失败；没有有效测量。
- `host_l2_pilot_v2/`：首次检查发现 host pool 审计读取了错误属性，正式 pilot 在请求前中止。独立 `control_probe/` 随后验证真实 HiCache L2 回读及 8-rank load-back 计数，但因该服务的初始化审计不完整，全部排除。最终审计已改为读取 UnifiedRadixCache 的 `host_pool_group`。
- `flexkv_host_pilot/`：验证 7936-token 快照、每请求 Main/Indexer 7936 slots 和 SWA 256 slots 的 H2D，以及默认 35072-token GPU SWA 容量。其 1 个 warmup 和 2 个试测 batch 仅用于验证协议，不进入正式 ABBA 数据。

正式 host 实验只读取 `host_results/` 下四个指定 block。全部运行后重新从请求、响应、metrics、H2D 日志和控制记录核验；未把旧 GPU-only 数据当作 host 对照。

## 第三组预实验

`bounded_on_flexkv_pilot/` 验证开启 encoder bounded replay 与 FlexKV 同时运行：八个 rank 均使用 `FlexKVRadixCache`，FlexKV 无 SWA host pool；每个第二遍请求 Main/Indexer 命中 7936 host token、device=0，H2D 为 `slots=7936, swa_slots=0`，每 batch 额外 replay=1024。全部输出与冷请求相同且检索正确。

该 pilot 含 1 对热身和 2 对试测，不计入第三组正式两轮/20 个样本。它启动的 MPS 在第三组正式运行期间保持可用；首次 MPS 启动时间和进程信息保存为 `mps_ownership.txt`、`mps_processes_at_start.txt`。服务退出时出现的 multiprocessing resource-tracker 警告发生在请求与审计完成后，原始日志完整保留。
