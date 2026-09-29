# 排除的预实验与启动记录

本目录全部排除在正式统计之外。第一、二组 host/L2 对照在 `../host_results/`，第三组在 `../third_group_results/`，三组汇总在 `../three_group_results/`。

## Host/L2 预实验

- `host_l2_pilot/`：初始化审计使用了本分支已移除的 scheduler rank 属性，启动失败；没有有效测量。
- `host_l2_pilot_v2/`：首次检查发现 host pool 审计读取了错误属性，正式 pilot 在请求前中止。独立 `control_probe/` 随后验证真实 HiCache L2 回读及 8-rank load-back 计数，但因该服务的初始化审计不完整，全部排除。最终审计已改为读取 UnifiedRadixCache 的 `host_pool_group`。
- `flexkv_host_pilot/`：验证 7936-token 快照、每请求 Main/Indexer 7936 slots 和 SWA 256 slots 的 H2D，以及默认 35072-token GPU SWA 容量。其 1 个 warmup 和 2 个试测 batch 仅用于验证协议，不进入正式 ABBA 数据。

正式 host 实验只读取 `host_results/` 下四个指定 block。全部运行后重新从请求、响应、metrics、H2D 日志和控制记录核验。

## 第三组预实验

`bounded_on_flexkv_pilot/` 验证开启 encoder bounded replay 与 FlexKV 同时运行：八个 rank 均使用 `FlexKVRadixCache`，FlexKV 无 SWA host pool；每个第二遍请求 Main/Indexer 命中 7936 host token、device=0，H2D 为 `slots=7936, swa_slots=0`，每 batch 额外 replay=1024。全部输出与冷请求相同且检索正确。

该 pilot 含 1 对热身和 2 对试测，不计入第三组正式两轮/20 个样本。它启动的 MPS 在第三组正式运行期间保持可用；首次 MPS 启动时间和进程信息保存为 `mps_ownership.txt`、`mps_processes_at_start.txt`。服务退出时出现的 multiprocessing resource-tracker 警告发生在请求与审计完成后，原始日志完整保留。
