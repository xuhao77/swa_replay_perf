# HiCache L2 与 FlexKV host 命中对照

本协议替代旧的 GPU-only 对照。旧数据保留在 `results/`，不混入本次统计。

## 两组定义

共同条件：本地 DeepSeek-V4.1-Flash，8×B200，TP8/EP8，batch=8，8192 输入 token/请求；Main KV、Indexer K 使用 FP4，SWA KV 使用 FP8。每请求输出 2 token，prefill/decode CUDA Graph 均关闭，GPU 最大 FULL tokens=131072，最大并发=8。

- 第一组 `bounded_on_l2`：开启 encoder bounded replay，启用原生 HiCache L2（pinned host memory）。Main KV 和 Indexer K 从 L2 回读，SWA 在 GPU 上有界重建，不在 L2 缓存 SWA。
- 第二组 `bounded_off_host`：关闭 encoder bounded replay，启用 FlexKV CPU cache。Main KV、Indexer K 及 SWA 从 FlexKV 管理的 host memory 回读。FlexKV 使用 bulk/no-layerwise transfer；不启用 SSD、远端存储或 native HiCache。

第一组 HiCache ratio=2；第二组 FlexKV `cpu_cache_gb: 16`、`swa_multi_group: true`，SWA host pool 为 FlexKV 独立分配。这些是 **host** 容量，未扩大 GPU paged SWA；第二组 GPU SWA 仍应为默认 35072 token/worker。

本次正式四轮开始前，FlexKV pilot 已启动默认 CUDA MPS 服务；该服务贯穿四轮，各 block 的 `gpu_before.txt` 均有记录。复现入口 `run_host_reproduction.sh` 会在第一组前确保 MPS 可用，避免在干净环境中直到第二组才启动 MPS。它不改变原测量脚本、驱逐控制或 kernel。

## 每个样本

1. 使用独立的 per-request、per-trial cache salt，避免历史 host cache 或 batch 内共有前缀影响冷填充。
2. 清理旧 GPU cache，然后一次提交完整的 8 条请求。首次推理时临时设置 chunk=7936、每个 prefill batch 最多 1 条请求，使每条请求都经过 7936-token 边界；随后推理剩余 256 token 并输出 2 token。输入本身始终是 8192 token，没有额外提交缩短的 prefix 请求。
3. FlexKV 通过已有的 `SGLANG_FLEXKV_SWA_GRID_PAGES=31` 在该边界保存 SWA 快照。只存 8192-token turn-end 快照不足以满足第二遍 7936-token 可复用前缀的 SWA 需求；因此不能直接 flush 后假定 host 命中。
4. 排空全部 D2H；调用缓存现有的 `evict(EvictParams(...))` 仅驱逐 L1/GPU 条目，保留 host 数据；逐 TP rank 保存驱逐前后容量、锁和 host pool 状态。**不能用清空全部 HiCache 的 `/flush_cache` 代替这个步骤。**
5. 恢复 chunk=8192、prefill batch 上限=8，原样提交相同请求，只有追踪 ID 改变。第二遍实际 prefill batch 必须为 8，且每请求 `device=0, host=7936`；否则样本立即失败。
6. 第一组额外 SWA replay=8×128=1024 token；第二组 replay=0。两组均有 8×256=2048 token 的正常尾部重算。

第一组还核对全部 8 rank 的 HiCache load-back 计数，每 rank 每 batch 为 63488 token。第二组逐 request ID 检查 H2D launch 的 `slots=7936, swa_slots=256` 和成功完成记录，确认并非只回读 Main/Indexer 而遗漏 SWA。

冷填充、snapshot store、D2H 排空、L1 驱逐和配置切换均不计时。第二遍计时从发送 HTTP 请求前开始，**包含 L2 lookup、H2D 回迁、SWA replay（第一组）、尾部 prefill，以及相应的调度/传输开销**。

`host_control.py` 只在 scheduler 初始化时安装实验专用控制 RPC；仅在无推理请求时调用，复用现有 cache API，不包装 forward，也不修改 timed cache lookup 或 kernel。控制入口及 8 rank 操作记录均保留在实验目录。

## 统计口径

按 A-B-B-A 重启服务，A=HiCache L2+bounded replay，B=FlexKV host+关闭 replay。每个 block 丢弃 3 对 warmup，再记录 10 对 cold→replay；每组共 20 个正式 batch。

- 首 token 阶段有效输入吞吐：`batch数×8×8192 / 各batch最后一条首token的到达耗时之和`。
- 完整请求有效输入吞吐：分母换为每条均产生 2 token 后的整批完成耗时。
- 全部 SSE 事件保存；第一个生成事件必须为 completion_tokens=1，最终事件为 2。
- 这是包含命中前缀的逻辑输入吞吐，不是长输出 decode 吞吐，也不是实际重算 8192 token 的算力指标。

本对照按用户要求同时改变 **缓存后端及 SWA 恢复方式**，不是只切换 bounded replay 的单变量消融，不能把全部差异单独归因于 replay。

## 运行与验证

```bash
bash /root/swa_replay_perf/run_host_reproduction.sh \
  /root/swa_replay_perf/host-repro-$(date -u +%Y%m%dT%H%M%SZ)
```

脚本拒绝覆盖已存在的结果目录。所有输入、原始请求/SSE、native metrics、H2D 日志、L1 驱逐记录、启动参数、硬件/源码指纹、代码快照及校验结果保存到该目录。完成后自动停止所启动的服务。

MPS 是共享服务，FlexKV 自身也不在 shutdown 时自动关闭它。复现脚本不会误杀已有的共享 MPS；若本次是新启动的服务，须在 `get_client_list <server PID>` 确认没有任何客户端后，才可执行 `echo quit | nvidia-cuda-mps-control`。本次实验完成后已单独确认并清理 pilot 启动的空闲 MPS。

环境沿用 `/root/nvfp4-validation/venv`、`/root/sglang`、`/root/FlexKV` 和本地模型。8K 冷填充所需的 FlashMLA shared-memory 容量保护补丁仍为 `compatibility.patch`，两组一致；相关 kernel 回归测试已通过 42 项。
