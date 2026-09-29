# SWA replay 性能实验

## 最新：三组真实 host/L2 命中实验

新增第三组已完成：开启 encoder bounded replay，Main KV 与 Indexer K 从 **FlexKV 管理的 host memory** 命中，SWA 采用 FP8 有界重建、不从 host 回迁。固定 8×B200、TP8/EP8、batch=8、8192 输入 token/请求，Main/Indexer FP4、SWA FP8；**未扩大 paged SWA**。

每组 20 个有效第二遍 batch。计时包含 host lookup、H2D、普通尾部 prefill 和有界重放（若开启）；“输入 token/s”包含命中的逻辑前缀，完整请求输出 2 token。

| 组别 | Main/Indexer 来源 | SWA 恢复方式 | 首 token 输入 token/s | 完整请求输入 token/s |
|---|---|---|---:|---:|
| 第一组 | 原生 HiCache L2 | 有界重放 | 41,412.1 | 37,667.9 |
| 第二组 | FlexKV host | FlexKV host snapshot 回迁 | 191,093.5 | 137,662.0 |
| 第三组 | FlexKV host | 有界重放 | **40,493.6** | **36,953.1** |

第三组相对第一组，首 token/完整请求吞吐分别低 **2.22% / 1.90%**；第二组分别为第三组的 **4.719× / 3.725×**。第三组是后续追加的两次独立重启采样，不是三组交错随机实验，小幅差异不做显著性断言。

全部第二遍请求均验证 **device=0、host=7936 token/请求**。第三组 H2D 的 `swa_slots=0`，每 batch SWA replay=1024，未使用 SWA host pool。三组含首次与热身共 1248 条请求均复核通过，输出 token IDs 和检索答案一致。其余 256 token/请求正常 prefill；不声称 8192/8192 token 全命中。

- 三组报告、数据表和校验：`three_group_results/REPORT.md`、`three_group_results/summary.json`、`three_group_results/trials.csv`、`three_group_results/validation.json`。
- 第三组原始数据及冻结脚本：`third_group_results/`；原第一、二组 `host_results/` 保持不变。
- 第三组协议/运行入口：`THIRD_EXPERIMENT.md`、`run_third_experiment.sh`；分析入口：`summarize_three_groups.py`。
- 第三组预跑单独保存在 `diagnostics/bounded_on_flexkv_pilot/`，不参与正式统计。
- 独立 SSE 吞吐复算、原始数据 SHA-256、源码及格式校验见 `validation/third_final_checks.log`；本次推理服务和确认无客户端的自建 MPS 已停止，8 卡显存均回到 0 MiB，清理证据见 `validation/third_mps_cleanup.log`。

```bash
bash /root/swa_replay_perf/run_third_experiment.sh \
  /root/swa_replay_perf/third-repro-$(date -u +%Y%m%dT%H%M%SZ) \
  /root/swa_replay_perf/host_results
```

## 原两组 host/L2 对照

第一组的 Main/Indexer 从 **HiCache L2** 命中；第二组的 Main/Indexer/SWA 从 **FlexKV host memory** 命中。协议与复现入口见 `HOST_EXPERIMENT.md`、`run_host_reproduction.sh`；这两组正式结果在 `host_results/REPORT.md`、`host_results/summary.json` 和 `host_results/trials.csv`。

每组 20 个正式 batch，均验证 `device=0, host=7936 token/请求`。第一组/第二组首 token 阶段有效输入吞吐为 **41,412.1 / 191,093.5 token/s**；完整 2-token 请求为 **37,667.9 / 137,662.0 token/s**。第二组分别为第一组的 **4.614× / 3.655×**，计时包含 host lookup 和 H2D。

下文及旧 `results/` 是 **GPU-only 历史对照**，不能作为上述 host 回读实验的结果。

## 原 GPU-only 实验

固定条件：DeepSeek-V4.1-Flash，8×B200，TP8/EP8，batch=8，输入 8192 token/request，Main KV 和 Indexer K 为 FP4，SWA KV 为 FP8。

对照组仅切换 `--enable-encoder-swa-bounded-replay`。每个样本独立执行 flush→首次填充→第二次重放，只计算第二次请求耗时。两次均固定输出 2 token，使用流式响应分别记录 8 条请求全部取得首 token 的时间和整批完成时间；不是长输出 decode 性能测试。

选择 2 token 是为了触发框架正常的 unfinished-request 缓存路径：只输出 1 token 会直接结束请求，未经过该路径的 SWA 窗口回收，默认池在 8 条 8K 请求之间发生逐出/抖动，第二组无法全命中。2-token pilot 已验证默认容量下 8 条请求全部命中 7936-token 可复用前缀。无效的 1-token 对照数据单独保留，不进入正式汇总；首 token 阶段的独立计时避免将额外 decode 混入 prefill 指标。

paged SWA 使用默认容量，不传入 `--swa-full-tokens-ratio` 或 `--swa-prefix-tails` 扩容参数。

量化配置使用 `SGLANG_DSV4_KV_LAYOUT=v41` 配合 `--kv-cache-dtype fp8_e4m3` 指定 FP8 SWA，使用 `SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp4` 指定 C1/C2 Main KV。V4.1 的低压缩率 Indexer K 自动采用 packed FP4，启动时检查其 `use_fp4_indexer=True` 和 68 bytes/token。物理布局以每个 worker 的 `layout` 和 `bytes_per_token` 为准；packed KV 底层存储是 `uint8`，不能只看通用 `dtype` 字段判断精度。

## 结果

完整报告生成在 `results/REPORT.md`，机器可读数据在 `results/summary.json` 和 `results/trials.csv`。

输入严格为 8192 token，但 SGLang 保留 next-token logits 的重算 token，并按 256-token page 对齐，因此第二次命中 7936 token、重算尾部 256 token。开启模式额外重放 128 token/request。脚本逐样本验证这些计数，不将 cache miss 当作缓存命中。

## 复现

```bash
bash /root/swa_replay_perf/run_experiments.sh \
  /root/swa_replay_perf/repro-$(date -u +%Y%m%dT%H%M%SZ)
```

默认复用 `/root/nvfp4-validation/venv`、`/root/sglang` 和 `/root/models/DeepSeek-V4.1-Flash`。可通过 `PYTHON_BINARY`、`SGLANG_DIRECTORY`、`MODEL_DIRECTORY`、`SERVER_PORT` 指定运行环境。注意该 venv 的 native FlashMLA 包包含本地 V4.1 FP4 layout 支持；当前环境与源码/native 指纹保存在 `results/environment/`，不可假定换成任意发布 wheel 结果相同。

当前 SGLang 源码另外应用了 `compatibility.patch`：仅当 FlashMLA 元数据预计算超出 48 KiB shared-memory 上限时，回退原生调度，解决 8K 冷填充崩溃。两组统一使用此补丁；它不改变 2048-token timed batch 或 128-token replay 的调度路径。旧源码复现前须先应用此补丁；失败日志与回归测试结果也保留在本目录。

启动顺序 A-B-B-A（A=开启，B=关闭），每个 block 排除 3 对 warmup，测量 10 对独立 cold→replay；每组共 20 个 measured batch。

## 文件

- `launch_server.sh`：支持历史 GPU-only 两模式，以及三种 host/L2 模式；分别设置 replay 和 host 后端。
- `server_entry.py`：仅在 KV pool 初始化时核验并记录全部 8 rank 的实际 FP4/FP8 layout，不包装 forward。
- `manage_server.py`：启动、等待、停止本实验自己的进程。
- `prepare_requests.py`、`requests.json`：生成并保存精确 token 输入和独立 cache salt。
- `benchmark.py`：逐样本保存完整响应、metrics 前后值和 prefill 日志；断言命中数量、重放数量、实际 batch=8。
- `collect_environment.py`：采集版本、硬件、模型配置和代码指纹。
- `summarize.py`：仅汇总 measured second-pass 请求，生成中文报告与 CSV。
- `validate_artifacts.py`：从原始响应、metrics 和日志重新验证全部请求及计时公式。

手工中断后可以清理：

```bash
/root/nvfp4-validation/venv/bin/python /root/swa_replay_perf/manage_server.py stop
```
