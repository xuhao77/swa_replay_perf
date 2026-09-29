# SWA replay 性能实验

## 三组 host/L2 命中实验

本实验比较三种从主机内存缓存恢复 KV 的方案：第一组从原生 HiCache L2 回读 Main KV / Indexer K，并通过有界重放重建 SWA；第二组从 FlexKV host 回读 Main KV / Indexer K / SWA；第三组从 FlexKV host 回读 Main KV / Indexer K，但仍通过有界重放重建 SWA。这里的 host/L2 均指主机内存中的缓存。

三组固定使用 DeepSeek-V4.1-Flash、8×B200、TP8/EP8，Main/Indexer FP4、SWA FP8；**未扩大 paged SWA**。每个 batch 同时发送 8 条请求，每条输入 8192 token、输出 2 token。先完成冷填充、写入 host cache 和 GPU 缓存驱逐，再提交相同输入的第二遍请求；每组只统计 20 个有效第二遍 batch。

### 吞吐指标说明

下表两列都是**有效输入吞吐**，单位为输入 token/s：用全部计时 batch 的输入 token 总数，除以这些 batch 的耗时之和。两列的分子相同，区别只在于何时停止计时：

- **按首 token 耗时计算**：从提交第二遍 batch 前开始，到 **8 条请求都收到第一个输出 token** 为止，反映等待首个输出这一阶段的输入吞吐。
- **按请求总耗时计算**：从同一时刻开始，到 **8 条请求全部完成**为止；每条请求都输出 2 token，因此还包含首 token 之后的生成与响应完成时间。

计算公式为 `有效输入吞吐 = 20 × 8 × 8192 / 20 个 batch 对应耗时之和（秒）`，不是逐请求吞吐的简单平均。计时包含 host lookup、主机到 GPU 的传输（H2D）、普通尾部 prefill 和有界重放（若开启），不包含首次冷填充与缓存驱逐。

**注意：这不是输出 token 的生成速度，也不是 GPU 重新计算全部输入的速度。** 分子包含从 host cache 命中的输入前缀；第二列虽然计时到输出完成，分子仍是输入 token 数，而不是输出 token 数。

### 三组结果与对比

| 组别 | Main/Indexer 来源 | SWA 恢复方式 | 有效输入吞吐（按首 token 耗时，token/s） | 有效输入吞吐（按请求总耗时，token/s） |
|---|---|---|---:|---:|
| 第一组 | 原生 HiCache L2 | 有界重放 | 41,412.1 | 37,667.9 |
| 第二组 | FlexKV host | FlexKV host snapshot 回迁 | 191,093.5 | 137,662.0 |
| 第三组 | FlexKV host | 有界重放 | **40,493.6** | **36,953.1** |

以下对比分别对应表中的“按首 token 耗时”和“按请求总耗时”两列：

- 第二组的有效输入吞吐分别为第一组的 **4.614× / 3.655×**。
- 第三组的有效输入吞吐比第一组分别低 **2.22% / 1.90%**。
- 第二组的有效输入吞吐分别为第三组的 **4.719× / 3.725×**。

第一、二组按 A-B-B-A 顺序重启采样；第三组是后续追加的两次独立重启采样。三组各有 20 个有效 batch，但不是三组交错随机实验，小幅差异不做显著性断言。

全部第二遍请求均验证 **device=0、host=7936 token/请求**。第三组 H2D 的 `swa_slots=0`，每 batch SWA replay=1024，未使用 SWA host pool。三组含首次与热身共 1248 条请求均复核通过，输出 token IDs 和检索答案一致。其余 256 token/请求正常 prefill；不声称 8192/8192 token 全命中。

### 数据、协议与校验

- 三组报告、数据表和校验：`three_group_results/REPORT.md`、`three_group_results/summary.json`、`three_group_results/trials.csv`、`three_group_results/validation.json`；分析入口为 `summarize_three_groups.py`。
- 第一、二组原始数据及归档脚本：`host_results/`；两组报告与汇总为 `host_results/REPORT.md`、`host_results/summary.json`、`host_results/trials.csv`。
- 第一、二组协议/运行入口：`HOST_EXPERIMENT.md`、`run_host_reproduction.sh`。
- 第三组原始数据及归档脚本：`third_group_results/`；协议/运行入口为 `THIRD_EXPERIMENT.md`、`run_third_experiment.sh`。
- 第三组预跑单独保存在 `diagnostics/bounded_on_flexkv_pilot/`，不参与正式统计。
- 独立 SSE 吞吐复算、原始数据 SHA-256、源码及格式校验见 `validation/third_final_checks.log`；本次推理服务和确认无客户端的自建 MPS 已停止，8 卡显存均回到 0 MiB，清理证据见 `validation/third_mps_cleanup.log`。

## 运行环境与复现

先复现第一、二组，再使用该基线运行第三组：

```bash
baseline_directory=/root/swa_replay_perf/host-repro-$(date -u +%Y%m%dT%H%M%SZ)
bash /root/swa_replay_perf/run_host_reproduction.sh "$baseline_directory"
bash /root/swa_replay_perf/run_third_experiment.sh \
  /root/swa_replay_perf/third-repro-$(date -u +%Y%m%dT%H%M%SZ) \
  "$baseline_directory"
```

第三组入口及参数见 `THIRD_EXPERIMENT.md`。默认复用 `/root/nvfp4-validation/venv`、`/root/sglang`、`/root/FlexKV` 和 `/root/models/DeepSeek-V4.1-Flash`；可通过 `PYTHON_BINARY`、`SGLANG_DIRECTORY`、`MODEL_DIRECTORY`、`SERVER_PORT` 指定运行环境。该 venv 的 native FlashMLA 包包含本地 V4.1 FP4 layout 支持；实际环境与源码/native 指纹保存在 `host_results/environment/` 和 `third_group_results/environment/`，不能假定任意发布 wheel 都能复现。

三组使用相同的 `compatibility.patch`：仅当 FlashMLA 元数据预计算超出 48 KiB shared-memory 上限时回退原生调度，支持 8K 冷填充；不改变 2048-token timed batch 或 128-token replay 的调度路径。旧源码复现前须先应用该补丁，相关 kernel 回归记录保留在 `validation/`。

## 文件

- `launch_server.sh`、`manage_server.py`：仅支持 `bounded_on_l2`、`bounded_off_host`、`bounded_on_flexkv` 三种 host/L2 模式，管理本实验自己的服务进程。
- `server_entry.py`、`host_control.py`：记录 8-rank FP4/FP8 布局，提供非计时阶段的 host I/O 排空和 L1 驱逐控制。
- `prepare_requests.py`、`requests.json`：生成并保存精确 token 输入和独立 cache salt。
- `benchmark_host.py`：执行 host/L2 cold→evict→replay 协议；`benchmark.py` 仅保留共用的请求、SSE、metrics 和日志校验函数，不提供独立实验入口。
- `collect_environment.py`：采集版本、硬件、模型配置和源码指纹。
- `summarize_host.py`、`summarize_three_groups.py`：生成两组和三组报告、CSV 及校验清单；`summarize.py` 仅保留共用统计函数。
- `validate_host_artifacts.py`：从原始请求、响应、metrics、H2D 和驱逐记录重新验证 host/L2 实验。

手工中断后可以清理本实验服务：

```bash
/root/nvfp4-validation/venv/bin/python /root/swa_replay_perf/manage_server.py stop
```

## 归档与校验

仓库仅保留 host/L2 实验及其预实验。归档 `code/` 已同步裁剪无关实验入口；原始请求、响应、SSE、计时、池审计和吞吐统计数值均未修改。`environment*/environment.json` 中保留的源码指纹仍反映采样时内容，不代表裁剪后脚本的散列；当前文件完整性以更新后的 `artifacts.sha256` 和 `three_group_results/source_artifacts.sha256` 为准。

重新采样时，必须先用当前脚本生成新的第一、二组基线，再用同一套脚本运行第三组；不要将新运行直接与归档 `host_results/` 混用，否则严格源码指纹检查会拒绝该比较。随仓库保存的三组数据仍可用于离线复核。

`validation/` 中的运行日志保留采样时的审计事实；本次仓库清理后的离线校验见 `validation/repository_checks.log`。实际 GPU 实验仍需上述 8×B200 环境。
