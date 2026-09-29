# SWA replay 性能实验

## 三组 host/L2 命中实验

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
- 第三组原始数据及归档脚本：`third_group_results/`；第一、二组 `host_results/` 的原始测量数据保持不变。
- 第三组协议/运行入口：`THIRD_EXPERIMENT.md`、`run_third_experiment.sh`；分析入口：`summarize_three_groups.py`。
- 第三组预跑单独保存在 `diagnostics/bounded_on_flexkv_pilot/`，不参与正式统计。
- 独立 SSE 吞吐复算、原始数据 SHA-256、源码及格式校验见 `validation/third_final_checks.log`；本次推理服务和确认无客户端的自建 MPS 已停止，8 卡显存均回到 0 MiB，清理证据见 `validation/third_mps_cleanup.log`。

```bash
baseline_directory=/root/swa_replay_perf/host-repro-$(date -u +%Y%m%dT%H%M%SZ)
bash /root/swa_replay_perf/run_host_reproduction.sh "$baseline_directory"
bash /root/swa_replay_perf/run_third_experiment.sh \
  /root/swa_replay_perf/third-repro-$(date -u +%Y%m%dT%H%M%SZ) \
  "$baseline_directory"
```

## 第一、二组 host/L2 对照

第一组的 Main/Indexer 从 **HiCache L2** 命中；第二组的 Main/Indexer/SWA 从 **FlexKV host memory** 命中。协议与复现入口见 `HOST_EXPERIMENT.md`、`run_host_reproduction.sh`；这两组正式结果在 `host_results/REPORT.md`、`host_results/summary.json` 和 `host_results/trials.csv`。

每组 20 个正式 batch，均验证 `device=0, host=7936 token/请求`。第一组/第二组首 token 阶段有效输入吞吐为 **41,412.1 / 191,093.5 token/s**；完整 2-token 请求为 **37,667.9 / 137,662.0 token/s**。第二组分别为第一组的 **4.614× / 3.655×**，计时包含 host lookup 和 H2D。

## 运行环境与复现

先复现第一、二组，再使用该基线运行第三组：

```bash
bash /root/swa_replay_perf/run_host_reproduction.sh \
  /root/swa_replay_perf/host-repro-$(date -u +%Y%m%dT%H%M%SZ)
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
