# 实验09 Exp5：7 组推理部署性能与精度对比

本实验重写后的目标，是在同一套具身任务评测口径下，对以下 7 组部署方案做统一对比：

- `Transformers_16bit`
- `Transformers_8bit`
- `Transformers_4bit`
- `vLLM_16bit`
- `vLLM_8bit`
- `vLLM_4bit`
- `vLLM_AWQ`

实验同时统计两类结果：

- 推理性能：平均延迟、P95 延迟、样本吞吐、Token 吞吐、峰值显存、进程 RSS
- 任务精度：`Parse OK Rate`、`Exact Match Rate`、`Action Match Rate`

如果某一组在初始化、benchmark 或 accuracy 阶段出现 `OOM`、依赖缺失、模型缺失或其他失败，脚本会把状态记录到汇总表里，然后继续执行下一组，不会中断整轮实验。

## 实验口径

- 前 6 组都基于同一份任务模型：`model/qwen2.5-3b-genesis-merged`
- 第 7 组 `vLLM_AWQ` 使用压缩目录：`model/qwen2.5-3b-genesis-merged-awq`
- benchmark 与 accuracy 分开执行，避免前一组残留显存影响后一组
- 所有子进程都绑定同一块 GPU，并强制 `GPU-only`
- `vLLM 8bit/4bit` 改为读取真正的 bitsandbytes 预量化目录，默认放在 `experiments/09_exp5_engine/.cache/prequantized_models/`

说明：

- `Transformers 8bit/4bit` 与 `vLLM 8bit/4bit` 共享同一份 merged 模型资产，只改变运行后端和量化加载路径
- `vLLM 8bit/4bit` 不再通过“只改 `config.json`”的方式伪装成预量化模型，而是先导出真正的 BNB 预量化权重目录
- `vLLM_AWQ` 属于预压缩部署路径，与 bitsandbytes 预量化不是完全同构的方案，因此解读时应视为“部署方案对比”

## 目录结构

- [run_exp5_engine_benchmark.py](/home/lin/Bishe_LLM_for_genesis/experiments/09_exp5_engine/run_exp5_engine_benchmark.py)
  - 主实验脚本
- [export_prequantized_bnb_model.py](/home/lin/Bishe_LLM_for_genesis/experiments/09_exp5_engine/export_prequantized_bnb_model.py)
  - 将 merged 模型导出为 bitsandbytes 8bit / 4bit 预量化目录
- [prompts/default_prompts.json](/home/lin/Bishe_LLM_for_genesis/experiments/09_exp5_engine/prompts/default_prompts.json)
  - benchmark 默认 prompts
- `reports/`
  - CSV / JSON / Markdown / PNG 汇总输出
- `logs/`
  - 每组 benchmark / accuracy 日志
- `.cache/`
  - Matplotlib 缓存与 vLLM bitsandbytes 预量化模型目录

## 运行方法

基础运行：

```bash
python experiments/09_exp5_engine/run_exp5_engine_benchmark.py
```

如果你希望由主脚本在发现 `vLLM_8bit/4bit` 目录缺失时自动导出预量化模型，可以加上：

```bash
python experiments/09_exp5_engine/run_exp5_engine_benchmark.py --auto-export-bnb-models
```

如果希望自动补装依赖：

```bash
python experiments/09_exp5_engine/run_exp5_engine_benchmark.py --auto-install-deps
```

如果当前环境中的 `vLLM / transformers / compressed-tensors` 组合已经在你机器上实测通过，但仓库的保守检查仍拦截了运行，可以手动跳过：

```bash
python experiments/09_exp5_engine/run_exp5_engine_benchmark.py --skip-vllm-compat-check
```

如果模型缺失并且你已经配置了下载源，可以打开自动下载：

```bash
python experiments/09_exp5_engine/run_exp5_engine_benchmark.py \
  --auto-install-deps \
  --auto-download-missing-models
```

如果你只想手动准备 `vLLM_8bit/4bit` 的预量化目录，再单独补跑这两组，可以先执行：

```bash
python experiments/09_exp5_engine/export_prequantized_bnb_model.py \
  --source-model-dir model/qwen2.5-3b-genesis-merged \
  --output-dir experiments/09_exp5_engine/.cache/prequantized_models/qwen2.5-3b-genesis-merged-bnb-8bit \
  --mode 8bit
```

```bash
python experiments/09_exp5_engine/export_prequantized_bnb_model.py \
  --source-model-dir model/qwen2.5-3b-genesis-merged \
  --output-dir experiments/09_exp5_engine/.cache/prequantized_models/qwen2.5-3b-genesis-merged-bnb-4bit \
  --mode 4bit
```

导出完成后，再只跑缺失的两组：

```bash
python experiments/09_exp5_engine/run_exp5_engine_benchmark.py \
  --skip-vllm-compat-check \
  --case-names vLLM_8bit,vLLM_4bit \
  --reuse-existing-summary experiments/09_exp5_engine/reports/exp5_engine_summary.json
```

## 关键参数

- `--benchmark-num-samples`
  - benchmark 样本数，默认 `200`
- `--accuracy-num-samples`
  - accuracy 样本数，默认 `200`
- `--batch-size`
  - 默认 `1`
- `--max-new-tokens`
  - 默认 `128`
- `--max-model-len`
  - 默认 `2048`
- `--gpu-id`
  - 指定监控与运行用的 GPU 编号
- `--sleep-seconds`
  - 每组执行后的冷却时间，默认 `15`
- `--auto-export-bnb-models`
  - 当 `vLLM_8bit/4bit` 预量化目录缺失时，自动调用导出脚本补齐
- `--force-reexport-bnb-models`
  - 强制重导 `vLLM_8bit/4bit` 预量化目录

## 输出产物

运行结束后会在 `experiments/09_exp5_engine/reports/` 下生成：

- `exp5_engine_comparison.csv`
  - 7 组方案的结构化汇总表
- `exp5_engine_summary.json`
  - 包含状态、最佳方案和公平性说明的 JSON 摘要
- `exp5_engine_report.md`
  - 人类可读的实验报告
- `exp5_engine_latency_bar.png`
  - 图1：7 组平均延迟对比
- `exp5_engine_throughput_bar.png`
  - 图2：7 组 Token 吞吐对比
- `exp5_engine_accuracy_bar.png`
  - 图3：7 组精度对比
- `exp5_engine_memory_bar.png`
  - 图4：7 组显存对比
- `<CaseName>_benchmark.json`
  - 单组 benchmark 原始结果
- `<CaseName>_accuracy.json`
  - 单组 accuracy 原始结果

## 建议解读顺序

- 如果你关心交互速度，先看 `Benchmark Avg Latency (s)` 与 `Benchmark Token Throughput (tokens/s)`
- 如果你关心模型能不能稳定落地，先看 `Benchmark Peak VRAM (MB)` 和状态字段是否为 `oom`
- 如果你关心任务质量，先看 `Exact Match Rate` 与 `Action Match Rate`
- 如果同一方案速度很好但 accuracy 很差，说明它更适合做部署优化参考，而不一定适合直接作为最终任务方案
