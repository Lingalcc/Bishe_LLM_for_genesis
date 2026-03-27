# 实验09 Exp5：本地部署栈速度基准

本实验用于在 `8GB VRAM` 消费级显卡、`Batch Size = 1` 的具身智能场景下，对三种本地推理引擎进行统一的**速度与资源占用**评测。

本次修订后的口径有两点：

- 只统计推理速度与资源指标，不再统计 `Action Match Rate`、`Exact Match Rate` 等任务准确率；
- 所有方案都按 `GPU-only` 标准执行，无法在 GPU 上初始化时直接失败，不允许静默回退到 CPU；
- 结果解释层级限定为“部署栈端到端表现”，不将结论写成“同构量化条件下的纯推理引擎优劣”。

原因也很直接：当前参与对比的模型都没有针对本任务做微调；同时尽管三组方案都在同一套 `GPU-only` benchmark 参数下运行，但其量化格式并不完全同构。

## 实验目标

- 比较本地部署栈在端侧场景下的端到端时延
- 观察尾部时延与吞吐差异
- 对比显存与进程内存占用
- 在 `OOM`、依赖缺失、模型缺失等情况下不中断整个实验流程

## 当前实验矩阵

- `Transformers_BNB_4bit`
- `LlamaCPP_GGUF_Q4_K_M`

说明：

- 三组都属于当前仓库可直接复现的 GPU 本地部署方案；
- benchmark 统一使用相同的 prompts、batch size、num samples、max_new_tokens、max_model_len；
- `Transformers_BNB_4bit` 与 `vLLM_BNB_4bit` 共享同一基座模型与运行时 4bit 设定；
- `Transformers_BNB_4bit` 与 `LlamaCPP_GGUF_Q4_K_M` 都属于 `4bit` GPU 部署方案，但量化格式不同；
- 若后续要研究“纯引擎差异”，需要进一步统一模型格式、量化格式和关键运行参数。

## 核心指标

- `Avg Latency (s)`
- `P50 Latency (s)`
- `P95 Latency (s)`
- `Sample Throughput (samples/s)`
- `Peak VRAM (MB)`
- `Avg Process RSS (MB)`

## 目录结构

- [run_exp5_engine_benchmark.py](/home/lin/Bishe_LLM_for_genesis/experiments/09_exp5_engine/run_exp5_engine_benchmark.py)
  - 速度基准主脚本
- [prompts/default_prompts.json](/home/lin/Bishe_LLM_for_genesis/experiments/09_exp5_engine/prompts/default_prompts.json)
  - benchmark 默认提示词
- `reports/`
  - Markdown、CSV、JSON、PNG 输出目录
- `logs/`
  - 每组 benchmark 子进程日志
- `.cache/`
  - Matplotlib 等临时缓存目录

## 运行方法

```bash
python experiments/09_exp5_engine/run_exp5_engine_benchmark.py
```

若希望自动补装推理引擎依赖：

```bash
python experiments/09_exp5_engine/run_exp5_engine_benchmark.py --auto-install-deps
```

若希望在模型缺失时自动从 Hugging Face 下载：

```bash
python experiments/09_exp5_engine/run_exp5_engine_benchmark.py \
  --auto-install-deps \
  --auto-download-missing-models
```

## 配置说明

主脚本顶部维护两块高扩展性配置区：

- `test_configs`
  - 定义当前参与比较的部署方案
- `MODEL_ARTIFACTS`
  - 定义基础模型与 GGUF 模型的本地路径和可选 Hugging Face 仓库 ID

说明：

- 当前仓库已存在 `model/Qwen_Qwen2.5-3B-Instruct`，因此 `Transformers_BNB_4bit` 与 `vLLM_BNB_4bit` 都可以直接复用该基础模型并通过 `bitsandbytes` 以运行时 `4bit` 方式加载；
- `model/Qwen_Qwen2.5-3B-Instruct-GGUF` 可用于 `LlamaCPP_GGUF_Q4_K_M`；
- 当前实验不再把 `accuracy` 纳入结论，因此也不再生成 `<CaseName>_accuracy.json`；
- 当前脚本会向 benchmark CLI 显式传入 `--require-gpu`，并把子进程 `CUDA_VISIBLE_DEVICES` 绑定到指定 GPU；
- 若后续补齐更统一的模型资产，再扩展到同构量化的严格对比会更合适。

## 输出产物

运行结束后会在 `experiments/09_exp5_engine/reports/` 下生成：

- `exp5_engine_speed_comparison.csv`
  - 全部部署方案的速度与资源汇总表
- `exp5_engine_speed_summary.json`
  - 结构化摘要，包含公平性说明
- `exp5_speed_report.md`
  - 人类可读的 Markdown 报告
- `exp5_engine_latency_bar.png`
  - 图1：端到端延迟对比
- `exp5_engine_throughput_bar.png`
  - 图2：样本吞吐对比
- `exp5_engine_memory_bar.png`
  - 图3：显存与进程内存占用对比
- `<CaseName>_benchmark.json`
  - 单组 benchmark 原始报告

## 解读建议

- 如果你关注交互体验，优先看 `Avg Latency` 与 `P95 Latency`；
- 如果你关注能不能在端侧稳定落地，优先看 `Peak VRAM`；
- 如果你想解释“为什么某个方案更慢”，需要继续下钻具体运行参数，而不是直接把当前结果归因为某个引擎天然更快或更慢。
