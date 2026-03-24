# Exp5 速度基准报告

## 口径修正

- 本报告只统计推理速度与资源占用，不再统计准确率。
- 当前参与方案均为未针对本任务微调的基座模型，因此不使用 `Action Match Rate`、`Exact Match Rate` 等任务指标。
- 所有方案都按 `GPU-only` 标准执行，无法在 GPU 上初始化时直接失败，不允许静默回退到 CPU。
- 当前结果只解释为“本地部署栈”的端到端表现，不将其写成同构量化条件下的纯推理引擎优劣结论。
- `Transformers_BNB_4bit` 与 `LlamaCPP_GGUF_Q4_K_M` 同属 4bit 部署方案，但量化格式分别是 `bitsandbytes 4bit` 与 `GGUF Q4_K_M`，实现路径不同。
- 本地 `ExLlamaV2_EXL2_LocalAsset` 的 README 标注 `Bits 8.0`，因此它可以纳入统一 GPU 基准，但不应写成严格同构 4bit 主结论。

## 当前结果

| 方案 | Backend | Quantization | Avg Latency (s) | P50 (s) | P95 (s) | Sample Throughput (samples/s) | Avg RSS (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Transformers_BNB_4bit | transformers | 4bit | 5.5530 | 5.7369 | 6.6024 | 0.1801 | 1788.28 |
| LlamaCPP_GGUF_Q4_K_M | llama.cpp | gguf_q4_k_m | 7.0241 | 7.2612 | 8.7073 | 0.1424 | 4272.85 |

## 解读

- 从当前报告看，`Transformers_BNB_4bit` 的平均时延和尾部时延都低于 `LlamaCPP_GGUF_Q4_K_M`。
- 两者输出长度差异不大，但 `llama.cpp` 的总体 token 吞吐更低，因此整体耗时更长。
- 该现象更适合解释为“当前部署栈默认参数下的整体性能差异”，而不是“同一种 4bit 条件下 llama.cpp 天生更慢”。

## 说明

- 目录中的 `Transformers_4bit_accuracy.json` 属于旧版实验口径遗留产物，不再纳入 Exp5 当前结论。
- `ExLlamaV2_EXL2_LocalAsset` 已纳入当前 Exp5 基准框架，但本仓库中的静态结果表尚未重跑出对应数值。
- 当前环境若未安装 `llama-cpp-python` 或 `exllamav2`，对应方案会在 benchmark 阶段以依赖缺失状态失败。
