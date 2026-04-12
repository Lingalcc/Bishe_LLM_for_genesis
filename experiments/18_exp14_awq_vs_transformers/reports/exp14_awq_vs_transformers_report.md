# Exp14 AWQ 与 Transformers 对比实验报告

## 实验目标

- 对比 `Transformers 16bit` 基线与 `vLLM + AWQ` 压缩目录在同一任务上的速度、显存和准确率表现。
- AWQ 模型来自 `llmcompressor` 导出的 `compressed-tensors` 目录，由项目内 vLLM 兼容层自动识别并加载。

## 对比矩阵

- `Transformers_16bit`：未量化基线。
- `vLLM_AWQ_CompressedTensors`：AWQ 压缩部署方案。

## 速度结果

| 方案 | Backend | 载入格式 | Avg Latency (s) | P50 (s) | P95 (s) | Samples/s | Tokens/s | Peak VRAM (MB) | 状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Transformers_16bit | transformers | HF Safetensors | 2.0358 | 1.7350 | 3.7706 | 0.4912 | 26.6728 | 7121.00 | success |
| vLLM_AWQ_CompressedTensors | vllm | Compressed Tensors (AWQ) | 0.7917 | 0.6613 | 1.6085 | 1.2631 | 77.2992 | 7804.00 | success |

## 精度结果

| 方案 | Parse OK | Exact Match | Action Match | Accuracy Avg Latency (s) | Accuracy Tokens/s | Accuracy Max VRAM (MB) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Transformers_16bit | 1.0000 | 0.4000 | 0.6500 | 2.8639 | 28.8966 | 6040.00 | success |
| vLLM_AWQ_CompressedTensors | 1.0000 | 0.3500 | 0.7500 | 0.9939 | 77.9307 | 0.00 | success |

## 结果分析

- 速度最优方案为 `vLLM_AWQ_CompressedTensors`，benchmark 平均延迟为 `0.7917s`。
- `Exact Match` 最高方案为 `Transformers_16bit`，精确匹配率为 `0.4000`。
- `Action Match` 最高方案为 `vLLM_AWQ_CompressedTensors`，动作匹配率为 `0.7500`。
- 如果速度领先和准确率领先不是同一个方案，就说明当前部署仍存在明确权衡。

## 解读建议

- 如果你关注部署时延，优先看 `Benchmark Avg Latency (s)` 和 `Benchmark Token Throughput (tokens/s)`。
- 如果你关注任务可用性，优先看 `Exact Match Rate` 和 `Action Match Rate`。
- 当前 AWQ 目录实际由 `llmcompressor` 导出，因此这里的结论应理解为“AWQ 压缩部署方案 vs Transformers 基线”的端到端对比。
