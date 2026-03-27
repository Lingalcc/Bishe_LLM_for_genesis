# Exp5 速度基准报告

## 口径修正

- 本报告只统计三种推理引擎的速度与资源占用，不再统计准确率。
- 参与方案均为未针对当前任务微调的基座模型，因此不使用 Action Match Rate、Exact Match 等任务指标。
- 当前结果仅解释为“本地部署栈”的端到端表现，不将其写成同构量化下的纯推理引擎优劣结论。
- Exp5 在执行层面强制 GPU-only：子进程会绑定 `CUDA_VISIBLE_DEVICES`，并向 benchmark CLI 显式传入 `--require-gpu`。
- 当前实验矩阵固定为三种引擎：`transformers`、`vllm`、`llama.cpp`。
- 三组方案统一使用相同的 prompts、batch size、num samples、max_new_tokens 和 max_model_len。
- `vLLM_BNB_4bit` 与 `Transformers_BNB_4bit` 共享同一基座模型与 bitsandbytes 4bit 设定，适合观察不同部署栈的端到端差异。
- `Transformers_BNB_4bit` 与 `LlamaCPP_GGUF_Q4_K_M` 同属 4bit 部署方案，但底层量化格式分别为 `bitsandbytes 4bit` 与 `GGUF Q4_K_M`，属于不同实现路径。

## 统计指标

- `Avg Latency (s)`：平均端到端时延
- `P50 Latency (s)`：中位数时延
- `P95 Latency (s)`：尾部时延
- `Sample Throughput (samples/s)`：样本吞吐
- `Peak VRAM (MB)`：通过 `nvidia-smi` 采样得到的峰值显存
- `Avg Process RSS (MB)`：进程常驻内存均值

## 当前结果

| 方案 | Backend | Quantization | 量化备注 | Avg Latency (s) | P50 (s) | P95 (s) | Samples/s | Peak VRAM (MB) | Avg RSS (MB) | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Transformers_BNB_4bit | transformers | 4bit | 运行时 4bit（bitsandbytes NF4） | 4.9635 | 5.3031 | 5.6315 | 0.2015 | 4546.00 | 1789.76 | success |
| vLLM_BNB_4bit | vllm | 4bit | 运行时 4bit（vLLM bitsandbytes） | 1.4559 | 1.5972 | 1.6209 | 0.6868 | 7910.00 | 1145.85 | success |
| LlamaCPP_GGUF_Q4_K_M | llama.cpp | gguf_q4_k_m | 离线量化 GGUF Q4_K_M | 6.8569 | 7.3462 | 8.0597 | 0.1458 | 507.00 | 4204.84 | success |

## 解读建议

- 如果关注交互响应，优先看 `Avg Latency` 与 `P95 Latency`。
- 如果关注端侧落地约束，优先看 `Peak VRAM` 是否接近 `8GB` 上限。
- 如果需要进一步追究“为什么某个部署栈更慢”，应继续下钻具体运行参数，例如 `llama.cpp` 的 `n_batch`、线程数、GPU offload 策略，或 `vllm` 的 `gpu_memory_utilization` 与 `max_model_len`，而不是仅凭当前报告直接归因到引擎本身。
