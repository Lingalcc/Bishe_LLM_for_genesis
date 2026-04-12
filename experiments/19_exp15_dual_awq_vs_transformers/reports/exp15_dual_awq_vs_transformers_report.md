# Exp15 LoRA4 / Top18Rank8 的 AWQ 与 Transformers 四组对比实验报告

## 实验目标

- 同时比较两类模型家族：`LoRA rank4 merged` 与 `Top18 rank8 merged`。
- 每个模型家族各跑两种部署：`Transformers 16bit` 与 `vLLM + AWQ`。
- 关注速度、显存与准确率，避免覆盖既有 `EXP14` 结果。

## 对比矩阵

- `LoRA4_Transformers_16bit`：LoRA4 merged 的未量化基线。
- `LoRA4_vLLM_AWQ`：LoRA4 merged 的 AWQ 压缩部署方案。
- `Top18Rank8_Transformers_16bit`：Top18 rank8 merged 的未量化基线。
- `Top18Rank8_vLLM_AWQ`：Top18 rank8 merged 的 AWQ 压缩部署方案。

## 速度结果

| 方案 | 模型家族 | Backend | 载入格式 | Avg Latency (s) | Tokens/s | Peak VRAM (MB) | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| LoRA4_Transformers_16bit | LoRA Rank4 Merged | transformers | HF Safetensors | 1.9884 | 27.3089 | 7744.00 | success |
| LoRA4_vLLM_AWQ | LoRA Rank4 Merged | vllm | Compressed Tensors (AWQ) | 0.7990 | 76.5912 | 7808.00 | success |
| Top18Rank8_Transformers_16bit | Top18 Rank8 Merged | transformers | HF Safetensors | 2.3045 | 26.9692 | 6742.00 | success |
| Top18Rank8_vLLM_AWQ | Top18 Rank8 Merged | vllm | Compressed Tensors (AWQ) | 0.6836 | 82.9447 | 7868.00 | success |

## 精度结果

| 方案 | 模型家族 | Parse OK | Exact Match | Action Match | Accuracy Avg Latency (s) | Accuracy Tokens/s | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA4_Transformers_16bit | LoRA Rank4 Merged | 1.0000 | 0.4000 | 0.6500 | 2.8444 | 29.1160 | success |
| LoRA4_vLLM_AWQ | LoRA Rank4 Merged | 1.0000 | 0.3500 | 0.7500 | 0.8092 | 96.2294 | success |
| Top18Rank8_Transformers_16bit | Top18 Rank8 Merged | 1.0000 | 0.4500 | 0.7000 | 2.6367 | 29.2135 | success |
| Top18Rank8_vLLM_AWQ | Top18 Rank8 Merged | 1.0000 | 0.4000 | 0.7500 | 1.2542 | 69.3970 | success |

## 结果分析

- 速度最优方案为 `Top18Rank8_vLLM_AWQ`，benchmark 平均延迟为 `0.6836s`。
- `Exact Match` 最高方案为 `Top18Rank8_Transformers_16bit`，精确匹配率为 `0.4500`。
- `Action Match` 最高方案为 `LoRA4_vLLM_AWQ`，动作匹配率为 `0.7500`。
- 由于这些方案同时包含“模型家族差异”和“部署栈差异”，解读时建议先做同家族内横向比较，再做跨家族比较。

## 解读建议

- 若关注同一模型在不同部署下的收益，可比较 `LoRA4_Transformers_16bit vs LoRA4_vLLM_AWQ`，以及 `Top18Rank8_Transformers_16bit vs Top18Rank8_vLLM_AWQ`。
- 若关注不同模型家族本身谁更强，可比较两组 `Transformers 16bit`，或比较两组 `vLLM + AWQ`。
- AWQ 目录均由 `llmcompressor` 导出，因此这里的 AWQ 结论应理解为“AWQ 压缩部署方案 vs Transformers 基线”的端到端对比。
