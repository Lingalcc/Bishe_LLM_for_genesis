# Exp15 LoRA4 / Top18Rank8 的 AWQ 扩展对比实验报告（含 4GB AWQ）

## 实验目标

- 保留既有四组结果：`LoRA4/Top18Rank8` 在 `Transformers 16bit` 与 `vLLM + AWQ` 下的原始对比。
- 新增两组 `4GB AWQ` 扩展案例：`LoRA4_vLLM_AWQ_4GB` 与 `Top18Rank8_vLLM_AWQ_4GB`。
- 扩展模式只运行新增两组，并与旧四组结果拼接输出，不覆盖之前的 CSV、JSON、Markdown 和图表。

## 对比矩阵

- `LoRA4_Transformers_16bit`：LoRA4 merged 的未量化基线（复用既有结果）。
- `LoRA4_vLLM_AWQ`：LoRA4 merged 的常规 AWQ 部署（复用既有结果）。
- `Top18Rank8_Transformers_16bit`：Top18 rank8 merged 的未量化基线（复用既有结果）。
- `Top18Rank8_vLLM_AWQ`：Top18 rank8 merged 的常规 AWQ 部署（复用既有结果）。
- `LoRA4_vLLM_AWQ_4GB`：LoRA4 merged 的 4GB 目标显存 AWQ 扩展组。
- `Top18Rank8_vLLM_AWQ_4GB`：Top18 rank8 merged 的 4GB 目标显存 AWQ 扩展组。

## 速度结果

| 方案 | 模型家族 | Backend | 载入格式 | Avg Latency (s) | Tokens/s | Peak VRAM (MB) | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| LoRA4_Transformers_16bit | LoRA Rank4 Merged | transformers | HF Safetensors | 1.9884 | 27.3089 | 7744.00 | success |
| LoRA4_vLLM_AWQ | LoRA Rank4 Merged | vllm | Compressed Tensors (AWQ) | 0.7990 | 76.5912 | 7808.00 | success |
| Top18Rank8_Transformers_16bit | Top18 Rank8 Merged | transformers | HF Safetensors | 2.3045 | 26.9692 | 6742.00 | success |
| Top18Rank8_vLLM_AWQ | Top18 Rank8 Merged | vllm | Compressed Tensors (AWQ) | 0.6836 | 82.9447 | 7868.00 | success |
| LoRA4_vLLM_AWQ_4GB | LoRA Rank4 Merged | vllm | Compressed Tensors (AWQ) | 0.6524 | 93.8086 | 6116.00 | success |
| Top18Rank8_vLLM_AWQ_4GB | Top18 Rank8 Merged | vllm | Compressed Tensors (AWQ) | 0.6148 | 92.2182 | 6049.00 | success |

## 精度结果

| 方案 | 模型家族 | Parse OK | Exact Match | Action Match | Accuracy Avg Latency (s) | Accuracy Tokens/s | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA4_Transformers_16bit | LoRA Rank4 Merged | 1.0000 | 0.4000 | 0.6500 | 2.8444 | 29.1160 | success |
| LoRA4_vLLM_AWQ | LoRA Rank4 Merged | 1.0000 | 0.3500 | 0.7500 | 0.8092 | 96.2294 | success |
| Top18Rank8_Transformers_16bit | Top18 Rank8 Merged | 1.0000 | 0.4500 | 0.7000 | 2.6367 | 29.2135 | success |
| Top18Rank8_vLLM_AWQ | Top18 Rank8 Merged | 1.0000 | 0.4000 | 0.7500 | 1.2542 | 69.3970 | success |
| LoRA4_vLLM_AWQ_4GB | LoRA Rank4 Merged | 1.0000 | 0.3500 | 0.7500 | 0.8061 | 95.6852 | success |
| Top18Rank8_vLLM_AWQ_4GB | Top18 Rank8 Merged | 1.0000 | 0.4000 | 0.7500 | 0.7440 | 99.5988 | success |

## 结果分析

- 速度最优方案为 `Top18Rank8_vLLM_AWQ_4GB`，benchmark 平均延迟为 `0.6148s`。
- `Exact Match` 最高方案为 `Top18Rank8_Transformers_16bit`，精确匹配率为 `0.4500`。
- `Action Match` 最高方案为 `LoRA4_vLLM_AWQ`，动作匹配率为 `0.7500`。
- 由于这些方案同时包含“模型家族差异”和“部署栈差异”，解读时建议先做同家族内横向比较，再做跨家族比较。

## 解读建议

- 若关注 4GB 约束下的可运行性，可直接比较 `LoRA4_vLLM_AWQ vs LoRA4_vLLM_AWQ_4GB`，以及 `Top18Rank8_vLLM_AWQ vs Top18Rank8_vLLM_AWQ_4GB`。
- 若关注整体最优方案，可在 6 组结果里同时观察延迟、吞吐、显存和动作匹配率，但要注意 4GB 组与常规组的显存目标不同。
- 4GB AWQ 组默认使用更保守的运行参数，目的是在小显存约束下完成推理，不应直接等同于常规高缓存配置。
- AWQ 目录均由 `llmcompressor` 导出，因此这里的 AWQ 结论应理解为“AWQ 压缩部署方案 vs Transformers 基线”的端到端对比。
