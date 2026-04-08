# Exp7 推理部署对比实验报告

## 实验目标

- 将原来的单一 `vLLM` 显存预算实验改为三方案统一对比。
- 当前对比矩阵为 `Transformers 16bit`、`vLLM AWQ`、`vLLM GGUF`。
- 每个方案都同时执行 benchmark 与 accuracy，分别观察速度、吞吐、解析率、精确匹配率和动作匹配率。

## 公平性口径

- 三组方案统一使用同一份 `merged` 系列模型资产，只改变加载后端或模型格式。
- benchmark 统一使用相同 prompts、batch size、num samples、max_new_tokens、max_model_len。
- accuracy 统一使用相同测试集、相同随机种子、相同 system prompt 与生成参数。
- `Transformers 16bit` 代表未量化基线；`vLLM AWQ` 与 `vLLM GGUF` 代表两种不同的部署格式。

## 速度结果

| 方案 | Backend | 载入格式 | Avg Latency (s) | P50 (s) | P95 (s) | Samples/s | Tokens/s | Peak VRAM (MB) | 状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Transformers_16bit | transformers | HF Safetensors | 1.9976 | 1.9198 | 3.5386 | 0.5006 | 28.5187 | 7669.00 | success |
| vLLM_AWQ | vllm | AWQ | 13.3468 | 11.4117 | 24.2194 | 0.0749 | 9.5903 | 7943.00 | success |
| vLLM_GGUF | vllm | GGUF | - | - | - | - | - | 7886.00 | failed |

## 精度结果

| 方案 | Parse OK | Exact Match | Action Match | Accuracy Avg Latency (s) | Accuracy Tokens/s | Accuracy Max VRAM (MB) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Transformers_16bit | 1.0000 | 0.2950 | 0.5900 | 2.6572 | 30.6390 | 6040.00 | success |
| vLLM_AWQ | 0.0000 | 0.0000 | 0.0000 | 11.0441 | 11.7096 | 0.00 | success |
| vLLM_GGUF | - | - | - | - | - | - | failed |

## 结果分析

- 速度最优方案为 `Transformers_16bit`，其 benchmark 平均延迟为 `1.9976s`。
- `Exact Match` 最高的方案为 `Transformers_16bit`，精确匹配率为 `0.2950`。
- `Action Match` 最高的方案为 `Transformers_16bit`，动作匹配率为 `0.5900`。
- 如果速度与精度最优方案不是同一个，就说明当前实验存在明显的部署权衡。

## 解读建议

- 如果你要写“部署效率”，优先引用 benchmark 的 `Avg Latency`、`Tokens/s` 和 `Peak VRAM`。
- 如果你要写“任务可用性”，优先引用 accuracy 的 `Parse OK Rate`、`Exact Match Rate` 和 `Action Match Rate`。
- `vLLM GGUF` 这里表示以 GGUF 文件格式加载；它与 `vLLM AWQ` 不是同一种量化机制，因此结论应写成“当前仓库内三种加载方案的端到端表现对比”，不要写成“同构量化下的纯引擎结论”。
