# Exp5 七组推理部署对比实验报告

## 实验目标

- 在同一份 `qwen2.5-3b-genesis-merged` 任务模型上，对比 `Transformers` 与 `vLLM` 的 16bit / 8bit / 4bit 部署表现。
- 额外纳入 `vLLM + AWQ` 作为第 7 组，观察预量化压缩目录在速度、显存与精度上的落点。
- 如果某组在 benchmark 或 accuracy 阶段出现显存不足，实验会记录其状态并继续执行后续组别。

## 对比矩阵

- `Transformers_16bit`：Transformers + FP16；量化标签 `16bit`；格式 `HF Safetensors`。
- `Transformers_8bit`：Transformers + bitsandbytes 8bit；量化标签 `8bit`；格式 `HF Safetensors`。
- `Transformers_4bit`：Transformers + bitsandbytes 4bit；量化标签 `4bit`；格式 `HF Safetensors`。
- `vLLM_16bit`：vLLM + FP16；量化标签 `16bit`；格式 `HF Safetensors`。
- `vLLM_8bit`：vLLM + bitsandbytes 8bit；量化标签 `8bit`；格式 `HF Pre-Quantized BNB`。
- `vLLM_4bit`：vLLM + bitsandbytes 4bit；量化标签 `4bit`；格式 `HF Pre-Quantized BNB`。
- `vLLM_AWQ`：vLLM + AWQ；量化标签 `awq`；格式 `Compressed Tensors (AWQ)`。

## 速度结果

| 方案 | Family | Backend | 量化 | Avg Latency (s) | P95 (s) | Samples/s | Tokens/s | Peak VRAM (MB) | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Transformers_16bit | Transformers | transformers | 16bit | 1.9775 | 3.5324 | 0.5057 | 28.8089 | 7492.00 | success |
| Transformers_8bit | Transformers | transformers | 8bit | 7.5157 | 14.4238 | 0.1331 | 6.9462 | 4771.00 | success |
| Transformers_4bit | Transformers | transformers | 4bit | 2.3155 | 4.3258 | 0.4319 | 19.5986 | 4340.00 | success |
| vLLM_16bit | vLLM | vllm | 16bit | - | - | - | - | - | oom |
| vLLM_8bit | vLLM | vllm | 8bit | 9.0873 | 18.6397 | 0.1100 | 6.7011 | 7947.00 | success |
| vLLM_4bit | vLLM | vllm | 4bit | 0.8616 | 1.4599 | 1.1606 | 65.5932 | 7942.00 | success |
| vLLM_AWQ | vLLM | vllm | awq | 0.7115 | 1.4130 | 1.4055 | 88.0069 | 7783.00 | success |

## 精度结果

| 方案 | Parse OK | Exact Match | Action Match | Accuracy Avg Latency (s) | Accuracy Tokens/s | Accuracy Max VRAM (MB) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Transformers_16bit | 1.0000 | 0.2950 | 0.5900 | 2.5695 | 31.6640 | 7508.00 | success |
| Transformers_8bit | 1.0000 | 0.3000 | 0.5950 | 10.6480 | 7.6733 | 4767.00 | success |
| Transformers_4bit | 0.9650 | 0.2500 | 0.5350 | 3.7948 | 21.1756 | 4348.00 | success |
| vLLM_16bit | - | - | - | - | - | - | oom |
| vLLM_8bit | 0.9950 | 0.2850 | 0.6050 | 11.4479 | 7.1816 | 7844.00 | success |
| vLLM_4bit | 0.9650 | 0.2450 | 0.5350 | 1.2209 | 65.2516 | 7934.00 | success |
| vLLM_AWQ | 0.9950 | 0.2450 | 0.5600 | 0.8426 | 95.4320 | 7927.00 | success |

## 异常记录

- `vLLM_16bit`：benchmark=`oom`，accuracy=`oom`，overall=`benchmark=oom;accuracy=oom`。

## 结果分析

- 速度最优方案为 `vLLM_AWQ`，benchmark 平均延迟为 `0.7115s`。
- `Exact Match` 最高方案为 `Transformers_8bit`，精确匹配率为 `0.3000`。
- `Action Match` 最高方案为 `vLLM_8bit`，动作匹配率为 `0.6050`。
- 显存占用最低的成功方案为 `Transformers_4bit`，benchmark 峰值显存为 `4340.00 MB`。

## 说明

- `Transformers 8bit/4bit` 与 `vLLM 8bit/4bit` 都使用同一份 merged 模型，只是运行时后端和量化路径不同。
- `vLLM 8bit/4bit` 使用的是预先导出的 bitsandbytes 量化目录，不再走运行时从 FP16 safetensors 现量化的路径。
- `vLLM_AWQ` 使用的是预先压缩好的 AWQ 目录，因此它与运行时 bitsandbytes 量化并不是完全同构的量化路径。
