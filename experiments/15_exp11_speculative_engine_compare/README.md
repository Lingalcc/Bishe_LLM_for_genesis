# 实验15：按 Exp8 口径对比 vLLM 与 Transformers 投机解码

本实验回答的问题已经更新为：

在与 [`12_exp8_speculative_decoding`](/home/lin/Bishe_LLM_for_genesis/experiments/12_exp8_speculative_decoding) 尽量一致的设置下，`vLLM(float16)` 基线、`Transformers` 基线、`Transformers` 投机解码三组方案的性能差异分别如何？

## 实验设计

- 主模型：默认使用 `model/qwen2.5-3b-genesis-merged`
- 助手模型：默认使用 `model/qwen2.5-0.5b-genesis-merged`
- 后端：`transformers` 与 `vllm`
- 解码模式：`vLLM` 仅保留 `standard`；`Transformers` 保留 `standard` 与 `speculative`
- `Transformers` 投机方式：`assistant_model`
- `vLLM` 精度：固定使用 `float16` 加载 3B 主模型
- 生成参数：`num_samples=50`、`batch_size=1`、`max_new_tokens=256`、`temperature=0.0`
- 预热参数：`warmup_samples=5`
- 投机参数：`assistant_num_tokens=8`
- Transformers 附加参数：`assistant_confidence_threshold=0.55`、`assistant_num_tokens_schedule=constant`
- 加载策略：优先与 `exp8` 保持一致，`Transformers` 默认开启 `prefer_same_gpu=true` 与 `allow_auto_device_map_fallback=true`

## 与 Exp8 的关系

- `Transformers` 路径会尽量复用 `exp8` 的运行口径，包括 BF16/FP16 自动选择、同卡优先加载、相同的 assisted generation 参数。
- `vLLM` 不再参与投机解码对比，只承担 `3B float16` 基线生成的测量任务，用来和 `Transformers` 的基线、投机解码结果直接比较。
- 为了提高 `vLLM` 在当前 `RTX 4060 Laptop 8GB` 环境中的可运行性，脚本保留 `gpu_memory_utilization` 与 `enforce_eager` 控制项；其余核心评测设置尽量与 `exp8` 对齐。
- 脚本会根据当前空闲显存自动下调 `vllm_gpu_memory_utilization`，尽量减少因为后台 CUDA 进程占用而导致的启动失败。

## 运行方式

默认会形成三组核心对比：

- `vLLM float16 无投机解码`
- `Transformers 无投机解码`
- `Transformers 投机解码`

默认运行命令：

```bash
conda run -n llm_genesis python experiments/15_exp11_speculative_engine_compare/run_exp11_speculative_engine_compare.py
```

只跑某一种解码模式：

```bash
conda run -n llm_genesis python experiments/15_exp11_speculative_engine_compare/run_exp11_speculative_engine_compare.py \
  --backends transformers \
  --decoding-modes speculative
```

只跑某一个后端：

```bash
conda run -n llm_genesis python experiments/15_exp11_speculative_engine_compare/run_exp11_speculative_engine_compare.py \
  --backends vllm
```

## 输出产物

运行完成后，默认会在 [`reports`](/home/lin/Bishe_LLM_for_genesis/experiments/15_exp11_speculative_engine_compare/reports) 目录下生成：

- `speculative_engine_compare_report.json`
- `exp11_speculative_engine_compare.csv`
- `exp11_speculative_engine_compare.md`
- `exp11_speculative_latency_bar.png`
- `exp11_speculative_throughput_bar.png`
- `exp11_speculative_memory_bar.png`
- `run_meta.json`

## 解读建议

- 如果你关心交互响应，优先看 `Avg Latency (s)`。
- 如果你关心生成效率，优先看 `Token Throughput (tokens/s)`。
- 如果你关心显存代价，优先看 `Peak VRAM (MB)`。
- 如果你担心解码策略影响结构化输出稳定性，优先看 `Parse OK Rate`。
