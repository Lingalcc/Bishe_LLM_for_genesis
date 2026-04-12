# 实验 18 Exp14：AWQ 与 Transformers 对比

这个实验专门回答一个更聚焦的问题：

- `Transformers 16bit`
- `vLLM + AWQ`

这两种部署方案在同一任务、同一测试集、同一生成口径下，谁更快、谁更省显存、谁更准确？

## 实验设计

- 方案 1：`Transformers_16bit`
  - 使用 `transformers` 直接以 `float16` 加载 `model/qwen2.5-3b-genesis-merged`
- 方案 2：`vLLM_AWQ_CompressedTensors`
  - 使用 `vllm` 加载 `model/qwen2.5-3b-genesis-merged-awq`
  - 该目录由 `llmcompressor` 导出，运行时按 `compressed-tensors` 方式加载

统一控制变量：

- 相同 benchmark prompts
- 相同 `batch_size`
- 相同 `max_new_tokens`
- 相同 `max_model_len`
- 相同 accuracy 测试集
- 相同 accuracy 随机种子

## 运行方式

默认执行一轮快速对比：

```bash
conda run -n llm_genesis python experiments/18_exp14_awq_vs_transformers/run_exp14_awq_vs_transformers.py
```

如果希望扩大样本量：

```bash
conda run -n llm_genesis python experiments/18_exp14_awq_vs_transformers/run_exp14_awq_vs_transformers.py \
  --benchmark-num-samples 100 \
  --accuracy-num-samples 100
```

## 输出产物

运行完成后会在 `experiments/18_exp14_awq_vs_transformers/reports/` 下生成：

- `exp14_awq_vs_transformers_comparison.csv`
- `exp14_awq_vs_transformers_summary.json`
- `exp14_awq_vs_transformers_report.md`
- `exp14_awq_vs_transformers_latency_bar.png`
- `exp14_awq_vs_transformers_throughput_bar.png`
- `exp14_awq_vs_transformers_accuracy_bar.png`
- `exp14_awq_vs_transformers_memory_bar.png`
- 每个方案对应的 `*_benchmark.json`
- 每个方案对应的 `*_accuracy.json`

## 解读建议

- 如果你关心部署速度，优先看 `Benchmark Avg Latency (s)` 和 `Benchmark Token Throughput (tokens/s)`
- 如果你关心任务正确率，优先看 `Exact Match Rate` 和 `Action Match Rate`
- 如果你关心落地成本，优先看 `Benchmark Peak VRAM (MB)` 与 `Accuracy Max Peak VRAM (MB)`

需要特别注意：

- 当前 AWQ 目录并不是传统 `autoawq` 风格目录，而是 `llmcompressor` 导出的压缩目录
- 因此这里更准确的表述是“AWQ 压缩部署方案 vs Transformers 基线”的端到端对比
