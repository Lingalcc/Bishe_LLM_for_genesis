# 实验 11_exp7_vllm 使用说明

这个实验已经从“单一 `vLLM + 4bit` 的显存预算切片”改成了新的三方案统一对比实验。

当前实验要回答的问题是：

- `Transformers 16bit`
- `vLLM AWQ`
- `vLLM GGUF`

这三种加载方案在同一任务、同一测试集、同一生成口径下，谁更快、谁更省显存、谁更准确？

## 实验设计

- 方案 1：`Transformers_16bit`
  - 使用 `transformers` 直接以 `float16` 加载 `model/qwen2.5-3b-genesis-merged`
- 方案 2：`vLLM_AWQ`
  - 使用 `vllm` 加载 `model/qwen2.5-3b-genesis-merged-awq`
- 方案 3：`vLLM_GGUF`
  - 使用 `vllm` 以 `GGUF` 文件格式加载 `model/qwen2.5-3b-genesis-merged-q4_k_m.f16.gguf`

统一控制变量：

- 相同 benchmark prompts
- 相同 `batch_size`
- 相同 `max_new_tokens`
- 相同 `max_model_len`
- 相同 accuracy 测试集
- 相同 accuracy 随机种子

输出分成两部分：

- 速度指标：`Avg Latency`、`P50/P95`、`Samples/s`、`Tokens/s`、`Peak VRAM`
- 精度指标：`Parse OK Rate`、`Exact Match Rate`、`Action Match Rate`

## 运行方式

默认执行三方案完整对比：

```bash
python experiments/11_exp7_vllm/run_exp7_vllm_benchmark.py
```

如果希望缩小测试规模做 smoke test：

```bash
python experiments/11_exp7_vllm/run_exp7_vllm_benchmark.py \
  --benchmark-num-samples 20 \
  --accuracy-num-samples 20
```

如果当前环境缺少依赖，希望自动补装：

```bash
python experiments/11_exp7_vllm/run_exp7_vllm_benchmark.py \
  --auto-install-deps
```

## 输出产物

运行完成后会在 `experiments/11_exp7_vllm/reports/` 下生成：

- `exp7_vllm_engine_comparison.csv`
- `exp7_vllm_summary.json`
- `exp7_vllm_report.md`
- `exp7_vllm_latency_bar.png`
- `exp7_vllm_throughput_bar.png`
- `exp7_vllm_accuracy_bar.png`
- `exp7_vllm_memory_bar.png`
- `run_meta.json`
- 每个方案对应的 `*_benchmark.json`
- 每个方案对应的 `*_accuracy.json`

## 如何解读

- 如果你关心部署速度，优先看 `Benchmark Avg Latency (s)` 和 `Benchmark Token Throughput (tokens/s)`
- 如果你关心任务正确率，优先看 `Exact Match Rate` 和 `Action Match Rate`
- 如果你关心落地成本，优先看 `Benchmark Peak VRAM (MB)` 与 `Accuracy Max Peak VRAM (MB)`

需要特别注意：

- `AWQ` 和 `GGUF` 不是同一种量化机制
- 所以这里更适合写成“当前仓库内三种加载方案的端到端对比”
- 不建议直接把结论写成“同构量化条件下某个引擎天然更快”

## 和旧版 Exp7 的关系

- 旧版 Exp7 回答的是：同一个 `vLLM + 4bit` 在不同显存预算下会怎样退化
- 新版 Exp7 回答的是：`Transformers 16bit`、`vLLM AWQ`、`vLLM GGUF` 三种部署方案的速度与精度谁更优

如果后续你还想保留“显存预算切片”这个问题，建议把它单独迁到一个新的实验目录，避免和现在这版三方案对比混在一起。
