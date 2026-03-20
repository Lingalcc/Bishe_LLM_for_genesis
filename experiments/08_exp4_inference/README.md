# 实验08 Exp4：不同量化推理性能对比

本实验用于统计并对比 3B 模型在 `4bit`、`8bit`、`16bit` 三种精度设置下的推理性能与资源占用。

## 实验目标

- 对比不同量化下的推理速度：
  - 平均延迟 `avg_latency_sec`
  - P50 / P95 延迟
  - 样本吞吐 `sample_throughput_sps`
  - Token 吞吐 `token_throughput_tps`
- 对比不同量化下的资源占用：
  - 推理阶段平均 / 峰值显存
  - 模型加载阶段峰值显存
  - 进程 RSS 内存
- 输出统一的 JSON / CSV / Markdown / PNG 报告，便于论文、答辩和复现实验。

## 默认实验设置

- 模型：`model/Qwen_Qwen2.5-3B-Instruct`
- 后端：`transformers`
- 对比项：`4bit,8bit,16bit`
- prompts：`experiments/08_exp4_inference/prompts/default_prompts.json`

说明：

- 这里的 `16bit` 在当前实现中对应 `transformers + torch.float16` 基线；
- `4bit` 和 `8bit` 使用 `bitsandbytes` 量化；
- 若改用 `vllm`，请确认当前模型与环境支持相应量化方式。

## 运行方法

```bash
python experiments/08_exp4_inference/run_exp4_inference.py
```

如需自定义参数：

```bash
python experiments/08_exp4_inference/run_exp4_inference.py \
  --model-path model/Qwen_Qwen2.5-3B-Instruct \
  --backend transformers \
  --cases 4bit,8bit,16bit \
  --batch-size 1 \
  --num-samples 24 \
  --max-new-tokens 128
```

## 产物说明

运行完成后会在 `experiments/08_exp4_inference/reports/` 下生成：

- `exp4_inference_summary.json`
  - 总实验汇总，包含每个量化配置的详细指标
- `exp4_inference_summary.md`
  - 适合直接粘贴到实验记录或论文草稿中的表格摘要
- `exp4_inference_results.csv`
  - 结构化结果表，便于后续二次分析
- `exp4_inference_dashboard.png`
  - 延迟、吞吐、推理峰值显存、加载峰值显存四联图
- `exp4_<量化>_benchmark.json`
  - 单个量化配置的详细 benchmark 数据
- `exp4_<量化>_batches.csv`
  - 单个量化配置逐 batch 的明细指标

## 指标解释

- `avg_latency_sec`
  - 每个 batch 推理耗时的平均值
- `sample_throughput_sps`
  - 单位时间内完成的样本数，单位 `samples/s`
- `token_throughput_tps`
  - 单位时间内生成的 token 数，单位 `tokens/s`
- `avg_infer_peak_vram_mb`
  - 推理阶段逐 batch 峰值显存的平均值
- `max_infer_peak_vram_mb`
  - 整个实验过程中观测到的最大峰值显存
- `load_peak_vram_mb`
  - 模型加载阶段的显存峰值
- `latency_speedup_vs_16bit`
  - 相对 `16bit` 的延迟加速比，越大越好
- `memory_saving_vs_16bit_pct`
  - 相对 `16bit` 的推理峰值显存节省比例，越大越好

## 建议解读方式

- 如果目标是“部署更省显存”，优先看 `max_infer_peak_vram_mb` 与 `load_peak_vram_mb`
- 如果目标是“响应更快”，优先看 `avg_latency_sec`、`p95_latency_sec`
- 如果目标是“吞吐更高”，优先看 `sample_throughput_sps` 与 `token_throughput_tps`
- 若 4bit 显存显著下降但延迟未明显改善，这是常见现象，因为量化并不总能带来线性加速
