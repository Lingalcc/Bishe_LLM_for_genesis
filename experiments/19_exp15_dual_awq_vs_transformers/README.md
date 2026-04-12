# 实验 19 Exp15：双模型 AWQ 与 Transformers 四组对比

这个实验专门回答一个更完整的问题：

- `LoRA rank4 merged + Transformers 16bit`
- `LoRA rank4 merged + vLLM AWQ`
- `Top18 rank8 merged + Transformers 16bit`
- `Top18 rank8 merged + vLLM AWQ`

这四组方案在同一任务、同一测试集、同一生成口径下，谁更快、谁更省显存、谁更准确？

现在额外支持一个“4GB AWQ 扩展模式”：

- 读取已经生成好的四组结果
- 只新增运行 `LoRA4_vLLM_AWQ_4GB`
- 只新增运行 `Top18Rank8_vLLM_AWQ_4GB`
- 将这两组与旧四组拼接成新的 6 组对比产物
- 不覆盖原始四组的 CSV、JSON、Markdown 和图表

## 实验设计

- 模型家族 A：`LoRA4`
  - `Transformers_16bit` 使用 `model/qwen2.5-3b-genesis-merged`
  - `vLLM_AWQ` 使用 `model/qwen2.5-3b-genesis-merged-awq`
- 模型家族 B：`Top18Rank8`
  - `Transformers_16bit` 使用 `model/qwen2.5-3b-top18-rank8-merged`
  - `vLLM_AWQ` 使用 `model/qwen2.5-3b-top18-rank8-merged-awq`

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
conda run -n llm_genesis python experiments/19_exp15_dual_awq_vs_transformers/run_exp15_dual_awq_vs_transformers.py
```

如果当前环境中的 `compressed-tensors` 版本与仓库既有检查不一致，但你确认要继续实测：

```bash
conda run -n llm_genesis python experiments/19_exp15_dual_awq_vs_transformers/run_exp15_dual_awq_vs_transformers.py \
  --skip-vllm-compat-check
```

如果希望扩大样本量：

```bash
conda run -n llm_genesis python experiments/19_exp15_dual_awq_vs_transformers/run_exp15_dual_awq_vs_transformers.py \
  --benchmark-num-samples 100 \
  --accuracy-num-samples 100 \
  --skip-vllm-compat-check
```

如果你已经有原始四组结果，现在只想追加两组 4GB AWQ 扩展对比：

```bash
conda run -n llm_genesis python experiments/19_exp15_dual_awq_vs_transformers/run_exp15_dual_awq_vs_transformers.py \
  --append-4gb-awq-cases \
  --skip-vllm-compat-check
```

扩展模式默认会读取：

- `experiments/19_exp15_dual_awq_vs_transformers/reports/exp15_dual_awq_vs_transformers_summary.json`

并生成一套新的扩展产物，默认前缀为：

- `exp15_dual_awq_vs_transformers_with_4gb_awq`

## 输出产物

运行完成后会在 `experiments/19_exp15_dual_awq_vs_transformers/reports/` 下生成：

- `exp15_dual_awq_vs_transformers_comparison.csv`
- `exp15_dual_awq_vs_transformers_summary.json`
- `exp15_dual_awq_vs_transformers_report.md`
- `exp15_dual_awq_vs_transformers_latency_bar.png`
- `exp15_dual_awq_vs_transformers_throughput_bar.png`
- `exp15_dual_awq_vs_transformers_accuracy_bar.png`
- `exp15_dual_awq_vs_transformers_memory_bar.png`
- 每个方案对应的 `*_benchmark.json`
- 每个方案对应的 `*_accuracy.json`

如果使用 `--append-4gb-awq-cases`，会额外生成：

- `exp15_dual_awq_vs_transformers_with_4gb_awq_comparison.csv`
- `exp15_dual_awq_vs_transformers_with_4gb_awq_summary.json`
- `exp15_dual_awq_vs_transformers_with_4gb_awq_report.md`
- `exp15_dual_awq_vs_transformers_with_4gb_awq_latency_bar.png`
- `exp15_dual_awq_vs_transformers_with_4gb_awq_throughput_bar.png`
- `exp15_dual_awq_vs_transformers_with_4gb_awq_accuracy_bar.png`
- `exp15_dual_awq_vs_transformers_with_4gb_awq_memory_bar.png`
- 新增两组对应的 `LoRA4_vLLM_AWQ_4GB_*`
- 新增两组对应的 `Top18Rank8_vLLM_AWQ_4GB_*`

## 解读建议

- 如果你关心同一模型在不同部署下的速度收益，优先做同家族内比较：
  `LoRA4_Transformers_16bit vs LoRA4_vLLM_AWQ`
  `Top18Rank8_Transformers_16bit vs Top18Rank8_vLLM_AWQ`
- 如果你关心 4GB 约束下还能否运行，就比较：
  `LoRA4_vLLM_AWQ vs LoRA4_vLLM_AWQ_4GB`
  `Top18Rank8_vLLM_AWQ vs Top18Rank8_vLLM_AWQ_4GB`
- 如果你关心模型家族本身谁更强，可以横向比较两组 `Transformers 16bit` 或两组 `vLLM + AWQ`
- 当前 AWQ 目录并不是传统 `autoawq` 风格目录，而是 `llmcompressor` 导出的压缩目录
- 因此这里更准确的表述是“AWQ 压缩部署方案 vs Transformers 基线”的端到端对比
