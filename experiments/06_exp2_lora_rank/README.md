# 06_exp2_lora_rank

用于研究在训练数据规模固定为 600 时，不同 LoRA rank 对 `Qwen2.5-3B-Instruct` 微调效果的影响。

## 实验设置

- 基础模型：`model/Qwen_Qwen2.5-3B-Instruct`
- 微调方法：`LoRA`
- 固定训练样本数：`600`
- 默认 rank 序列：`[4, 8, 16, 32, 64]`
- 其余训练超参数：沿用现有 `experiments/02_finetune_exp/configs/llamafactory_train_lora_sft.yaml`

## 运行方式

```bash
python experiments/06_exp2_lora_rank/run_exp2_lora_rank.py
```

## 输出位置

- 汇总表：`experiments/06_exp2_lora_rank/reports/exp2_lora_rank_results.csv`
- 对比图：`experiments/06_exp2_lora_rank/reports/exp2_lora_rank_dashboard.png`
- 断点状态：`experiments/06_exp2_lora_rank/reports/progress_state.json`
- 每个 rank 的评测报告：`experiments/06_exp2_lora_rank/reports/accuracy_report_rank_<rank>.json`
- 完整日志：`experiments/06_exp2_lora_rank/logs/`
- 全量训练集快照：`experiments/06_exp2_lora_rank/.cache/full_train_snapshot.json`

## 指标说明

脚本会同时汇总训练阶段和评测阶段的关键指标：

- 训练阶段：`train_time_sec`、`train_time_min`、`final_loss`、`min_loss`、`train_peak_vram_mb`
- 评测阶段：`parse_ok_rate`、`exact_match_rate`、`action_match_rate`
- 推理性能：`avg_latency_sec`、`avg_throughput_tps`、`avg_peak_vram_mb`、`max_peak_vram_mb`

## 断点续跑与数据一致性

- 脚本首次运行时会把当前 `data_prepare/splits/train.json` 保存为全量训练集快照。
- 每轮实验都会把前 600 条样本覆写到 `train.json`，确保所有 rank 使用完全相同的训练子集。
- 每完成一个 rank，脚本都会立即刷新 CSV、图表和断点状态。
- 如果中途中断，重新执行同一命令会自动跳过已完成 rank，继续剩余实验。
- 无论实验成功、中断还是失败，脚本都会在 `finally` 中恢复原始 `train.json`。
