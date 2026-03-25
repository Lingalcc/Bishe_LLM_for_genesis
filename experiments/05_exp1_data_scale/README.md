# 05_exp1_data_scale

用于研究不同训练数据量对大语言模型微调效果的影响。

## 运行方式

```bash
python experiments/05_exp1_data_scale/run_exp1_data_scale.py
```

## 输出位置

- 评测报告：`experiments/05_exp1_data_scale/reports/accuracy_report_exp1_data_scale.json`
- 统计表：`experiments/05_exp1_data_scale/reports/exp1_data_scale_results.csv`
- 折线图：`experiments/05_exp1_data_scale/reports/exp1_data_scale_chart.png`
- 断点状态：`experiments/05_exp1_data_scale/reports/progress_state.json`
- 完整日志：`experiments/05_exp1_data_scale/logs/`
- 全量训练集快照：`experiments/05_exp1_data_scale/.cache/full_train_snapshot.json`

## 说明

- 脚本会在实验开始前读取完整的 `data_prepare/splits/train.json`。
- 每轮实验都会将前 N 条样本覆写到 `train.json`，保证小数据集是大数据集的严格子集。
- 首次在“完整训练集”上运行时，脚本会自动保存一份全量训练集快照，后续断点续跑都会基于这份快照重建子集，避免把已经截断的 `train.json` 误当作全量数据。
- 终端只显示关键信息与阶段进度，训练与评测的完整原始输出会实时保存到 `logs/`。
- 每完成一个数据规模，脚本都会立刻保存当前结果和断点状态；如果中途中断，重新运行同一命令会自动跳过已完成的数据规模并继续实验。
- 无论实验中途是否报错，都会在 `finally` 中恢复原始训练集。
