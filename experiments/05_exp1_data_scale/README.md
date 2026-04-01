# 05_exp1_data_scale

用于研究不同训练数据量对大语言模型微调效果的影响。

## 运行方式

```bash
python experiments/05_exp1_data_scale/run_exp1_data_scale.py
```

## 扩展实验：4000 条数据 + QLoRA rank=4

当需要在 `1600` 之后继续扩展数据规模，并且不覆盖旧 `exp1` 结果时，使用下面这组独立脚本与独立输出：

### 1) 生成 4000 条专用 split

这一步不会改写旧的 `data_prepare/splits/`，而是写入新的 `data_prepare/splits_4000_exp1_qlora_rank4/`。

```bash
python cli.py data split --config experiments/05_exp1_data_scale/configs/split_4000_qlora_rank4.yaml
```

说明：

- 为了支持训练规模做到 `3600`，这里采用独立的 `90% / 5% / 5%` 切分。
- 对应训练/验证/测试规模约为 `3600 / 200 / 200`。

### 2) 运行扩展版 exp1

```bash
python experiments/05_exp1_data_scale/run_exp1_data_scale_qlora_rank4.py
```

扩展版默认设置：

- 微调方法：`QLoRA`
- `lora_rank=4`
- `lora_alpha=32`
- 数据规模：`[1800, 2000, 2200, ..., 3600]`
- 使用独立 split：`data_prepare/splits_4000_exp1_qlora_rank4/`
- 使用独立模型输出：`output/exp1_data_scale_qlora_rank4/`
- 使用独立报告目录：`experiments/05_exp1_data_scale/reports/qlora_rank4/`
- 使用独立日志目录：`experiments/05_exp1_data_scale/logs/qlora_rank4/`

扩展版脚本首次运行时也会自动检查并生成专用 split；如果你想强制重切分，可以这样执行：

```bash
python experiments/05_exp1_data_scale/run_exp1_data_scale_qlora_rank4.py --force-resplit
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

### 扩展版额外产物

扩展版会额外输出：

- 汇总表：`experiments/05_exp1_data_scale/reports/qlora_rank4/exp1_data_scale_qlora_rank4_results.csv`
- 仪表盘：`experiments/05_exp1_data_scale/reports/qlora_rank4/exp1_data_scale_qlora_rank4_dashboard.png`
- 边界摘要：`experiments/05_exp1_data_scale/reports/qlora_rank4/exp1_data_scale_qlora_rank4_boundary.json`
- 断点状态：`experiments/05_exp1_data_scale/reports/qlora_rank4/progress_state.json`
- 分规模准确率报告：`experiments/05_exp1_data_scale/reports/qlora_rank4/accuracy_report_size_<size>.json`

## 接续实验：改回 LoRA，并从 1600 之后继续

如果你要保持 `exp1` 的原始 LoRA 主线，并把 `1800-3600` 直接续接到原来的
`experiments/05_exp1_data_scale/reports/exp1_data_scale_results.csv` 和
`experiments/05_exp1_data_scale/reports/progress_state.json`，使用这组文件：

### 1) 准备 4000 条专用 split

```bash
python cli.py data split --config experiments/05_exp1_data_scale/configs/split_4000_lora_continue.yaml
```

### 2) 运行 LoRA 接续版 exp1

```bash
python experiments/05_exp1_data_scale/run_exp1_data_scale_lora_continue.py
```

说明：

- 微调方法：`LoRA`
- 数据规模：`[1800, 2000, 2200, ..., 3600]`
- 旧的 `200-1600` 结果不会被删掉，脚本会读取现有 `progress_state.json` 后继续追加。
- 汇总 CSV 和折线图仍写回原始主线结果：
  `experiments/05_exp1_data_scale/reports/exp1_data_scale_results.csv`
  和 `experiments/05_exp1_data_scale/reports/exp1_data_scale_chart.png`
- 新增阶段的日志和分规模评测报告会单独写到：
  `experiments/05_exp1_data_scale/logs/lora_continue/`
  和 `experiments/05_exp1_data_scale/reports/lora_continue/`
