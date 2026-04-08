# 实验 13_exp11：层选择消融

本实验专门回答下面这个问题：

`Top-18 层 + rank8` 的收益，到底来自“重要层选择”，还是只是“少量层 + 更高 rank”碰巧有效？

为了解开这个因果，我们把消融补齐为 6 条线：

- `full_rank4`
  现成的全层 LoRA rank4 baseline，只评测，不重复训练。
- `random18_rank8`
  不打分，随机选 18 层 + rank8。
- `top18_rank8`
  先打分，再按分数选 Top-18 层 + rank8。
- `high18_rank8`
  只选高层 18 层 + rank8。
- `mid18_rank8`
  只选中层 18 层 + rank8。
- `low18_rank8`
  只选低层 18 层 + rank8。

这样设计后，结论会更干净：

- 如果 `top18_rank8` 明显优于 `random18_rank8`，说明收益不是随机层碰巧带来的。
- 如果 `top18_rank8` 明显优于 `high18/mid18/low18_rank8`，说明收益不只是“偏某个深度段”，而更接近“重要层被选中了”。
- 如果 `top18_rank8` 只和 `high18_rank8` 接近，那结论会变成“重要层主要集中在高层”，而不是“打分机制本身必不可少”。


## 目录结构

```text
experiments/13_exp11/
├── README.md
├── run_layer_scoring.py
├── run_exp11_ablation.py
└── configs/
    └── train_rank8_subset_template.yaml
```


## 运行前提

- 已有基础配置 [`configs/base.yaml`](/home/lin/Bishe_LLM_for_genesis/configs/base.yaml)
- 已有训练/验证/测试集：
  [`data_prepare/splits/train.json`](/home/lin/Bishe_LLM_for_genesis/data_prepare/splits/train.json)
  [`data_prepare/splits/val.json`](/home/lin/Bishe_LLM_for_genesis/data_prepare/splits/val.json)
  [`data_prepare/splits/test.json`](/home/lin/Bishe_LLM_for_genesis/data_prepare/splits/test.json)
- 已有基础模型：
  [`model/Qwen_Qwen2.5-3B-Instruct`](/home/lin/Bishe_LLM_for_genesis/model/Qwen_Qwen2.5-3B-Instruct)
- 已有 baseline 模型：
  [`model/qwen2.5-3b-genesis-lora-rank-4`](/home/lin/Bishe_LLM_for_genesis/model/qwen2.5-3b-genesis-lora-rank-4)


## 设计细节

- 训练公平性：
  所有 rank8 分支都自动复用同一个固定训练子集，默认取训练集前 600 条，写到 [`experiments/13_exp11/.cache/train_subset_600.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/.cache/train_subset_600.json)
- 打分数据：
  默认直接用这 600 条固定训练子集做层打分，避免“训练看的是一批数据，打分看的是另一批数据”
- 模块范围控制：
  所有 rank8 分支默认都用同一套 LoRA target 模块
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- 中层定义：
  对 32 层模型、18 层窗口来说，中层默认是连续的 `7-24` 层
- 高层定义：
  默认是连续的 `14-31` 层
- 低层定义：
  默认是连续的 `0-17` 层


## 推荐运行方式

先做 dry-run 检查，不真正启动训练：

```bash
python experiments/13_exp11/run_exp11_ablation.py --dry-run
```

只跑打分和训练，不做评测：

```bash
python experiments/13_exp11/run_exp11_ablation.py --skip-eval
```

跑完整实验：

```bash
python experiments/13_exp11/run_exp11_ablation.py
```

指定 GPU：

```bash
python experiments/13_exp11/run_exp11_ablation.py --gpus 0
```

如果你想换随机层的随机种子：

```bash
python experiments/13_exp11/run_exp11_ablation.py --random-layer-seed 123
```

如果你想提高层打分稳定性，把打分样本从 100 提高到 200：

```bash
python experiments/13_exp11/run_exp11_ablation.py --layer-sample-size 200
```

如果你想改统一的选层数量，例如从 18 改成 16：

```bash
python experiments/13_exp11/run_exp11_ablation.py --selected-layer-count 16
```


## 输出位置

训练输出会写到：

- [`output/exp13_exp11_ablation/random18_rank8`](/home/lin/Bishe_LLM_for_genesis/output/exp13_exp11_ablation/random18_rank8)
- [`output/exp13_exp11_ablation/top18_rank8`](/home/lin/Bishe_LLM_for_genesis/output/exp13_exp11_ablation/top18_rank8)
- [`output/exp13_exp11_ablation/high18_rank8`](/home/lin/Bishe_LLM_for_genesis/output/exp13_exp11_ablation/high18_rank8)
- [`output/exp13_exp11_ablation/mid18_rank8`](/home/lin/Bishe_LLM_for_genesis/output/exp13_exp11_ablation/mid18_rank8)
- [`output/exp13_exp11_ablation/low18_rank8`](/home/lin/Bishe_LLM_for_genesis/output/exp13_exp11_ablation/low18_rank8)

日志会写到：

- [`experiments/13_exp11/logs/layer_scoring.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/layer_scoring.log)
- [`experiments/13_exp11/logs/train_random18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/train_random18_rank8.log)
- [`experiments/13_exp11/logs/train_top18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/train_top18_rank8.log)
- [`experiments/13_exp11/logs/train_high18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/train_high18_rank8.log)
- [`experiments/13_exp11/logs/train_mid18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/train_mid18_rank8.log)
- [`experiments/13_exp11/logs/train_low18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/train_low18_rank8.log)
- [`experiments/13_exp11/logs/eval_full_rank4.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/eval_full_rank4.log)
- [`experiments/13_exp11/logs/eval_random18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/eval_random18_rank8.log)
- [`experiments/13_exp11/logs/eval_top18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/eval_top18_rank8.log)
- [`experiments/13_exp11/logs/eval_high18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/eval_high18_rank8.log)
- [`experiments/13_exp11/logs/eval_mid18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/eval_mid18_rank8.log)
- [`experiments/13_exp11/logs/eval_low18_rank8.log`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/logs/eval_low18_rank8.log)

报告会写到：

- 层打分报告：
  [`experiments/13_exp11/reports/layer_scores.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/layer_scores.json)
- 各分支层选择报告：
  [`experiments/13_exp11/reports/random18_rank8_layers.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/random18_rank8_layers.json)
  [`experiments/13_exp11/reports/top18_rank8_layers.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/top18_rank8_layers.json)
  [`experiments/13_exp11/reports/high18_rank8_layers.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/high18_rank8_layers.json)
  [`experiments/13_exp11/reports/mid18_rank8_layers.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/mid18_rank8_layers.json)
  [`experiments/13_exp11/reports/low18_rank8_layers.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/low18_rank8_layers.json)
- 各分支评测报告：
  [`experiments/13_exp11/reports/accuracy_report_full_rank4.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/accuracy_report_full_rank4.json)
  [`experiments/13_exp11/reports/accuracy_report_random18_rank8.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/accuracy_report_random18_rank8.json)
  [`experiments/13_exp11/reports/accuracy_report_top18_rank8.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/accuracy_report_top18_rank8.json)
  [`experiments/13_exp11/reports/accuracy_report_high18_rank8.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/accuracy_report_high18_rank8.json)
  [`experiments/13_exp11/reports/accuracy_report_mid18_rank8.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/accuracy_report_mid18_rank8.json)
  [`experiments/13_exp11/reports/accuracy_report_low18_rank8.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/accuracy_report_low18_rank8.json)
- 最终汇总：
  [`experiments/13_exp11/reports/comparison_summary.json`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/comparison_summary.json)
  [`experiments/13_exp11/reports/comparison_summary.md`](/home/lin/Bishe_LLM_for_genesis/experiments/13_exp11/reports/comparison_summary.md)


## 我建议你实际运行的命令

先检查命令拼接：

```bash
python experiments/13_exp11/run_exp11_ablation.py --dry-run
```

确认无误后，直接跑完整消融：

```bash
python experiments/13_exp11/run_exp11_ablation.py --gpus 0
```

如果你更关心打分稳定性，我更推荐这一条：

```bash
python experiments/13_exp11/run_exp11_ablation.py --gpus 0 --layer-sample-size 200
```
