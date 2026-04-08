# 实验17 Exp13：端到端仿真成功率评测

本实验用于把“自然语言指令 -> 模型生成 JSON action -> Genesis 执行”的整条链路量化成可引用的实验结果。

它重点回答的问题是：

- 模型输出的 JSON 在仿真闭环里有多大比例能成功执行？
- 端到端执行的平均耗时是多少？
- 失败主要发生在生成阶段还是执行阶段？
- 常见失败类型是什么？

## 默认评测口径

- 数据集：默认使用 `data_prepare/splits/test.json`
- 采样数：默认 `20`
- 推理模式：默认读取全局 `app.inference` 配置
- 运行方式：对每条指令执行一次完整端到端流程

## 核心指标

- `parse_ok_rate`
- `exact_match_rate`
- `action_match_rate`
- `execution_success_rate`
- `command_success_rate`
- `avg_end_to_end_sec`
- `avg_num_commands`
- `failure_type_counts`

其中：

- `execution_success_rate`
  - 一条样本内所有命令都执行成功，记为成功
- `command_success_rate`
  - 所有已执行命令中，状态为 `ok` 的比例

## 运行方式

```bash
python experiments/17_exp13_sim_success/run_exp13_sim_success.py
```

自定义配置：

```bash
python experiments/17_exp13_sim_success/run_exp13_sim_success.py \
  --config experiments/17_exp13_sim_success/configs/sim_success.yaml
```

## 输出产物

- `reports/sim_success_report.json`
- `reports/sim_success_summary.csv`
- `reports/sim_success_summary.md`
- `reports/sim_success_samples.json`
- `reports/run_meta.json`

## 结果解读建议

- 如果 `parse_ok_rate` 很高但 `execution_success_rate` 偏低，通常说明动作参数或动作顺序仍存在问题
- 如果失败大多集中在 `generation` 阶段，优先检查模型输出稳定性
- 如果失败大多集中在 `execution` 阶段，优先检查动作 schema、空间参数和仿真环境约束
