# 实验24 / Exp20：复杂、噪声与长文本指令鲁棒性分析

本实验用于回应老师关于“复杂指令、噪声指令、长文本指令下最终微调推理方案是否会退化”的意见。实验固定底层目标 action，不改变标准答案，只改变自然语言输入形式，因此可以直接观察输入表达复杂度对结构化输出的影响。

## 实验目标

- 按输入类型分别统计 `JSON 可解析率`、`exact_match_rate`、`action_match_rate`。
- 判断复杂、噪声、长文本、边界/含糊表达是否造成相对标准指令的性能下降。
- 将错误归因为 `语义理解`、`动作顺序`、`参数映射` 或 `JSON 解析`。
- 统计输入长度与推理时延的相关性，回答“时延是否随文本长度增加”。

## 输入类型

| 输入类型 | 构造方式 | 重点观察 |
| --- | --- | --- |
| 标准指令 | 直接使用原测试集指令 | 作为对照组，给出基础可解析率、动作匹配率和时延 |
| 复杂多步指令 | 优先选择多 commands 样本，并强调顺序约束 | 是否出现动作遗漏、动作顺序调换、步骤合并 |
| 含噪声/冗余描述指令 | 在核心指令前后加入背景、备注和重复提醒 | 是否误把背景当动作，或忽略真正任务 |
| 长文本指令 | 加入长段实验记录，再显式标出当前用户指令 | JSON 是否仍可解析，动作是否仍匹配，时延是否上升 |
| 边界或含糊指令 | 加入默认值、边界、安全距离等说明 | 是否出现参数发明、默认值误用或语义理解错误 |

## 运行方式

只生成 stress set 和实验设计说明：

```bash
python experiments/24_exp20_instruction_robustness/run_exp20_instruction_robustness.py
```

使用最终方案 `Top18Rank8 + vLLM + AWQ/compressed-tensors` 实测：

```bash
python experiments/24_exp20_instruction_robustness/run_exp20_instruction_robustness.py --mode local
```

配置文件默认设置了 `local.skip_vllm_compat_check: true`，会自动导出 `LLM_GENESIS_SKIP_VLLM_COMPAT_CHECK=1`，用于跳过 vLLM 与 compressed-tensors 版本组合的保守拦截。若需要严格检查 requirements 中锁定的版本组合，可运行：

```bash
python experiments/24_exp20_instruction_robustness/run_exp20_instruction_robustness.py \
  --mode local \
  --strict-vllm-compat-check
```

如果已经保存了预测结果，也可以离线复算指标：

```bash
python experiments/24_exp20_instruction_robustness/run_exp20_instruction_robustness.py \
  --mode predictions \
  --predictions-file experiments/24_exp20_instruction_robustness/reports/predictions.json
```

预测文件可以是按 stress set 顺序排列的 JSON list，也可以是以 `case_id` 为 key 的 JSON object。每条预测可直接是字符串，也可以是包含 `prediction`、`latency_sec`、`throughput_tps`、`peak_vram_mb` 字段的对象。

## 输出文件

- `reports/exp20_stress_dataset.json`
- `reports/exp20_instruction_robustness_summary.json`
- `reports/exp20_instruction_robustness_summary.csv`
- `reports/exp20_instruction_robustness_summary.md`
- `reports/exp20_instruction_robustness_details.jsonl`

## 论文写法建议

论文中可按五类输入分别讨论：标准指令作为对照；复杂多步指令主要关注动作顺序和动作遗漏；噪声/冗余指令主要关注语义理解是否被干扰；长文本指令除准确率外必须报告时延变化；边界或含糊指令主要关注默认值和参数映射错误。
