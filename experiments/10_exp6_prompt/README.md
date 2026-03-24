# 实验10 Exp6：Prompt 对照消融实验

本实验用于对比同一模型在两种系统提示词下的空间推理与工具调用生成效果。

当前默认评测方式已经调整为：

- 从正式测试集 `data_prepare/splits/test.json` 中按固定随机种子抽样 100 条进行对照测试
- 继续保持同一模型、同一推理参数、同一采样规则，仅比较 `system_prompt`

另外，目录中的 `data/spatial_state_context_cases.json` 仍然保留，适合在需要做定向压力测试时单独使用。

Prompt 版本包括：

- `Baseline`
  - 直接复用仓库当前零样本系统提示词
- `Optimized`
  - 在零样本基线之上加入动作 schema、空间关系规则、few-shot 示例和严格输出约束

本实验目录与其他实验完全隔离，默认只复用底层评测基础设施 `src/eval_core/*`。

## 实验目标

- 验证 Prompt 工程是否能提升复杂 `STATE_CONTEXT` 场景下的 `exact_match_rate`
- 控制变量，保证以下条件一致：
  - 同一模型
  - 同一推理参数
  - 同一测试集
  - 唯一变化因素为 `system_prompt`

## 目录结构

```text
experiments/10_exp6_prompt/
├── README.md
├── run_exp6_prompt_ablation.py
├── configs/
│   └── compare.yaml
├── data/
│   └── spatial_state_context_cases.json
└── prompts/
    ├── baseline_system_prompt.txt
    └── optimized_system_prompt.txt
```

## 测试集设计

默认测试集来自 `data_prepare/splits/test.json`，运行时会在其中随机抽样 100 条样本，主要目的是：

- 提高评测样本量，避免 7 条样本带来的高方差
- 让 Prompt 对照结果更接近整体测试集分布
- 保持可复现性：固定 `seed=42`

手工构造的小型高难样本集 `data/spatial_state_context_cases.json` 仍可用于补充分析，重点覆盖：

- 相对方位：`右侧 / 左侧 / 前方 / 后方 / 上方`
- 复合方位：`右前方 / 左后方`
- 单位换算：`厘米 -> 米`
- 双目标解析：需要同时识别被搬运物体和参照物体

其中包含你要求的代表性样本：

- “将桌子上的红色方块移动到蓝色圆柱体右侧 5 厘米处”

## Prompt 版本

- `prompts/baseline_system_prompt.txt`
  - 与当前代码中的 `src/eval_core/prompting.py` 默认系统提示词保持一致
- `prompts/optimized_system_prompt.txt`
  - 增加动作接口约束、空间坐标映射、固定 pick-place 模板和 few-shot 示例

## 运行方法

推荐先用本地模型模式：

```bash
python experiments/10_exp6_prompt/run_exp6_prompt_ablation.py
```

如需只做配置检查而不真正调用模型：

```bash
python experiments/10_exp6_prompt/run_exp6_prompt_ablation.py --dry-run
```

如需只运行单个 Prompt 版本：

```bash
python experiments/10_exp6_prompt/run_exp6_prompt_ablation.py --variants baseline
```

## 配置说明

默认配置文件为 `configs/compare.yaml`，核心参数包括：

- `test.accuracy_eval`
  - 复用现有准确率评测参数
- `prompt_experiment.variants`
  - 定义 Prompt 版本、Prompt 文件和各自报告输出路径

默认设置为：

- `mode: local`
- `model_path: output/qwen2.5-3b-genesis-lora-rank-4`
- `temperature: 0.0`
- `test_file: data_prepare/splits/test.json`
- `num_samples: 100`

如果你希望切回手工高难样本集，只需要修改 `configs/compare.yaml` 中的 `test.accuracy_eval.test_file` 为 `experiments/10_exp6_prompt/data/spatial_state_context_cases.json`，并把 `num_samples` 调回 `7`。

如果你希望改成 API 模式，只需要修改 `configs/compare.yaml` 中的 `test.accuracy_eval.mode`、`model`、`api_key_env` 等参数。

## 输出产物

运行完成后会在 `experiments/10_exp6_prompt/reports/` 下生成：

- `baseline_accuracy.json`
  - Baseline Prompt 的评测报告
- `optimized_accuracy.json`
  - Optimized Prompt 的评测报告
- `prompt_ablation_summary.json`
  - 两组实验的结构化对比摘要
- `prompt_ablation_summary.md`
  - 适合直接贴入实验记录或论文草稿的 Markdown 摘要
- `run_meta.json`
  - 本次实验的环境、配置和数据快照

## 结果解读建议

- 优先观察 `exact_match_rate`
  - 它最能反映 Prompt 是否让模型输出更接近预期动作序列
- 同时关注 `parse_ok_rate`
  - 若明显提升，说明 schema 与输出约束降低了格式错误
- 再看 `action_match_rate`
  - 若动作序列更稳定但参数仍有偏差，通常会先体现在这里
