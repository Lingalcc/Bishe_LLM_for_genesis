# Exp17 IIDEE

本实验在现有 `自然语言指令 -> 结构化动作 JSON` 任务上，实现“重要性感知动态早退（IIDEE）”的最小可运行版本。

## 目标

- 在现有 HuggingFace / PEFT 本地推理链路上增加可开关的动态早退能力
- 复用已有静态层重要性分析结果，构造累计重要性 `C_l`
- 在候选退出层集合 `E` 上执行统一阈值版本判定
  - `C_l >= tau_I`
  - `max_prob >= tau_C`
- 输出与现有 accuracy eval 兼容的性能与任务指标

## 关键文件

- `src/eval_core/importance_loader.py`
  读取仓库已有的 `layer_scores.json`，并在运行时补齐：
  `layer_scores / layer_probs / cum_importance`
- `src/eval_core/early_exit_policy.py`
  封装退出层集合、累计重要性阈值、置信度阈值判定
- `src/eval_core/generation_early_exit.py`
  在 token 生成循环中执行真实停层
- `experiments/21_exp17_iidee/run_exp17_iidee.py`
  Exp17 运行入口
- `experiments/21_exp17_iidee/configs/iidee.yaml`
  默认实验配置

## 默认配置

- 模型：`model/qwen2.5-3b-top18-rank8-merged`
- 重要性文件：`experiments/13_exp11/reports/layer_scores.json`
- 候选退出层：`10,14,18,22,28`
- `tau_importance=0.6`
- `tau_confidence=0.2`
- `early_exit_warmup_tokens=16`
- `early_exit_min_streak=4`
- `early_exit_protect_open_string=false`
- `early_exit_draft_only_layers=null`
- `early_exit_fallback_on_invalid_json=true`

## 运行方式

```bash
python experiments/21_exp17_iidee/run_exp17_iidee.py
```

当前默认值是“保守版 JSON 早退”：

- 先让前 `16` 个生成 token 保持当前深度
- 只有连续 `4` 个 token 都命中退出条件，才会对下一 token 提交降层

这样做的目的不是最大化降层，而是先避免结构化 JSON 在前几个关键 token 上被错误早退破坏。

另外默认开启了一个任务级保护：

- 如果早退结果不是可解析 JSON
- 会自动回退到同一模型的 full-depth 重跑

这更符合当前任务的目标，因为这里的输出必须是可执行动作 JSON，而不是一般开放式文本。

做一个最小冒烟：

```bash
python experiments/21_exp17_iidee/run_exp17_iidee.py \
  --num-samples 2 \
  --max-new-tokens 96
```

关闭早退，得到 full-depth 对照：

```bash
python experiments/21_exp17_iidee/run_exp17_iidee.py \
  --disable-early-exit \
  --report-file experiments/21_exp17_iidee/reports/exp17_full_depth_report.json
```

更激进的调参示例：

```bash
python experiments/21_exp17_iidee/run_exp17_iidee.py \
  --tau-confidence 0.2 \
  --early-exit-warmup-tokens 8 \
  --early-exit-min-streak 2 \
  --early-exit-fallback-on-invalid-json
```

实验 A：字符串状态保护

```bash
python experiments/21_exp17_iidee/run_exp17_iidee.py \
  --config experiments/21_exp17_iidee/configs/expA_string_guard.yaml \
  --num-samples 5
```

实验 B：22 层直接退出

```bash
python experiments/21_exp17_iidee/run_exp17_iidee.py \
  --config experiments/21_exp17_iidee/configs/expB_layer22_direct_exit.yaml \
  --num-samples 5
```

实验 B：22 层只做草拟 + 深层校验

```bash
python experiments/21_exp17_iidee/run_exp17_iidee.py \
  --config experiments/21_exp17_iidee/configs/expB_layer22_draft_verify.yaml \
  --num-samples 5
```

## 输出指标

报告 JSON 会保留现有 accuracy eval 指标，并新增 `early_exit` 汇总：

- 每条样本平均退出层
- 全部 token 的退出层直方图
- 平均退出层（按 sample / 按 token）
- 每个候选层的探针统计：
  `avg_max_prob / meets_importance_rate / meets_confidence_rate / exit_rate`
- fallback 使用次数
- open string 保护拦截次数
- draft-only 候选次数
- draft 校验一致 / 不一致次数
- 平均时延
- 平均 tokens/s
- JSON 可解析率
- 精确匹配率
- 动作匹配率

如果存在 parse fail，报告里还会新增 `parse_failure_diagnostics`：

- 第一个非法位置的错误类型统计
  例如：`缺右括号 / 缺右中括号 / 少逗号 / 引号未闭合 / 非法键名 / value 类型错`
- 第一个出错 token 的类别统计
  `结构 token / 动作 token / 数值 token`
- 第一个出错 token 对应的退出层统计
- 每条失败样本的明细
  包括 `dataset_index / char_position / token_text / token_category / exit_layer / context_excerpt`

如果开启了 `early_exit_fallback_on_invalid_json=true`，即便最终结果被 full-depth 回退救回，报告里也会额外保留
`early_exit_parse_break_diagnostics`：

- 统计“早退版本最先在哪里把 JSON 写坏”
- 不会因为 fallback 成功就丢失坏点信息
- 适合直接回答“哪些 token 最先打崩 parse、它们退在哪一层”

## 说明

本仓库已经存在静态层重要性文件，所以当前实现直接读取旧格式 `layer_scores.json`，并在运行时推导 `cum_importance`，不额外修改原有层打分脚本。

为了保证 KV cache 一致性，MVP 实现采用“有效深度单调不升”的策略：

- 一旦某个 token 在较浅层退出
- 后续 token 的最大可用深度不会再升高

当前版本还加入了两层工程保护：

- `warmup_tokens`
  前若干 token 不提交降层，先稳定 JSON 结构前缀
- `min_exit_streak`
  连续多 token 命中退出条件后，才对下一 token 提交降层

这保证了当前版本是真实停层，而不是“全层跑完后再事后统计”的伪早退，同时也避免单个假阳性 token 过早把整条序列拖入低层生成。
