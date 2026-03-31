# 实验14 Exp10：DeepConf + Speculative Decoding

本实验把 `Deep Think with Confidence` 的“轨迹置信度筛选”思想迁移到当前的机器人 ToolCall / JSON 生成任务，并与现有投机采样实验结合。

核心思路分成两部分：

- `Speculative Decoding` 负责更快地生成候选结果。
- `DeepConf-style Reranking` 负责在多个候选里，用主模型对生成轨迹的置信度评分来选更优结果。

实验目录：[`experiments/14_exp10_deepconf_speculative`](/home/lin/Bishe_LLM_for_genesis/experiments/14_exp10_deepconf_speculative)

## 1. 为什么这样设计

直接把 DeepConf 论文里的方法照搬到“层内提前退出”上并不合适，因为论文主要针对的是：

- 多条 reasoning traces
- 基于 token / trace confidence 的筛轨迹
- 目标是减少无效推理轨迹的 token 开销

而你当前任务更像是：

- 多个候选 JSON / ToolCall 输出
- 需要在候选中挑出更正确的一条
- 同时希望候选生成本身尽量快

因此本实验采用更贴近论文思想、也更贴近任务本质的联合方案：

1. 用 target-only 或 speculative decoding 生成多个候选。
2. 用 target model 对每个候选做 teacher-forcing 评分，提取 token-level confidence。
3. 计算 DeepConf 风格的轨迹置信度分数。
4. 对可解析 JSON 候选做 confidence-weighted vote，选最终输出。

## 2. 实验分支

默认会跑 8 个分支：

- `baseline_off`
  普通 greedy decoding，单次生成。
- `speculative_on`
  speculative decoding，单次生成。
- `deepconf_target_k2` / `deepconf_target_k3` / `deepconf_target_k4`
  不用 assistant，分别生成 `2/3/4` 个候选，再做 DeepConf 风格置信度重排序。
- `deepconf_speculative_k2` / `deepconf_speculative_k3` / `deepconf_speculative_k4`
  用 assistant 做 speculative candidate generation，再做 DeepConf 风格置信度重排序。

这样可以回答三个问题：

- 准确率提升是否来自 DeepConf 风格多候选筛选？
- 在同样是多候选 rerank 的前提下，speculative 是否带来额外速度收益？
- `2/3/4` 个候选数之间，哪个点的速度/精度更平衡？

## 3. 置信度打分方式

对于每个候选输出，脚本会让主模型在 teacher-forcing 模式下重新看一遍 `prompt + prediction`，并在每个生成 token 位置上统计：

- `avg_confidence`
  全局平均 token confidence，这里统计的是 `top-k` 概率质量而不是简单均值，区分度会明显高于旧版实现
- `bottom_confidence`
  最低置信 token 子集的平均值
- `tail_confidence`
  轨迹尾部 token 的平均置信度
- `avg_actual_token_prob`
  实际生成 token 的平均概率

最后按配置中的权重合成为：

- `deepconf_score`

若候选能成功解析为合法 ToolCall JSON，则会参与 `confidence-weighted vote`。若多个候选 canonical command 相同，则它们的 `deepconf_score` 会累加，分数最高的组胜出，再从该组里取置信度最高的一条作为最终结果。

## 4. 配置文件

默认配置见 [`configs/deepconf_speculative.yaml`](/home/lin/Bishe_LLM_for_genesis/experiments/14_exp10_deepconf_speculative/configs/deepconf_speculative.yaml)

默认关键参数如下：

- target：`model/qwen2.5-3b-genesis-merged`
- assistant：`model/qwen2.5-0.5b-genesis-merged`
- `num_samples=50`
- `temperature=0.7`
- `top_p=0.95`
- `num_candidates=4`
- `deepconf_candidate_counts=[2,3,4]`
- `assistant_num_tokens=8`
- `assistant_confidence_threshold=0.55`

DeepConf 评分相关参数：

- `top_k_confidence=20`
- `bottom_fraction=0.2`
- `tail_fraction=0.2`
- `avg_weight=0.35`
- `bottom_weight=0.35`
- `tail_weight=0.15`
- `actual_prob_weight=0.15`

候选多样性增强参数：

- `candidate_temperature_step=0.08`
- `candidate_top_p_step=0.02`
- `candidate_duplicate_retry_limit=2`
- `candidate_retry_temperature_bump=0.06`
- `candidate_retry_top_p_bump=0.01`
- `candidate_retry_seed_stride=97`

这些参数用于：

- 让不同候选带一点可控的采样差异
- 当新候选和已有候选完全重复时，自动触发重采样

## 5. 运行方式

### 5.1 Dry Run

```bash
python experiments/14_exp10_deepconf_speculative/run_deepconf_speculative_benchmark.py --dry-run
```

### 5.2 跑完整扫描

```bash
python experiments/14_exp10_deepconf_speculative/run_deepconf_speculative_benchmark.py
```

默认会自动展开成 `baseline/speculative + k2/k3/k4` 的完整扫描。

### 5.3 只跑 DeepConf + Speculative

```bash
python experiments/14_exp10_deepconf_speculative/run_deepconf_speculative_benchmark.py \
  --cases deepconf_speculative
```

### 5.4 调整候选数

```bash
python experiments/14_exp10_deepconf_speculative/run_deepconf_speculative_benchmark.py \
  --deepconf-candidate-counts 2,4,6
```

### 5.5 调整采样强度

```bash
python experiments/14_exp10_deepconf_speculative/run_deepconf_speculative_benchmark.py \
  --temperature 0.6 \
  --top-p 0.9
```

### 5.6 只跑某个候选数

```bash
python experiments/14_exp10_deepconf_speculative/run_deepconf_speculative_benchmark.py \
  --cases deepconf_speculative \
  --deepconf-candidate-counts 3
```

### 5.7 当前结果归档

这次初始正式结果已经另存到：

- [`reports/archive/20260331_initial_run/deepconf_speculative_report.json`](/home/lin/Bishe_LLM_for_genesis/experiments/14_exp10_deepconf_speculative/reports/archive/20260331_initial_run/deepconf_speculative_report.json)
- [`reports/archive/20260331_initial_run/run_meta.json`](/home/lin/Bishe_LLM_for_genesis/experiments/14_exp10_deepconf_speculative/reports/archive/20260331_initial_run/run_meta.json)

## 6. 输出结果

默认输出到：

- [`reports/deepconf_speculative_report.json`](/home/lin/Bishe_LLM_for_genesis/experiments/14_exp10_deepconf_speculative/reports/deepconf_speculative_report.json)

结果里重点看：

- `parse_ok_rate`
- `exact_match_rate`
- `action_match_rate`
- `avg_latency_sec_per_sample`
- `avg_generation_time_sec_per_sample`
- `avg_scoring_time_sec_per_sample`
- `candidate_token_throughput_tps`
- `selected_token_throughput_tps`
- `avg_unique_predictions_per_sample`
- `avg_unique_action_signatures_per_sample`
- `avg_duplicate_prediction_rate`
- `avg_duplicate_action_signature_rate`
- `avg_resample_attempts_per_sample`
- `selected_nonzero_candidate_rate`

以及最终的 `comparison`：

- `speculative_accuracy_gain_vs_baseline`
- `speculative_latency_speedup_vs_baseline`
- `deepconf_speculative_k*_accuracy_gain_vs_speculative`
- `deepconf_speculative_k*_latency_speedup_vs_deepconf_target_k*`
- `deepconf_speculative_k*_action_gain_vs_speculative`
- `best_deepconf_target_by_exact`
- `best_deepconf_speculative_by_exact`

## 7. 如何解读

如果你看到：

- 某个 `deepconf_speculative_k*` 相比 `speculative_on` 的 `exact_match_rate` / `action_match_rate` 提升
- 同时它又比对应的 `deepconf_target_k*` 更快

那么就说明：

- DeepConf 风格 rerank 确实改善了准确率
- speculative decoding 又把多候选代价压下来了

这正是“把论文方法与投机采样相结合”的目标形态。

如果你同时看到：

- `avg_unique_predictions_per_sample` 很低
- `avg_duplicate_prediction_rate` 很高
- `selected_nonzero_candidate_rate` 接近 0

那通常说明当前候选生成没有真正分叉，rerank 几乎没有发挥空间。

## 8. 当前边界

当前版本实现的是 `offline reranking`，不是论文里最激进的 `online early stop`。也就是说：

- 已经做到了“speculative 负责快，DeepConf 负责挑”
- 但还没有做到“在候选生成过程中，低置信轨迹直接中途终止”

如果这版实验有效，下一步就值得继续做：

- speculative candidate online pruning
- confidence-aware dynamic candidate budget
- confidence-aware acceptance scheduling
