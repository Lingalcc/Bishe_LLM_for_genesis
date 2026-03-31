# Exp14 初始正式结果归档

本目录保存的是 `Exp14 DeepConf + Speculative` 在第一次正式实现后的原始结果，用于后续与修复版 / 扩展版结果做对照。

归档文件：

- `deepconf_speculative_report.json`
- `run_meta.json`

当时的主要结论：

- `speculative_on` 相比 `baseline_off` 明显更快，但 `exact_match_rate` 从 `0.42` 降到了 `0.38`
- `deepconf_target` 把 `exact_match_rate` 提到了 `0.46`，但延迟升到了 `18.25s/sample`
- `deepconf_speculative` 把 `exact_match_rate` 拉回到 `0.42`，`action_match_rate` 提到 `0.68`
- 但 `deepconf_speculative` 的延迟仍为 `11.04s/sample`，明显高于纯 speculative 的 `2.45s/sample`

因此，这份初始结果支持的判断是：

- `DeepConf-style rerank` 能补回 speculative 丢失的一部分精度
- 但旧版实现还没有达到“同时提升 speculative 的速度和准确率”

旧版实现暴露出的关键问题：

- 候选多样性不足，很多样本的多个候选几乎相同
- `avg_confidence` 的区分度偏弱，旧版定义容易塌缩到接近常数
- 旧版只支持单个 `num_candidates=4`，无法判断 `2/3/4` 哪个更划算

后续修复版在此基础上新增了：

- `deepconf_candidate_counts=[2,3,4]` 扫描
- 候选重复时自动重采样
- 候选采样温度 / top-p 微扰
- 更多多样性与重排序诊断字段
