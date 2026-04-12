# Exp12 最终方案总对比报告

## 结果总表

| 方案 | Parse OK | Exact Match | Action Match | Avg Latency (s) | Tokens/s | Avg VRAM (MB) | Max VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top18 Spec t8 thr0.65 (speculative) | 0.9500 | 0.1500 | 0.5500 | 12.9247 | 6.9170 | 0.0 | 0.0 |

## 简要结论

- 按当前配置的主排序规则，综合最优方案为 `Top18 Spec t8 thr0.65`，其 `exact_match_rate=0.1500`，`action_match_rate=0.5500`。
