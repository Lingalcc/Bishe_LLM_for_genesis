# Exp12 最终方案总对比报告

## 结果总表

| 方案 | Parse OK | Exact Match | Action Match | Avg Latency (s) | Tokens/s | Avg VRAM (MB) | Max VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA Rank4 Spec Smoke (speculative) | 1.0000 | 0.2000 | 0.8000 | 2.0570 | 35.4880 | 7277.6 | 7280.0 |

## 简要结论

- 按当前配置的主排序规则，综合最优方案为 `LoRA Rank4 Spec Smoke`，其 `exact_match_rate=0.2000`，`action_match_rate=0.8000`。
