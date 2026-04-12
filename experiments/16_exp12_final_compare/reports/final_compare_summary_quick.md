# Exp12 最终方案总对比报告

## 结果总表

| 方案 | Parse OK | Exact Match | Action Match | Avg Latency (s) | Tokens/s | Avg VRAM (MB) | Max VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA Rank4 + Transformers 16bit (accuracy) | 1.0000 | 0.4000 | 0.6500 | 5.1479 | 17.3078 | 6082.0 | 6082.0 |
| LoRA Rank4 + Speculative (speculative) | 1.0000 | 0.2000 | 0.6500 | 2.4135 | 31.4060 | 7026.8 | 7030.0 |
| Top18 Rank8 + Transformers 16bit (accuracy) | 1.0000 | 0.4500 | 0.7000 | 4.0928 | 20.4306 | 6944.0 | 6944.0 |
| Top18 Rank8 + Speculative (speculative) | 0.9500 | 0.1500 | 0.5500 | 2.6020 | 28.8620 | 7038.8 | 7042.0 |

## 简要结论

- 按当前配置的主排序规则，综合最优方案为 `Top18 Rank8 + Transformers 16bit`，其 `exact_match_rate=0.4500`，`action_match_rate=0.7000`。
