# Exp18 Top-K 重要层扫描实验

- 实验问题：按重要性排名选前 K 层做 rank8 微调时，K 如何影响任务性能与推理成本。
- baseline：`model/qwen2.5-3b-genesis-lora-rank-4`
- 固定训练集：`600` 条（`experiments/22_exp18_topk_scan/.cache/train_subset_600.json`）

## Baseline

- `parse_ok_rate`: 1.0000
- `exact_match_rate`: 0.3000
- `action_match_rate`: 0.5850

## K 扫描结果

| K | Parse OK | Exact Match | Action Match | Avg Latency (s) | Avg Throughput (tokens/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.0300 | 0.0050 | 0.0100 | 3.0314 | 29.6716 |
| 8 | 0.0650 | 0.0000 | 0.0250 | 2.8464 | 27.9948 |
| 12 | 0.0600 | 0.0350 | 0.0400 | 3.1567 | 26.6734 |
| 18 | 0.9950 | 0.3050 | 0.5800 | 3.3725 | 24.7324 |
| 24 | 0.9900 | 0.3100 | 0.5900 | 3.6882 | 22.8203 |
| 28 | 0.9900 | 0.3100 | 0.5900 | 3.7743 | 21.6848 |

## 最佳 Exact Match

- K：`24`
- case：`top24_rank8`

## 最佳 Action Match

- K：`24`
- case：`top24_rank8`

## 最佳 Parse OK

- K：`18`
- case：`top18_rank8`

## 最低时延

- K：`8`
- case：`top8_rank8`

## 最高吞吐

- K：`4`
- case：`top4_rank8`
