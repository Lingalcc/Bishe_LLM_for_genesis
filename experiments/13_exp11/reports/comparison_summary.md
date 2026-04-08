# exp11 层选择消融结果

- 实验问题：收益是否来自重要层选择，而不是少量层 + 更高 rank 的偶然组合。
- baseline：`full_rank4`，模型路径 `model/qwen2.5-3b-genesis-lora-rank-4`
- 固定训练集：`600` 条，路径 `experiments/13_exp11/.cache/train_subset_600.json`
- rank8 分支优胜者：`mid18_rank8`

## 指标总表

| 模型 | 选层方式 | parse_ok_rate | exact_match_rate | action_match_rate | avg_latency_sec | avg_throughput_tps | avg_peak_vram_mb |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_rank4 | 全层 rank4 baseline | 1.0000 | 0.3000 | 0.5850 | 4.4022 | 19.8389 | 6610.0 |
| random18_rank8 | 不打分，随机选 18 层 | 0.9950 | 0.3050 | 0.5900 | 3.3350 | 24.6256 | 6610.0 |
| top18_rank8 | 按打分选 Top-18 层 | 0.9900 | 0.2800 | 0.5800 | 3.4148 | 24.5069 | 6610.0 |
| high18_rank8 | 只选高层 18 层 | 0.9850 | 0.2950 | 0.5650 | 3.3106 | 24.9060 | 6610.0 |
| mid18_rank8 | 只选中层 18 层 | 0.9900 | 0.3200 | 0.5800 | 3.4577 | 24.3144 | 6610.0 |
| low18_rank8 | 只选低层 18 层 | 0.4450 | 0.1250 | 0.2200 | 3.5370 | 23.2874 | 6610.0 |

## 各 rank8 分支相对 full_rank4 的差值

### random18_rank8

- 选层方式：`不打分，随机选 18 层`
- 层编号：`0, 1, 2, 3, 4, 6, 7, 13, 14, 15, 17, 18, 21, 23, 24, 29, 33, 34`
- `parse_ok_rate_delta`: -0.0050
- `exact_match_rate_delta`: +0.0050
- `action_match_rate_delta`: +0.0050
- `avg_latency_sec_delta`: -1.0671
- `avg_throughput_tps_delta`: +4.7867
- `avg_peak_vram_mb_delta`: +0.0

### top18_rank8

- 选层方式：`按打分选 Top-18 层`
- 层编号：`0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27`
- `parse_ok_rate_delta`: -0.0100
- `exact_match_rate_delta`: -0.0200
- `action_match_rate_delta`: -0.0050
- `avg_latency_sec_delta`: -0.9874
- `avg_throughput_tps_delta`: +4.6680
- `avg_peak_vram_mb_delta`: +0.0

### high18_rank8

- 选层方式：`只选高层 18 层`
- 层编号：`18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35`
- `parse_ok_rate_delta`: -0.0150
- `exact_match_rate_delta`: -0.0050
- `action_match_rate_delta`: -0.0200
- `avg_latency_sec_delta`: -1.0916
- `avg_throughput_tps_delta`: +5.0671
- `avg_peak_vram_mb_delta`: +0.0

### mid18_rank8

- 选层方式：`只选中层 18 层`
- 层编号：`9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26`
- `parse_ok_rate_delta`: -0.0100
- `exact_match_rate_delta`: +0.0200
- `action_match_rate_delta`: -0.0050
- `avg_latency_sec_delta`: -0.9445
- `avg_throughput_tps_delta`: +4.4755
- `avg_peak_vram_mb_delta`: +0.0

### low18_rank8

- 选层方式：`只选低层 18 层`
- 层编号：`0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17`
- `parse_ok_rate_delta`: -0.5550
- `exact_match_rate_delta`: -0.1750
- `action_match_rate_delta`: -0.3650
- `avg_latency_sec_delta`: -0.8652
- `avg_throughput_tps_delta`: +3.4485
- `avg_peak_vram_mb_delta`: +0.0
