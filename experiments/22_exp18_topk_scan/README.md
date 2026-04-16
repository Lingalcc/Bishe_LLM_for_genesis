# Exp18 重要层 Top-K 扫描

本实验回答的问题是：

- 当前 `Top18 rank8` 的收益，是否会随着重要层数量 `K` 改变而继续上升或下降？
- 最优 `K` 是否真的是 `18`，还是只是此前实验里的固定值？
- 当 `K` 变小时，参数更集中是否更高效；当 `K` 变大时，是否会接近全层微调的效果上限？

这里的 `Top-K` 指的是：

- 先用 `layer_scores.json` 对层做重要性排序
- 再取前 `K` 层作为 LoRA 注入层
- 每个分支统一使用 `rank8`
- 最后分别评测性能

不是解码采样里的 `top_k`。

## 默认设置

- 基础模型：`model/Qwen_Qwen2.5-3B-Instruct`
- baseline：`model/qwen2.5-3b-genesis-lora-rank-4`
- 固定训练子集：`600` 条
- 层打分样本数：`100`
- 模块白名单：
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- 扫描 K：
  `4, 8, 12, 18, 24, 28`

## 运行方式

先做 dry-run，确认每个 K 的命令与输出目录都正确：

```bash
python experiments/22_exp18_topk_scan/run_exp18_topk_scan.py --dry-run
```

完整运行：

```bash
python experiments/22_exp18_topk_scan/run_exp18_topk_scan.py --gpus 0
```

结果画成折线图：

```bash
python experiments/22_exp18_topk_scan/plot_exp18_topk_lines.py
```

如果想指定输入 summary 或输出文件名：

```bash
python experiments/22_exp18_topk_scan/plot_exp18_topk_lines.py \
  --summary-path experiments/22_exp18_topk_scan/reports/exp18_topk_summary.json \
  --output-path experiments/22_exp18_topk_scan/reports/exp18_topk_linecharts.png
```

自定义 K 扫描范围：

```bash
python experiments/22_exp18_topk_scan/run_exp18_topk_scan.py \
  --gpus 0 \
  --k-values 6,10,14,18,22,26,30
```

如果你只想先训练，不做评测：

```bash
python experiments/22_exp18_topk_scan/run_exp18_topk_scan.py \
  --gpus 0 \
  --skip-eval
```

如果你已经有各个 K 的模型目录，只想补评测：

```bash
python experiments/22_exp18_topk_scan/run_exp18_topk_scan.py \
  --skip-train
```

## 输出内容

训练输出：

- `output/exp18_topk_scan/top4_rank8`
- `output/exp18_topk_scan/top8_rank8`
- `output/exp18_topk_scan/top12_rank8`
- `output/exp18_topk_scan/top18_rank8`
- `output/exp18_topk_scan/top24_rank8`
- `output/exp18_topk_scan/top28_rank8`

日志输出：

- `experiments/22_exp18_topk_scan/logs/layer_scoring.log`
- `experiments/22_exp18_topk_scan/logs/train_top{k}_rank8.log`
- `experiments/22_exp18_topk_scan/logs/eval_top{k}_rank8.log`
- `experiments/22_exp18_topk_scan/logs/eval_full_rank4.log`

报告输出：

- `experiments/22_exp18_topk_scan/reports/layer_scores.json`
- `experiments/22_exp18_topk_scan/reports/top{k}_rank8_layers.json`
- `experiments/22_exp18_topk_scan/reports/accuracy_report_top{k}_rank8.json`
- `experiments/22_exp18_topk_scan/reports/exp18_topk_summary.json`
- `experiments/22_exp18_topk_scan/reports/exp18_topk_summary.csv`
- `experiments/22_exp18_topk_scan/reports/exp18_topk_summary.md`
- `experiments/22_exp18_topk_scan/reports/exp18_topk_linecharts.png`

## 建议如何解读

先看三组核心指标：

- `exact_match_rate`
- `action_match_rate`
- `parse_ok_rate`

再结合：

- `avg_latency_sec`
- `avg_throughput_tps`
- `avg_peak_vram_mb`

推荐的判断方式是：

- 如果 `K` 从小到大时精度持续上升，说明重要层覆盖还不够
- 如果精度在某个 `K` 后趋于平稳，说明已经接近收益饱和
- 如果 `K` 继续变大但精度不升反降，说明更多层并没有带来真正有效的参数利用

这类结果很适合直接做一张：

- 横轴 `K`
- 纵轴 `exact_match_rate / action_match_rate`

通常会比只报一个 `Top18` 点更有说服力。
