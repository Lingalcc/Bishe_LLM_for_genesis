# 实验16 Exp12：最终方案总对比

本实验用于把论文中的“最终推荐方案”与两个核心基线放到**同一测试口径**下统一评测，输出可直接放入论文主表的结构化结果。

默认回答的问题是：

- 基座模型在当前任务上的表现如何？
- 训练后的主线 LoRA 模型提升了多少？
- 最终推荐模型是否在准确率与资源开销之间取得了更好的平衡？

## 默认对比方案

- `base_model`
  - 基座模型 `model/Qwen_Qwen2.5-3B-Instruct`
- `lora_rank4`
  - 当前主线 LoRA 适配器 `output/qwen2.5-3b-genesis-lora-rank-4`
- `final_merged`
  - 最终推荐合并模型 `model/qwen2.5-3b-genesis-merged`

你可以直接修改配置文件中的 `cases`，替换为你实际想用于论文结论的模型。

## 统一指标

每个方案都会在同一测试集上输出：

- `parse_ok_rate`
- `exact_match_rate`
- `action_match_rate`
- `avg_latency_sec`
- `avg_throughput_tps`
- `avg_peak_vram_mb`
- `max_peak_vram_mb`

## 运行方式

```bash
python experiments/16_exp12_final_compare/run_exp12_final_compare.py
```

如需自定义配置：

```bash
python experiments/16_exp12_final_compare/run_exp12_final_compare.py \
  --config experiments/16_exp12_final_compare/configs/compare.yaml
```

## 输出产物

运行完成后默认生成：

- `reports/final_compare_summary.json`
- `reports/final_compare_summary.csv`
- `reports/final_compare_summary.md`
- `reports/<case_name>_accuracy.json`
- `reports/run_meta.json`

## 解读建议

- 论文主表优先引用 `final_compare_summary.csv / md`
- 若答辩老师追问单个方案细节，可进一步打开对应的 `<case_name>_accuracy.json`
- 若最终推荐方案不是当前默认的 `final_merged`，只需要改配置，不需要改脚本
