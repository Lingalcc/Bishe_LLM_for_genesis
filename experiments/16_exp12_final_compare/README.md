# 实验16 Exp12：最终方案总对比

本实验用于把论文中的最终候选方案放到**同一测试口径**下统一评测，输出可直接放入论文主表的结构化结果。

默认回答的问题是：

- 原始 `LoRA rank4` 模型在 `Transformers 16bit` 部署下的准确率与性能如何？
- 同一个 `LoRA rank4` 模型启用投机解码后，延迟和吞吐能改善多少？
- `Top18 rank8` 模型在普通部署与投机解码下，是否优于 `LoRA rank4`？

## 默认对比方案

- `lora_rank4_transformers_16bit`
  - 原始 `LoRA rank4` 微调模型，`Transformers 16bit` 普通解码
- `lora_rank4_speculative`
  - 原始 `LoRA rank4` 微调模型，`Transformers` 投机解码
  - 草稿模型固定为 `model/qwen2.5-0.5b-genesis-merged`
- `top18_rank8_transformers_16bit`
  - `Top18 rank8` 微调模型，`Transformers 16bit` 普通解码
- `top18_rank8_speculative`
  - `Top18 rank8` 微调模型，`Transformers` 投机解码
  - 草稿模型固定为 `model/qwen2.5-0.5b-genesis-merged`

其中普通 case 走现有 accuracy 评测链路，投机解码 case 复用 `exp8` 的 speculative decoding 实现，但会额外在总表里补齐 `exact_match_rate / action_match_rate`，保证四组结果可以直接横向比较。

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
- `reports/<case_name>.json`
- `reports/run_meta.json`

## 解读建议

- 论文主表优先引用 `final_compare_summary.csv / md`
- 若答辩老师追问单个方案细节，可进一步打开对应的单 case JSON 报告
- 若你要替换模型路径或 speculative 参数，只需要修改 `configs/compare.yaml`
