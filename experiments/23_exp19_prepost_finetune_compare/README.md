# 实验 23 / Exp19：微调前后模型对比

本实验用于补充论文中的横向对比：未微调基座模型与多个微调后模型在结构化机器人动作生成任务上的准确率和推理性能差异。

## 实验目标

- 对比微调前 `Qwen2.5-3B-Instruct` 与微调后模型的 `parse_ok_rate`、`exact_match_rate`、`action_match_rate`。
- 同时比较平均延迟、吞吐和显存占用，避免只报告准确率。
- 复用仓库内已有 accuracy report，不重复训练模型。

## 运行方式

仅汇总已有报告：

```bash
python experiments/23_exp19_prepost_finetune_compare/run_exp19_prepost_finetune_compare.py
```

如果未微调基座模型报告不存在，补跑基座模型评测后再汇总：

```bash
python experiments/23_exp19_prepost_finetune_compare/run_exp19_prepost_finetune_compare.py --eval-missing-base
```

如果想先快速补跑小样本基座评测，可调整样本数：

```bash
python experiments/23_exp19_prepost_finetune_compare/run_exp19_prepost_finetune_compare.py --eval-missing-base --base-num-samples 50
```

8GB 显存下若 FP16 基座模型评测显存不足，可先用量化得到参考结果：

```bash
python experiments/23_exp19_prepost_finetune_compare/run_exp19_prepost_finetune_compare.py --eval-missing-base --base-quantization 4bit
```

## 输出文件

- `reports/exp19_prepost_finetune_summary.json`
- `reports/exp19_prepost_finetune_summary.csv`
- `reports/exp19_prepost_finetune_summary.md`
- `reports/exp19_accuracy_comparison.png`
- `reports/exp19_performance_comparison.png`
- `reports/accuracy_report_pretrain_base.json`：仅在使用 `--eval-missing-base` 且报告缺失时生成

## 默认纳入的模型

- 微调前：`model/Qwen_Qwen2.5-3B-Instruct`
- 全层 LoRA：`LoRA Full r4`
- LoRA rank 扫描：`r8`、`r16`、`r32`、`r64`
- 重要层选择：`Top18 r8`、`Top24 r8`、`Top28 r8`
- 方法对比：`QLoRA`、`DoRA`、`GaLore`

若基座模型报告尚未生成，汇总文件会记录缺失项，并给出补跑命令。
