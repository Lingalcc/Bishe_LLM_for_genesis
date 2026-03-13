# 02_finetune_exp — 微调与对比基准

目标：基于 LLaMA-Factory 完成 SFT 微调，并输出微调前后准确率对比报告。

## 目录

```text
experiments/02_finetune_exp/
├── README.md
├── run_train.py
├── run_benchmark.py
├── configs/
│   ├── train.yaml
│   ├── llamafactory_train_lora_sft.yaml
│   ├── llamafactory_train_qlora_sft.yaml
│   ├── llamafactory_train_dora_sft.yaml
│   └── llamafactory_train_galore_sft.yaml
└── reports/
```

## 推荐命令（统一 CLI）

### 1) 单次微调

```bash
# 先 dry-run：检查最终训练命令
python cli.py finetune start --config experiments/02_finetune_exp/configs/train.yaml --dry-run

# 实际执行
python cli.py finetune start --config experiments/02_finetune_exp/configs/train.yaml
```

### 2) 微调前后 benchmark（推荐答辩展示）

```bash
# 完整流程：基座评测 -> 训练 -> 微调后评测
python cli.py finetune benchmark --config experiments/02_finetune_exp/configs/train.yaml

# 跳过训练，仅对比已有模型
python cli.py finetune benchmark --config experiments/02_finetune_exp/configs/train.yaml --skip-train
```

## 配置路径

- 基础配置：`configs/base.yaml`
- 本实验覆盖：`experiments/02_finetune_exp/configs/train.yaml`
- 方法配置模板：`experiments/02_finetune_exp/configs/llamafactory_train_*.yaml`

说明：

- `finetune.train.finetune_method` 支持 `lora | qlora | dora | galore`
- 若未显式写 `finetune.train.config`，系统按 method 自动匹配对应 YAML

## 方法状态建议

- 稳定：`lora`、`qlora`
- 实验性：`dora`、`galore`

## 注意事项

- `--finetune-method` 在 CLI 中为兼容参数，推荐优先在 YAML 中配置。
- 如果直接跑脚本入口，请使用：

```bash
PYTHONPATH=. python experiments/02_finetune_exp/run_train.py --dry-run
```

（直接 `python experiments/02_finetune_exp/run_train.py` 在未设置 `PYTHONPATH` 时可能导入失败）
