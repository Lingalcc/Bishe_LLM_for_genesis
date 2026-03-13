# 03_eval_exp — 准确率评测与推理 benchmark

本目录对应两类评估：

1. 任务准确率评测（`eval accuracy`）
2. 本地推理性能 benchmark（`eval benchmark`）

## 目录

```text
experiments/03_eval_exp/
├── README.md
├── run_accuracy.py
├── configs/
│   └── accuracy.yaml
└── reports/
```

## 1) 准确率评测（推荐）

```bash
python cli.py eval accuracy --config experiments/03_eval_exp/configs/accuracy.yaml
```

输出核心指标：

- `parse_ok_rate`
- `exact_match_rate`
- `action_match_rate`

## 2) 本地推理 benchmark（吞吐/延迟/显存）

```bash
python cli.py eval benchmark \
  --backend transformers \
  --model-path model/Qwen_Qwen2.5-3B-Instruct \
  --num-samples 32 \
  --batch-size 1 \
  --output-json experiments/03_eval_exp/reports/inference_benchmark.json
```

说明：

- 该命令测试推理性能，不输出 `exact_match_rate`。
- 如需任务正确率，请使用 `eval accuracy`。

## 配置路径

- 基础配置：`configs/base.yaml`
- 本实验覆盖：`experiments/03_eval_exp/configs/accuracy.yaml`

`test.accuracy_eval.mode`：

- `api`：在线 API 评测
- `local`：本地模型评测（`transformers` 或 `vllm`）

## 功能边界

- 稳定：`mode=api`
- 实验性：`mode=local`（尤其 `backend=vllm`）

## 脚本入口说明

若必须使用脚本入口，请带 `PYTHONPATH`：

```bash
PYTHONPATH=. python experiments/03_eval_exp/run_accuracy.py
```
