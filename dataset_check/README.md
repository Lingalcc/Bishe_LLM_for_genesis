# 数据集准备校准

本目录专注于第一阶段：构建并校准 Franka 工具调用数据集。  
该目录内脚本可以独立执行，不依赖统一式 CLI。

## 统一配置

默认读取统一配置文件：`config/pipeline_config.json`  
对应配置段：

- `dataset_prepare.generate`
- `dataset_prepare.augment`
- `dataset_prepare.calibration`

可选指定其他配置文件：

```bash
python 数据集准备校准/run_generate_dataset.py --config config/pipeline_config.json
```

命令行参数会覆盖配置文件中的同名参数。

其中 `dataset_prepare.generate.state_context_ratio` 控制训练样本中带状态注入的比例。  
状态会以 `[STATE_CONTEXT]...[/STATE_CONTEXT]` 附加到指令中。

## 目录结构

- `run_generate_dataset.py`：生成基础数据集（Alpaca + ShareGPT + 统计）
- `run_augment_dataset.py`：调用模型 API 做同义表达增强
- `run_dataset_calibration.py`：对数据集做 JSON/Schema 校准检查

## 1) 生成基础数据集

```bash
python 数据集准备校准/run_generate_dataset.py \
  --num-samples 4000 \
  --seed 42 \
  --out-dir data_prepare
```

主要输出（默认在 `data_prepare/`）：

- `genesis_franka_toolcall_alpaca.json`
- `genesis_franka_toolcall_sharegpt.json`
- `genesis_franka_toolcall_stats.json`

## 2) 数据增强（可选）

建议直接在 `config/pipeline_config.json` 里填写：

`dataset_prepare.augment.api_key`

执行增强：

```bash
python 数据集准备校准/run_augment_dataset.py \
  --input-file data_prepare/genesis_franka_toolcall_alpaca.json \
  --output-file data_prepare/genesis_franka_toolcall_alpaca_augmented.json \
  --output-sharegpt-file data_prepare/genesis_franka_toolcall_sharegpt_augmented.json \
  --num-source 800 \
  --aug-per-sample 2 \
  --model gpt-5
```

## 3) 校准检查

对生成后的数据进行合法性检查：

```bash
python 数据集准备校准/run_dataset_calibration.py \
  --dataset-file data_prepare/genesis_franka_toolcall_alpaca.json
```

严格模式（有坏样本就非零退出，便于 CI）：

```bash
python 数据集准备校准/run_dataset_calibration.py \
  --dataset-file data_prepare/genesis_franka_toolcall_alpaca.json \
  --strict
```

## 常见问题

1. 报错 `Missing API key`：请先检查 `config/pipeline_config.json` 里的 `dataset_prepare.augment.api_key` 是否已填写。
2. 增强失败重试：默认包含指数退避和重试，可调 `--max-retries`、`--timeout`。
3. 校准失败：先看 `sample_errors` 定位具体行，再回查对应 `output` JSON。
