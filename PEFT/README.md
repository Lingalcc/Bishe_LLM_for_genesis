# 微调

本目录专注于第二阶段：模型微调。  
入口为 `run_finetune.py`，直接对接 `LLaMA-Factory`。

## 统一配置

默认读取统一配置文件：`config/pipeline_config.json`  
对应配置段：`finetune.train`

入口脚本的 `--config` 用于指定“统一配置文件路径”。  
训练 YAML 请在统一配置里设置 `finetune.train.config`，或用透传方式覆盖：

```bash
python PEFT/run_finetune.py -- --config LLaMA-Factory/examples/train_lora/qwen3_lora_sft_genesis_toolcall.yaml
```

## 前置依赖

1. 已安装 `LLaMA-Factory` 运行依赖
2. 可访问训练配置中引用的数据文件
3. GPU 环境可用时再传 `--gpus`

## 目录结构

- `run_finetune.py`：启动微调（支持 dry-run 和额外参数透传）

## 快速开始

先做 dry-run 检查命令和环境：

```bash
python PEFT/run_finetune.py --dry-run --gpus 0
```

正式启动：

```bash
python PEFT/run_finetune.py --gpus 0
```

## 常用参数

- `--config`：统一配置文件路径（默认 `config/pipeline_config.json`）
- `--gpus`：设置 `CUDA_VISIBLE_DEVICES`，例如 `0` 或 `0,1`
- `--dry-run`：只打印命令，不执行训练
- `--` 后参数：透传给 `llamafactory-cli train`

示例（透传参数）：

```bash
python PEFT/run_finetune.py --gpus 0 -- --output_dir saves/genesis_sft --num_train_epochs 3
```

## 产物与检查

训练产物由配置文件中的 `output_dir` 决定。  
建议保存：

1. 训练日志
2. 最优 checkpoint 路径
3. 最终推理配置（便于测试目录复现）
