# 02 — 微调实验 (Fine-tuning Experiment)

本实验通过 [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 对基座模型进行监督微调（SFT），使其学会将自然语言指令转换为 Franka 机械臂可执行的 JSON action。

## 目录结构

```
experiments/02_finetune_exp/
├── configs/
│   └── train.yaml            # 实验覆盖配置（覆盖 configs/base.yaml）
│   └── llamafactory_train_qlora_sft.yaml # 本地维护的 QLoRA 训练配置（默认）
├── reports/                   # 评估报告输出目录
│   ├── benchmark_report.json  # 完整基准测试报告
│   ├── pre_finetune_accuracy.json
│   └── post_finetune_accuracy.json
├── run_train.py               # 单次微调启动脚本
├── run_benchmark.py           # 微调前后基准对比脚本
└── README.md
```

## 支持的微调方法

| 方法 | 说明 | 配置值 |
|------|------|--------|
| **LoRA** | 低秩适配 | `lora` |
| **QLoRA** | 4-bit 量化 + LoRA | `qlora` |
| **DoRA** | 权重分解 LoRA | `dora` |
| **GaLore** | 梯度低秩投影 | `galore` |

修改 `configs/train.yaml` 中的 `finetune_method` 即可切换：

```yaml
finetune:
  train:
    finetune_method: qlora   # lora | qlora | dora | galore
```

## 快速开始

### 1. 单次微调

```bash
# 默认 QLoRA 微调（dry-run 模式，仅打印命令）
python experiments/02_finetune_exp/run_train.py --dry-run

# 实际执行微调
python experiments/02_finetune_exp/run_train.py

# 指定不同方法
python cli.py finetune start --finetune-method qlora
python cli.py finetune start --finetune-method dora
```

### 2. 基准测试（推荐）

基准测试流程自动完成三步：评估基座模型 → 微调 → 评估微调后模型，并生成对比报告。

```bash
# 完整基准测试
python experiments/02_finetune_exp/run_benchmark.py

# 或通过 CLI
python cli.py finetune benchmark

# dry-run：仅查看配置，不执行
python cli.py finetune benchmark --dry-run

# 跳过训练，仅对比两个已有模型的准确率
python cli.py finetune benchmark --skip-train

# 只评估基座模型
python cli.py finetune benchmark --eval-only base

# 只评估微调后模型
python cli.py finetune benchmark --eval-only finetuned
```

## 评估指标

基准测试输出三组核心指标：

### 任务成功率 (Accuracy)

| 指标 | 含义 |
|------|------|
| `parse_ok_rate` | 模型输出能被解析为合法 JSON 的比例 |
| `exact_match_rate` | 输出与标注完全一致（参数值也相同）的比例 |
| `action_match_rate` | 动作类型匹配（忽略参数差异）的比例 |

### 训练显存峰值 (Peak VRAM)

训练期间通过后台线程轮询 `nvidia-smi` 采集 GPU 显存使用量，报告中包含：

- `peak_vram_mb` — 训练过程中的显存峰值
- `avg_vram_mb` — 平均显存占用
- 每张 GPU 的独立统计

### 收敛速度 (Loss Curve)

从 LLaMA Factory 输出的 `trainer_state.json` 中提取：

- `loss_curve` — 每个 step 的训练 loss
- `final_loss` / `min_loss` — 最终与最低 loss
- `total_steps` / `total_epochs` — 总步数与轮次

## 报告示例

基准测试完成后在终端打印对比摘要：

```
======================================================================
  Fine-tuning Benchmark Report
======================================================================
  Method     : lora
  Dataset    : data_prepare/genesis_franka_toolcall_alpaca.json
  Eval samples: 100

  --- Training ---
  Total Steps  : 500
  Final Loss   : 0.3421
  Min Loss     : 0.2987 (step 420)
  Peak VRAM    : 12450 MB
  Training Time: 1823 sec

  Metric                     Base Model    Finetuned      Delta
  ------------------------------------------------------------
  Accuracy (exact_match)        12.00%       78.00%    +66.00%
  Action Match Rate             35.00%       91.00%    +56.00%
  Parse OK Rate                 48.00%       97.00%    +49.00%
  Avg Latency (sec)              0.432        0.445     +0.013
  Peak VRAM (MB)                  6821         6821         —
======================================================================
```

完整 JSON 报告保存在 `reports/benchmark_report.json`。

## 配置说明

全局配置位于 `configs/base.yaml`，实验覆盖配置位于 `configs/train.yaml`。

### 微调配置 (`finetune.train`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llamafactory_dir` | `LlamaFactory` | LLaMA Factory 安装路径 |
| `config` | `experiments/02_finetune_exp/configs/llamafactory_train_qlora_sft.yaml` | 训练配置文件（独立于 LLaMA-Factory 仓库） |
| `gpus` | `"0"` | 使用的 GPU 编号（逗号分隔） |
| `finetune_method` | `qlora` | 微调方法 |
| `dry_run` | `false` | 仅打印命令，不执行 |

### 基准测试配置 (`benchmark`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_model_path` | `model/Qwen_Qwen2.5-3B-Instruct` | 基座模型路径 |
| `finetuned_model_path` | `model/my_lora_merged_model` | 微调后模型路径 |
| `dataset_file` | `data_prepare/genesis_franka_toolcall_alpaca.json` | 评估数据集 |
| `num_samples` | `100` | 评估样本数 |
| `backend` | `transformers` | 推理后端（`transformers` / `vllm`） |
| `quantization` | `null` | 推理量化（`null` / `4bit` / `8bit`） |

## 依赖

- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) — 安装在 `LLaMA-Factory/` 目录下
- PyTorch + CUDA
- transformers / vLLM（用于本地模型评估）
- nvidia-smi（用于 VRAM 监控）
