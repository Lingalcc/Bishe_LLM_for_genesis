# Google Colab 云端运行指南

本文档介绍如何在 Google Colab 上运行 Bishe LLM for Genesis 项目，包括环境配置、模型微调、评估和推理基准测试。

## 概述

本项目支持在 Google Colab 的单张 GPU 上完成 LLM 微调全流程：

- 环境自动检测与依赖安装
- Google Drive 持久化（模型、检查点、数据集）
- 运行中断后自动恢复训练
- 支持 LoRA / QLoRA / DoRA / GaLore 四种微调方法

## 前置条件

| 条件 | 说明 |
|------|------|
| Google 账号 | 用于 Colab 和 Google Drive |
| Colab GPU 运行时 | 免费版提供 T4，Colab Pro 提供 L4/A100 |
| Google Drive 空间 | 建议预留 **15-20 GB**（模型约 6GB + 训练输出 + 数据集） |
| HuggingFace Token（可选） | 下载需认证的模型时使用 |
| DeepSeek API Key（可选） | 通过 API 生成训练数据时使用 |

## 快速开始

### 第 1 步：打开 Notebook

在 GitHub 仓库中找到 `colab_run.ipynb`，点击 "Open in Colab" 徽章，或手动上传到 Colab。

### 第 2 步：选择 GPU 运行时

1. 菜单栏 → **运行时** → **更改运行时类型**
2. **硬件加速器** 选择 **GPU**
3. GPU 类型建议：
   - **T4**（免费版默认）：16GB 显存，推荐 QLoRA
   - **L4**（Pro）：24GB 显存，推荐 LoRA / QLoRA / DoRA
   - **A100**（Pro+）：40/80GB 显存，所有方法均可

### 第 3 步：配置参数

在 Notebook 的 **Cell 3（用户配置）** 中，根据需要修改：

```python
FINETUNE_METHOD = "qlora"     # 微调方法
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # 基座模型
HF_TOKEN = ""                 # HuggingFace Token（可选）
DEEPSEEK_API_KEY = ""         # DeepSeek API Key（可选）
```

> **提示**：也可以使用 Colab Secrets 安全存储 Token 和 API Key。在 Colab 左侧栏 → 🔑 密钥 中添加 `HF_TOKEN`、`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`，Notebook 会自动读取。

### 第 4 步：按顺序运行 Cell

依次运行 Cell 1 到 Cell 13，每个 Cell 都是幂等的（可安全重复运行）。

## GPU 选择指南

| GPU | 显存 | 推荐方法 | 预计训练时间 (3B, 5 epochs) | 备注 |
|-----|------|---------|---------------------------|------|
| T4 | 16 GB | QLoRA | ~40-60 min | 免费版可用，LoRA 可能 OOM |
| L4 | 24 GB | LoRA / QLoRA / DoRA | ~30-45 min | 性价比最高 |
| A100 40GB | 40 GB | 所有方法 | ~15-25 min | GaLore (全参数微调) 需要 40GB+ |
| A100 80GB | 80 GB | 所有方法 | ~15-25 min | 可尝试 7B 模型 |

### 各方法显存占用参考 (Qwen2.5-3B)

| 方法 | 显存占用 | 说明 |
|------|---------|------|
| QLoRA | ~8-10 GB | 4-bit 量化 + LoRA，显存最省 |
| LoRA | ~14-16 GB | 标准 LoRA，T4 可能紧张 |
| DoRA | ~14-16 GB | 权重分解 LoRA 变体 |
| GaLore | ~30-35 GB | 全参数微调 + 梯度低秩投影，需 A100 |

## 数据持久化

### 存储结构

Notebook 会自动将以下目录通过符号链接指向 Google Drive：

```
Google Drive/
└── MyDrive/
    └── Bishe_LLM_for_genesis/
        ├── model/              # 基座模型和微调模型
        │   ├── Qwen_Qwen2.5-3B-Instruct/  # 基座模型 (~6GB)
        │   └── qwen2.5-3b-genesis-qlora/   # 微调模型
        ├── output/             # 训练输出和检查点
        │   └── qwen2.5-3b-genesis-qlora/
        │       ├── checkpoint-100/
        │       ├── checkpoint-200/
        │       └── ...
        └── data_prepare/       # 训练数据集
            ├── genesis_franka_toolcall_alpaca.json
            └── splits/
                ├── train.json
                ├── val.json
                └── test.json
```

### 数据安全

- **模型权重**、**训练检查点**、**数据集** 全部存放在 Google Drive
- Colab 运行时重置/断开不会导致数据丢失
- 训练默认每 100 步保存一次检查点（`save_steps: 100`）

## 中断恢复

Colab 运行时可能因超时、配额耗尽等原因断开。恢复步骤：

### 1. 重新连接运行时

打开 Notebook，连接到新的 GPU 运行时。

### 2. 修改配置

在 **Cell 3** 中设置：

```python
RESUME_FROM_CHECKPOINT = True
```

### 3. 重新运行环境设置

按顺序运行 **Cell 1 → Cell 7**，这些 Cell 都是幂等的，会快速恢复环境：
- 挂载 Google Drive
- 克隆/更新代码
- 安装依赖
- 重建符号链接

### 4. 继续训练

运行 **Cell 10**，程序会自动查找最近的检查点并从该点继续训练。

> **提示**：Cell 1-7 的恢复通常只需 2-3 分钟（依赖已缓存在 Drive 中）。

## 各实验步骤说明

### 数据生成（Cell 9）

通过 DeepSeek API 生成 Franka 机械臂的 toolcall 训练数据：
- 需要设置 `DEEPSEEK_API_KEY`
- 也可以直接上传已有数据集到 Google Drive 的 `data_prepare/` 目录

### 模型微调（Cell 10）

使用 LlamaFactory 框架进行 SFT 训练：
- 自动选择与微调方法匹配的配置文件
- 训练进度和日志实时输出
- 检查点自动保存到 Google Drive

### 准确率评估（Cell 11）

评估微调模型的 toolcall 生成准确率：
- 指标：`parse_ok`（JSON 解析成功率）、`exact_match`（精确匹配率）、`action_match`（动作匹配率）
- 报告保存为 JSON 文件

### 推理基准测试（Cell 12）

对比基座模型与微调模型的推理性能：
- 指标：延迟 (latency)、吞吐量 (throughput)、显存峰值

## 常见问题

### Q: 运行时选择了 GPU 但 CUDA 不可用？

确认步骤：
1. 菜单 → 运行时 → 更改运行时类型 → GPU
2. 重启运行时后重新运行 Cell 1 检查

### Q: 训练过程中出现 OOM (Out of Memory)？

解决方案：
- 切换到 **QLoRA** 方法（显存占用最低）
- 在 LlamaFactory 配置中减小 `per_device_train_batch_size`（默认为 2，可改为 1）
- 减小 `cutoff_len`（默认 1024，可改为 512）
- 如果使用 GaLore，需要 A100 40GB+

可通过 extra_args 传递参数覆盖，例如：
```python
# 在 Cell 10 的 extra_args 列表中添加
extra_args.extend(["--", "per_device_train_batch_size=1"])
```

### Q: Google Drive 空间不足？

- Qwen2.5-3B 模型约 6GB
- 每个训练检查点约 0.5-1GB
- 建议预留 15-20GB 总空间
- 可以删除旧的 checkpoint：保留最后一个即可

### Q: 依赖安装失败？

```python
# 清除 pip 缓存后重试
!pip cache purge
# 重新运行 Cell 5
```

### Q: `llamafactory-cli` 找不到？

Notebook 中的 `cli.py` 会自动回退到 `python -m llamafactory.cli`，无需手动处理。

### Q: 如何使用 7B 模型？

在 Cell 3 中修改：
```python
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
```
注意 7B 模型需要更多显存，建议使用 QLoRA + A100。

### Q: 如何查看训练损失曲线？

训练完成后，检查 `output/<method>/training_loss.png`（LlamaFactory 的 `plot_loss: true` 会自动生成）。
