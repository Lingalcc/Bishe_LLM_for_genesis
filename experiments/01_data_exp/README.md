# 01_data_exp — 数据生成实验

目标：通过 OpenAI 兼容 API 生成 Franka 机械臂 `自然语言指令 -> JSON action` 数据集。

## 目录

```text
experiments/01_data_exp/
├── README.md
├── run_api_generate.py
└── configs/
    └── api_generate.yaml
```

## 推荐命令（仓库根目录执行）

### 1) 查看动作 schema（不调用 API）

```bash
python experiments/01_data_exp/run_api_generate.py --show-actions
```

### 2) 小样本 demo（10 条）

```bash
python experiments/01_data_exp/run_api_generate.py --demo
```

### 3) 生成正式数据集（推荐走统一 CLI）

```bash
python cli.py data generate --config experiments/01_data_exp/configs/api_generate.yaml
```

### 4) 数据校验

```bash
python cli.py data calibrate
```

## 配置路径

- 基础配置：`configs/base.yaml`
- 本实验覆盖：`experiments/01_data_exp/configs/api_generate.yaml`

常改字段在 `dataset_prepare.generate`：

- `num_samples`
- `batch_size`
- `api_base` / `model`
- `api_key_env`
- `out_dir` 与输出文件名

API Key 通过环境变量提供（推荐）：

```bash
export DEEPSEEK_API_KEY="sk-..."
```

## 输出文件

默认在 `data_prepare/` 下产出：

- `genesis_franka_toolcall_alpaca.json`
- `genesis_franka_toolcall_sharegpt.json`
- `genesis_franka_toolcall_stats.json`

## 功能边界

- 已实现：数据生成、数据校验
- 计划中：`data augment` 统一 CLI 子命令（当前仓库 CLI 未提供）
